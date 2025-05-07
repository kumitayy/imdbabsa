import os
import sys
import json
import logging
from typing import Dict, List, Optional
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

import torch
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from config.config import CONFIG

MODEL_NAME = "upstage/SOLAR-10.7B-Instruct-v1.0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 1024
BATCH_SIZE = 4

# Конфигурация для 4-битного квантования
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

# Глобальные переменные для модели и токенизатора
tokenizer = None
model = None

def load_model_if_needed():
    """Загружает модель и токенизатор при необходимости."""
    global tokenizer, model
    
    if tokenizer is None or model is None:
        logger.info(f"Loading model {MODEL_NAME} on {DEVICE} with 4-bit quantization")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        logger.info("Model loaded successfully")

        # Настройка токенизатора
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
    return tokenizer, model

def make_prompt(review_text: str, aspect: str, target_sentiment: str = None) -> str:
    """
    Creates a prompt for SOLAR-10.7B-Instruct-v1.0 to generate aspect-specific sentiment analysis with three categories.
    
    Args:
        review_text (str): The original movie review text
        aspect (str): The aspect to analyze sentiment for
        target_sentiment (str, optional): Target sentiment to generate (positive/negative/neutral)
            If None, the system will analyze without bias
    
    Returns:
        str: Formatted prompt for the language model
    """
    return f"""<human>: You are a sentiment analysis expert specializing in movie reviews. Your task is to analyze the sentiment toward a specific aspect in a movie review.

Movie Review: "{review_text}"

What is the sentiment toward the aspect "{aspect}" in this review? 

Respond with EXACTLY ONE of these words only: POSITIVE, NEGATIVE, or NEUTRAL.

Guidelines:
- POSITIVE: The review expresses positive opinions about this specific aspect
- NEGATIVE: The review expresses negative opinions about this specific aspect
- NEUTRAL: The aspect is mentioned without clear sentiment, mentioned only factually, or mentioned as a supporting element without its own sentiment

Remember to respond with ONLY ONE WORD.

<assistant>: """


def normalize_sentiment(text: str) -> str:
    """
    Normalizes the model's text response into one of three values: positive, negative, or neutral.

    Args:
        text (str): Model's text response.

    Returns:
        str: Normalized sentiment value ("positive", "negative", or "neutral").
    """
    text = text.lower().strip()
    logger.debug(f"Normalizing text: '{text}'")
    
    # Check for exact matches first
    if any(pattern in text for pattern in ["positive", "positive.", "\"positive\"", "positive\""]):
        return "positive"
    if any(pattern in text for pattern in ["negative", "negative.", "\"negative\"", "negative\""]):
        return "negative"
    if any(pattern in text for pattern in ["neutral", "neutral.", "\"neutral\"", "neutral\""]):
        return "neutral"
    
    # Check for word beginnings
    if text.startswith("posit"):
        return "positive"
    if text.startswith("negat"):
        return "negative"
    if text.startswith("neutr"):
        return "neutral"
    
    # Check for phrases
    positive_indicators = ["positive", "good", "great", "excellent", "favorable", "liked", "enjoyed"]
    negative_indicators = ["negative", "bad", "poor", "terrible", "unfavorable", "disliked", "disappointed"]
    neutral_indicators = ["neutral", "unclear", "ambiguous", "factual", "neither", "ambivalent", "objective"]
    
    first_few_words = ' '.join(text.split()[:5])
    
    # Check the first few words for indicators
    for indicator in positive_indicators:
        if indicator in first_few_words:
            return "positive"
            
    for indicator in negative_indicators:
        if indicator in first_few_words:
            return "negative"
            
    for indicator in neutral_indicators:
        if indicator in first_few_words:
            return "neutral"
    
    # Check full text if necessary
    for indicator in positive_indicators:
        if indicator in text:
            return "positive"
            
    for indicator in negative_indicators:
        if indicator in text:
            return "negative"
            
    for indicator in neutral_indicators:
        if indicator in text:
            return "neutral"
    
    logger.warning(f"Could not normalize sentiment from: '{text}'. Defaulting to neutral.")
    return "neutral"


def batch_generate(prompts: List[str]) -> List[str]:
    """
    Generates sentiments for a batch of prompts using the SOLAR-10.7B-Instruct-v1.0 model.
    
    This function processes multiple prompts at once for efficiency and is
    optimized for SOLAR's capabilities in sentiment classification tasks.

    Args:
        prompts (List[str]): List of formatted prompts for generation.

    Returns:
        List[str]: List of normalized sentiments ("positive", "negative", or "neutral").
    """
    try:
        # Загружаем модель при первом использовании
        tokenizer, model = load_model_if_needed()
        
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=MAX_LENGTH
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=16,
                do_sample=True,  # Enable sampling for SOLAR
                temperature=0.3,  # SOLAR works well with slightly higher temperature
                top_p=0.95,       # More precise filtering of tokens
                repetition_penalty=1.1,  # Discourage repetitive outputs
                num_return_sequences=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
            
        generated_texts = [
            tokenizer.decode(output[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip() 
            for output in outputs
        ]
        
        normalized_sentiments = [normalize_sentiment(text) for text in generated_texts]
        
        return normalized_sentiments
    
    except Exception as e:
        logger.error(f"Failed batch inference: {e}")
        logger.exception("Exception details:")
        return ["neutral"] * len(prompts)


def generate_synthetic_data(max_samples_per_split: Optional[Dict[str, int]] = {"train": 60000, "val": 10000, "test": 10000}) -> None:
    """
    Generates synthetic sentiment data for aspect-based sentiment analysis using SOLAR-10.7B-Instruct-v1.0.
    
    This function:
    1. Reads processed review data with aspects
    2. Extracts all unique review-aspect pairs
    3. Generates sentiment (positive/negative/neutral) for each pair using SOLAR-10.7B
    4. Filters out neutral sentiments to produce high-quality binary classifications
    5. Saves the results for training a RoBERTa model
    
    The function processes each split (train/val/test) separately, maintaining
    the data distribution and preventing leakage between splits.

    Args:
        max_samples_per_split (Optional[Dict[str, int]]): Maximum number of samples for each split.

    Returns:
        None
        
    Raises:
        FileNotFoundError: If input files do not exist.
        RuntimeError: If there is an error with the model or tokenizer.
        Exception: If there is an error generating or saving data.
    """
    synthetic_sentiments_path = CONFIG.get("synthetic_sentiments_path")
    if not synthetic_sentiments_path:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        synthetic_sentiments_path = os.path.join(base_dir, "data", "synthetic_sentiments")
    os.makedirs(synthetic_sentiments_path, exist_ok=True)
    
    split_names = ["train", "val", "test"]
    for split in split_names:
        input_csv = os.path.join(CONFIG["useful_aspects_path"], f"{split}_data_processed.csv")
        output_csv = os.path.join(synthetic_sentiments_path, f"{split}_data.csv")
        filtered_output_csv = os.path.join(synthetic_sentiments_path, f"{split}_data_filtered.csv")
        meta_output = os.path.join(synthetic_sentiments_path, f"{split}_meta.json")

        logger.info(f"Processing split: {split}")
        logger.info(f"Loading data from {input_csv}")

        df = pd.read_csv(input_csv)
        df["final_aspects"] = df["final_aspects"].apply(eval)

        logger.info("Expanding review-aspect pairs...")
        pairs = []
        for _, row in df.iterrows():
            for aspect in row["final_aspects"]:
                aspect_contexts = eval(row["aspect_contexts"]) if isinstance(row["aspect_contexts"], str) else row["aspect_contexts"]
                if aspect in aspect_contexts and aspect_contexts[aspect]:
                    pairs.append({
                        "review": row["review"],
                        "aspect": aspect,
                        "context": " ".join(aspect_contexts[aspect])
                    })

        pairs_df = pd.DataFrame(pairs)
        
        if len(pairs_df) == 0:
            logger.warning(f"No valid aspect-context pairs found for split: {split}")
            continue
            
        max_pairs = max_samples_per_split.get(split, None)
        if max_pairs and len(pairs_df) > max_pairs:
            pairs_df = pairs_df.sample(n=max_pairs, random_state=42).reset_index(drop=True)
            logger.info(f"Sampled {max_pairs} aspect-level pairs for split: {split}")
        
        pairs_df["prompt"] = pairs_df.apply(lambda row: make_prompt(row["context"], row["aspect"]), axis=1)
        logger.info(f"Created {len(pairs_df)} prompts for sentiment generation")

        results = []
        positive_count = 0
        negative_count = 0
        neutral_count = 0
        
        logger.info(f"Generating sentiments for {split} split...")
        
        # Custom progress tracking, but logging only at 10% intervals
        total_pairs = len(pairs_df)
        processed_pairs = 0
        last_logged_percentage = 0
        start_time = time.time()
        
        # Calculate size of 10% for progress updates
        ten_percent = total_pairs // 10
        if ten_percent == 0:  # Handle very small datasets
            ten_percent = 1
        
        for i in range(0, total_pairs, BATCH_SIZE):
            batch = pairs_df.iloc[i:i+BATCH_SIZE]
            sentiments = batch_generate(batch["prompt"].tolist())
            
            for j, sentiment in enumerate(sentiments):
                try:
                    results.append({
                        "review": batch.iloc[j]["review"],
                        "aspect": batch.iloc[j]["aspect"],
                        "context": batch.iloc[j]["context"],
                        "sentiment": sentiment
                    })
                    
                    if sentiment == "positive":
                        positive_count += 1
                    elif sentiment == "negative":
                        negative_count += 1
                    else:
                        neutral_count += 1
                        
                except Exception as e:
                    logger.error(f"Error appending result: {e}")
            
            # Update progress counter
            processed_pairs += len(batch)
            
            # Only log at 10% intervals
            current_percentage = (processed_pairs * 10) // total_pairs * 10  # Round to nearest 10%
            if current_percentage > last_logged_percentage:
                elapsed = time.time() - start_time
                rate = processed_pairs / elapsed if elapsed > 0 else 0
                eta = (total_pairs - processed_pairs) / rate if rate > 0 else 0
                
                logger.info(f"Progress: {processed_pairs}/{total_pairs} ({current_percentage}%) - "
                            f"pos: {positive_count}, neg: {negative_count}, neutral: {neutral_count} - "
                            f"ETA: {eta:.1f}s")
                last_logged_percentage = current_percentage
        
        logger.info(f"Completed sentiment generation for {split} split")
        out_df = pd.DataFrame(results)
        
        sentiment_counts = Counter(out_df["sentiment"])
        logger.info(f"Final sentiment distribution for {split}: {dict(sentiment_counts)}")
        
        # Save complete dataset with all three sentiment classes
        out_df.to_csv(output_csv, index=False)
        logger.info(f"Saved complete dataset to {output_csv}")
        
        # Filter out neutral sentiments and save filtered dataset
        filtered_df = out_df[out_df["sentiment"].isin(["positive", "negative"])].reset_index(drop=True)
        filtered_sentiment_counts = Counter(filtered_df["sentiment"])
        logger.info(f"Filtered sentiment distribution for {split}: {dict(filtered_sentiment_counts)}")
        
        # Balance binary classes if needed in training set
        if split == "train" and len(filtered_df) > 0:
            min_class = min(filtered_sentiment_counts.values())
            max_class = max(filtered_sentiment_counts.values())
            if max_class / min_class > 3:
                logger.info("Balancing classes due to high imbalance")
                filtered_df = balance_classes(filtered_df)
                balanced_sentiment_counts = Counter(filtered_df["sentiment"])
                logger.info(f"Balanced class distribution: {dict(balanced_sentiment_counts)}")
        
        filtered_df.to_csv(filtered_output_csv, index=False)
        logger.info(f"Saved filtered dataset to {filtered_output_csv}")

        metadata = {
            "split": split,
            "model": MODEL_NAME,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "num_samples_total": len(out_df),
            "num_samples_filtered": len(filtered_df),
            "full_distribution": dict(sentiment_counts),
            "filtered_distribution": dict(filtered_sentiment_counts if len(filtered_df) > 0 else {})
        }

        with open(meta_output, "w") as f:
            json.dump(metadata, f, indent=4)
        logger.info(f"Saved metadata to {meta_output}")


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Balances the classes in the dataset by downsampling the majority class.
    
    This function ensures that the dataset has approximately equal numbers
    of positive and negative examples, which is important for training
    a balanced classifier.

    Args:
        df (pd.DataFrame): DataFrame with sentiment labels.
        
    Returns:
        pd.DataFrame: Balanced DataFrame.
        
    Raises:
        ValueError: If the DataFrame is empty or missing required columns.
    """
    if len(df) == 0 or "sentiment" not in df.columns:
        raise ValueError("DataFrame must contain samples with 'sentiment' column")
    
    sentiment_counts = Counter(df["sentiment"])
    min_class_count = min(sentiment_counts.values())
    
    balanced_samples = []
    for sentiment in sentiment_counts.keys():
        class_samples = df[df["sentiment"] == sentiment]
        
        if len(class_samples) > min_class_count:
            sampled = class_samples.sample(n=min_class_count, random_state=42)
            balanced_samples.append(sampled)
        else:
            balanced_samples.append(class_samples)
    
    balanced_df = pd.concat(balanced_samples).reset_index(drop=True)
    return balanced_df


if __name__ == "__main__":
    logger.info("Starting synthetic sentiment generation with SOLAR-10.7B-Instruct-v1.0")
    generate_synthetic_data()
    logger.info("Sentiment generation complete")