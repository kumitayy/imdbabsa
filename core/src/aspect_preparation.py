"""
Comprehensive aspect preparation module for ABSA (Aspect-Based Sentiment Analysis).

This module combines aspect extraction, context analysis, and whitelist filtering
into a unified pipeline for preparing movie review data for ABSA tasks.
"""

import os
import sys
import logging
import re
from collections import Counter
from typing import List, Set, Optional, Dict, Tuple, Any
from ast import literal_eval
import multiprocessing
from functools import partial
import time

import tqdm as tqdm_module
tqdm_module.tqdm = lambda *args, **kwargs: args[0]

import nltk
import torch
import pandas as pd
import numpy as np
from nltk.corpus import wordnet
from sentence_transformers import SentenceTransformer
import spacy

# Disable progress bars in libraries
import transformers
transformers.logging.set_verbosity_error()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import CONFIG

# Configure minimal logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress verbose output from libraries
for name in logging.root.manager.loggerDict:
    if name.startswith(('sentence_transformers', 'transformers', 'spacy', 'torch')):
        logging.getLogger(name).setLevel(logging.ERROR)
        logging.getLogger(name).propagate = False

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    logger.info(f"CUDA is available. Using GPU: {torch.cuda.get_device_name(0)}")
else:
    logger.info("CUDA is not available. Using CPU.")

# Load models
bert_model = SentenceTransformer("all-MiniLM-L6-v2")
# Reduce SpaCy model size for speed - we only need sentence segmentation
nlp = spacy.load("en_core_web_sm", disable=["ner", "tagger", "lemmatizer", "attribute_ruler"])

# Добавить после импортов и настройки логирования
original_stderr = sys.stderr

class DummyFile:
    def write(self, x): pass
    def flush(self): pass

class suppress_output:
    def __enter__(self):
        self.old_stderr = sys.stderr
        sys.stderr = DummyFile()
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr = self.old_stderr

# ======= ASPECT EXTRACTION FUNCTIONS ======= #

def is_noise(aspect: str, min_len: int = 4, stopwords: Optional[Set[str]] = None) -> bool:
    """
    Determine if an aspect is considered noise based on criteria.
    
    Args:
        aspect (str): The aspect string to evaluate
        min_len (int): Minimum character length for valid aspects
        stopwords (Optional[Set[str]]): Set of words to be considered as noise
        
    Returns:
        bool: True if the aspect is considered noise, False otherwise
        
    Raises:
        Exception: If there is an error processing the aspect
    """
    if stopwords is None:
        stopwords = {
            "thing", "stuff", "bit", "part", "scene", "element", "everything",
            "nothing", "something", "aspect", "film", "movie", "story"
        }
    aspect_lower = aspect.lower()
    if len(aspect_lower) < min_len:
        return True
    if aspect_lower in stopwords:
        return True
    if aspect_lower.isdigit():
        return True
    if re.fullmatch(r"[a-zA-Z]{1,2}", aspect):
        return True
    return False


def filter_semantic_noise(aspect_lists: List[List[str]], min_len: int = 4) -> List[List[str]]:
    """
    Filter out noisy aspects from lists based on semantic criteria.
    
    Args:
        aspect_lists (List[List[str]]): List of aspect lists to filter
        min_len (int): Minimum character length for valid aspects
        
    Returns:
        List[List[str]]: Lists of aspects with noise removed
        
    Raises:
        Exception: If there is an error during filtering
    """
    removed_counter = Counter()
    refined_lists = []
    for aspects in aspect_lists:
        clean = []
        for a in aspects:
            if is_noise(a, min_len=min_len):
                removed_counter[a] += 1
            else:
                clean.append(a)
        refined_lists.append(clean)
    return refined_lists


def filter_aspects(aspect_lists: List[List[str]], min_freq: int = 50, stop_aspects: Optional[Set[str]] = None) -> List[List[str]]:
    """
    Filter aspects based on frequency and stop words.
    
    Args:
        aspect_lists (List[List[str]]): List of aspect lists to filter
        min_freq (int): Minimum frequency threshold for aspects to keep
        stop_aspects (Optional[Set[str]]): Set of aspects to exclude regardless of frequency
        
    Returns:
        List[List[str]]: Lists of filtered aspects
        
    Raises:
        Exception: If there is an error during filtering
    """
    if stop_aspects is None:
        stop_aspects = {"film", "movie", "scene", "story", "part", "thing"}
    
    all_aspects = [aspect for aspects in aspect_lists for aspect in aspects]
    aspect_counts = Counter(all_aspects)
    filtered = [
        [a for a in aspects if aspect_counts[a] >= min_freq and a not in stop_aspects]
        for aspects in aspect_lists
    ]
    return filtered


def normalize_aspects(aspect_lists: List[List[str]], similarity_threshold: float = 0.85, use_gpu: bool = True) -> List[List[str]]:
    """
    Normalize aspects by mapping them to canonical forms using semantic similarity.
    
    Args:
        aspect_lists (List[List[str]]): List of aspect lists to normalize
        similarity_threshold (float): Minimum similarity score to map an aspect to a canonical form
        use_gpu (bool): Whether to use GPU for computing embeddings
        
    Returns:
        List[List[str]]: Lists of normalized aspects
        
    Raises:
        Exception: If there is an error during normalization
        RuntimeError: If GPU is requested but not available
    """
    aspect_synonyms = {
        "acting": ["performance", "actor", "actors"],
        "soundtrack": ["music", "score", "background music"],
        "cinematography": ["visuals", "photography", "camera work"],
        "plot": ["storyline", "narrative"],
        "characters": ["protagonist", "antagonist", "roles"]
    }
    cluster_keys = list(aspect_synonyms.keys())
    device = "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
    cluster_embeddings = bert_model.encode(cluster_keys, convert_to_numpy=True, device=device)
    
    all_aspects = list(set(a for aspects in aspect_lists for a in aspects))
    
    batch_size = 512 if CUDA_AVAILABLE else 128
    aspect_embeddings = []
    
    for i in range(0, len(all_aspects), batch_size):
        batch = all_aspects[i:i+batch_size]
        batch_embeddings = bert_model.encode(
            batch, 
            convert_to_numpy=True, 
            device=device, 
            show_progress_bar=False
        )
        aspect_embeddings.append(batch_embeddings)
    
    aspect_embeddings = np.vstack(aspect_embeddings)
    
    similarities = np.dot(aspect_embeddings, cluster_embeddings.T) / (
        np.linalg.norm(aspect_embeddings, axis=1, keepdims=True) @ 
        np.linalg.norm(cluster_embeddings, axis=1, keepdims=True).T
    )
    
    aspect_to_cluster = {}
    for i, aspect in enumerate(all_aspects):
        max_sim_idx = int(np.argmax(similarities[i]))
        if similarities[i, max_sim_idx] >= similarity_threshold:
            aspect_to_cluster[aspect] = cluster_keys[max_sim_idx]
        else:
            aspect_to_cluster[aspect] = aspect
    
    normalized_lists = [[aspect_to_cluster[a] for a in aspects] for aspects in aspect_lists]
    return normalized_lists


def merge_similar_aspects(aspect_lists: List[List[str]], similarity_threshold: float = 0.8, use_gpu: bool = True) -> List[List[str]]:
    """
    Merge similar aspects based on embedding similarity.
    
    Args:
        aspect_lists (List[List[str]]): List of aspect lists to process
        similarity_threshold (float): Minimum similarity score to merge two aspects
        use_gpu (bool): Whether to use GPU for computing embeddings
        
    Returns:
        List[List[str]]: Lists of merged aspects
        
    Raises:
        Exception: If there is an error during merging
        RuntimeError: If GPU is requested but not available
    """
    all_aspects = list(set(a for aspects in aspect_lists for a in aspects))
    
    device = "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
    
    batch_size = 512 if CUDA_AVAILABLE else 128
    embeddings = []
    
    for i in range(0, len(all_aspects), batch_size):
        batch = all_aspects[i:i+batch_size]
        batch_embeddings = bert_model.encode(
            batch,
            convert_to_numpy=True,
            device=device,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings)
    
    if len(embeddings) > 1:
        embeddings = np.vstack(embeddings)
    else:
        embeddings = embeddings[0]
    
    aspect_to_vec = dict(zip(all_aspects, embeddings))
    
    similarity_matrix = np.dot(embeddings, embeddings.T) / (
        np.linalg.norm(embeddings, axis=1, keepdims=True) @ 
        np.linalg.norm(embeddings, axis=1, keepdims=True).T
    )
    
    merged_map = {}
    for i, a1 in enumerate(all_aspects):
        if a1 in merged_map:
            continue
        merged_map[a1] = a1
        
        similar_indices = np.where(similarity_matrix[i] > similarity_threshold)[0]
        for j in similar_indices:
            if i != j:
                a2 = all_aspects[j]
                if a2 not in merged_map:
                    merged_map[a2] = a1
    
    merged_lists = [[merged_map[a] for a in aspects if a in merged_map] for aspects in aspect_lists]
    return merged_lists


def extract_aspect_contexts_batch(reviews, all_aspects, use_gpu=True):
    """
    Extract context sentences for each aspect in batches of reviews.
    
    For each review-aspect pair, this function:
    1. Identifies sentences containing the aspect (direct string matching)
    2. For aspects not found, uses semantic similarity to find relevant sentences
    3. Detects potential sentiment contrast signals in the context
    4. Returns the contexts and contrast information for each aspect
    
    Args:
        reviews (List[str]): List of review texts
        all_aspects (List[List[str]]): List of aspect lists for each review
        use_gpu (bool): Whether to use GPU for embedding calculations
        
    Returns:
        Tuple[List[Dict], List[Dict]]: Contexts and contrast information for each aspect
        
    Raises:
        Exception: If there is an error during processing
    """
    contexts_batch = []
    contrasts_batch = []
    
    device = "cuda" if use_gpu and CUDA_AVAILABLE else "cpu"
    contrast_indicators = [
        "but", "however", "although", "though", "despite", "in spite", "except",
        "while", "nevertheless", "on the other hand", "unfortunately", "still",
        "otherwise", "that said", "yet", "even though", "other than that"
    ]
    
    for text, aspects in zip(reviews, all_aspects):
        if not aspects:
            contexts_batch.append({})
            contrasts_batch.append({})
            continue
            
        doc = nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        lower_sentences = [s.lower() for s in sentences]
        
        if not sentences:
            contexts_batch.append({aspect: [] for aspect in aspects})
            contrasts_batch.append({aspect: {'contrast_signal': 0} for aspect in aspects})
            continue
        
        sentences_with_contrast = set()
        for i, sentence_lower in enumerate(lower_sentences):
            if any(indicator in sentence_lower for indicator in contrast_indicators):
                sentences_with_contrast.add(i)
        
        aspect_contexts = {}
        aspect_contrasts = {}
        
        aspects_found = set()
        for aspect in aspects:
            aspect_lower = aspect.lower()
            relevant_sentences = []
            matching_sentences = set()
            
            for i, sentence_lower in enumerate(lower_sentences):
                if aspect_lower in sentence_lower:
                    relevant_sentences.append(sentences[i])
                    matching_sentences.add(i)
            
            if relevant_sentences:
                aspect_contexts[aspect] = relevant_sentences
                aspects_found.add(aspect)
                
                has_contrast = bool(matching_sentences & sentences_with_contrast)
                aspect_contrasts[aspect] = {'contrast_signal': 1 if has_contrast else 0}
        
        aspects_not_found = set(aspects) - aspects_found
        if aspects_not_found and sentences:
            sentence_embeddings = bert_model.encode(
                sentences, 
                convert_to_numpy=True, 
                device=device, 
                show_progress_bar=False
            )
            
            aspect_embeddings = bert_model.encode(
                list(aspects_not_found), 
                convert_to_numpy=True, 
                device=device, 
                show_progress_bar=False
            )
            
            similarities = np.dot(aspect_embeddings, sentence_embeddings.T) / (
                np.linalg.norm(aspect_embeddings, axis=1, keepdims=True) @ 
                np.linalg.norm(sentence_embeddings, axis=1, keepdims=True).T
            )
            
            for i, aspect in enumerate(aspects_not_found):
                top_indices = similarities[i].argsort()[-2:][::-1]
                relevant_sentences = []
                
                if similarities[i, top_indices[0]] > 0.4:
                    matching_sentences = set()
                    for idx in top_indices:
                        relevant_sentences.append(sentences[idx])
                        matching_sentences.add(idx)
                        
                    has_contrast = bool(matching_sentences & sentences_with_contrast)
                    aspect_contrasts[aspect] = {'contrast_signal': 1 if has_contrast else 0}
                else:
                    aspect_contrasts[aspect] = {'contrast_signal': 0}
                    
                aspect_contexts[aspect] = relevant_sentences
        
        contexts_batch.append(aspect_contexts)
        contrasts_batch.append(aspect_contrasts)
    
    return contexts_batch, contrasts_batch


def process_aspects(df: pd.DataFrame, aspect_column: str = "aspects", use_gpu: bool = True) -> pd.DataFrame:
    logger.info("Processing aspects: started")
    df = df.copy(deep=True)
    df[aspect_column] = df[aspect_column].apply(eval)
    
    logger.info("Filtering aspects by frequency")
    df["filtered_aspects"] = filter_aspects(df[aspect_column])
    
    logger.info("Filtering semantic noise")
    df["semantic_filtered_aspects"] = filter_semantic_noise(df["filtered_aspects"])
    
    logger.info("Normalizing aspects")
    with suppress_output():
        df["normalized_aspects"] = normalize_aspects(df["semantic_filtered_aspects"], use_gpu=use_gpu)
    
    logger.info("Merging similar aspects")
    with suppress_output():
        df["final_aspects"] = merge_similar_aspects(df["normalized_aspects"], use_gpu=use_gpu)
    
    # Удаление промежуточных столбцов и столбца score, если он есть
    columns_to_keep = ["review", "sentiment", "final_aspects"]
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    logger.info("Processing aspects: completed")
    return df


# ======= WHITELIST FILTERING FUNCTIONS ======= #

def load_whitelist(whitelist_path: str) -> Set[str]:
    """
    Load the whitelist from the specified file, normalize all entries to lowercase,
    and return a set of whitelist aspects.

    Args:
        whitelist_path (str): Path to the whitelist file.

    Returns:
        Set[str]: A set of normalized whitelist aspects.
        
    Raises:
        FileNotFoundError: If the whitelist file does not exist.
        Exception: If there is an error loading the whitelist.
    """
    try:
        with open(whitelist_path, "r", encoding="utf-8") as f:
            whitelist = {line.strip().lower() for line in f if line.strip()}
            
        logger.info(f"Loaded whitelist with {len(whitelist)} aspects")
        if not whitelist:
            logger.warning("Whitelist is empty! This will filter out all aspects.")
        return whitelist
    except Exception as e:
        logger.error(f"Error loading whitelist: {e}")
        return set()


def review_contains_whitelisted_aspect(aspect_list: List[str], whitelist: Set[str]) -> bool:
    """
    Check if the review's aspect list contains at least one aspect from the whitelist.

    Args:
        aspect_list (List[str]): List of aspects.
        whitelist (Set[str]): Set of whitelisted aspects.

    Returns:
        bool: True if at least one aspect from the whitelist is found, False otherwise.
        
    Raises:
        Exception: If there is an error processing aspects.
    """
    try:
        normalized_aspects = {aspect.lower() for aspect in aspect_list if isinstance(aspect, str)}
        return bool(normalized_aspects & whitelist)
    except Exception as e:
        logger.error(f"Error processing aspects: {e}")
        return False


def filter_aspects_by_whitelist(aspects: List[str], whitelist: Set[str]) -> List[str]:
    """
    Filter aspects to keep only those in the whitelist.
    
    Args:
        aspects (List[str]): List of aspects.
        whitelist (Set[str]): Set of whitelisted aspects.
        
    Returns:
        List[str]: List of filtered aspects.
        
    Raises:
        Exception: If there is an error processing aspects.
    """
    try:
        filtered_aspects = []
        seen = set()
        for aspect in aspects:
            if not isinstance(aspect, str):
                logger.warning(f"Non-string aspect found: {type(aspect)}")
                continue
                
            aspect_lower = aspect.lower().strip()
            
            if aspect_lower in whitelist and aspect_lower not in seen:
                filtered_aspects.append(aspect) 
                seen.add(aspect_lower)
        return filtered_aspects
    except Exception as e:
        logger.error(f"Error processing aspects: {e}")
        return []


def filter_dataset_by_whitelist(df: pd.DataFrame, whitelist: Set[str]) -> pd.DataFrame:
    """
    Filter DataFrame by retaining only rows with whitelisted aspects and
    keep only whitelisted aspects in the 'final_aspects' column.

    Args:
        df (pd.DataFrame): DataFrame to filter
        whitelist (Set[str]): Set of whitelisted aspects

    Returns:
        pd.DataFrame: Filtered DataFrame
        
    Raises:
        KeyError: If 'final_aspects' column is not found in the dataset.
        Exception: If there is an error during filtering.
    """
    if "final_aspects" not in df.columns:
        logger.error("Column 'final_aspects' not found in the dataset.")
        return df

    df = df.copy(deep=True)
    df['final_aspects'] = df['final_aspects'].apply(
        lambda aspects: filter_aspects_by_whitelist(aspects, whitelist)
    )
    
    df_filtered = df[df['final_aspects'].apply(lambda x: len(x) > 0)]
    
    return df_filtered


# ======= INTEGRATED PIPELINE FUNCTIONS ======= #

def prepare_aspects_with_whitelist(input_path: str, output_path: str, whitelist_path: str, use_gpu: bool = True) -> None:
    """
    Process a dataset with complete aspect extraction and whitelist filtering.
    
    Args:
        input_path (str): Path to input CSV file
        output_path (str): Path to save the processed CSV file
        whitelist_path (str): Path to the whitelist file
        use_gpu (bool): Whether to use GPU for embedding calculations
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If input file or whitelist does not exist
        Exception: If there is an error during processing
    """
    try:
        # Обработка аспектов
        df = pd.read_csv(input_path)
        df = process_aspects(df, use_gpu=use_gpu)
        
        # Фильтрация по белому списку
        whitelist = load_whitelist(whitelist_path)
        df_filtered = filter_dataset_by_whitelist(df, whitelist)
        
        # Извлечение контекстов ТОЛЬКО для отфильтрованных аспектов
        logger.info("Extracting aspect contexts for whitelisted aspects")
        with suppress_output():
            contexts_batch, contrasts_batch = extract_aspect_contexts_batch(
                df_filtered["review"].tolist(), 
                df_filtered["final_aspects"].tolist(), 
                use_gpu=use_gpu
            )
        
        df_filtered["aspect_contexts"] = contexts_batch
        df_filtered["aspect_contrasts"] = contrasts_batch
        
        df_filtered.to_csv(output_path, index=False)
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {e}")


def process_split(split_name: str, use_gpu: bool = True) -> None:
    """
    Process a single data split (train/val/test) through the complete pipeline.
    
    Args:
        split_name (str): Name of the split ('train', 'val', 'test')
        use_gpu (bool): Whether to use GPU for embedding calculations
        
    Returns:
        None
        
    Raises:
        FileNotFoundError: If input files do not exist
        Exception: If there is an error during processing
    """
    input_file = f"{split_name}_data.csv"
    output_file = f"{split_name}_data_processed.csv"
    
    input_path = os.path.join(CONFIG["unprocessed_aspects_path"], input_file)
    output_path = os.path.join(CONFIG["useful_aspects_path"], output_file)
    whitelist_path = CONFIG["whitelist_path"]
    
    prepare_aspects_with_whitelist(input_path, output_path, whitelist_path, use_gpu=use_gpu)


def run_aspect_preparation(use_gpu: bool = True) -> None:
    os.makedirs(CONFIG["useful_aspects_path"], exist_ok=True)
    
    splits = ["train", "val", "test"]
    logger.info(f"Starting aspect preparation pipeline")
    
    for split in splits:
        logger.info(f"Processing split: {split}")
        process_split(split, use_gpu=use_gpu)
        
    logger.info("Aspect preparation pipeline completed")