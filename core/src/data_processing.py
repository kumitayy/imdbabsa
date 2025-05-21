# Standard library imports
import os
import sys
import logging
import re
from typing import List, Optional

# Third-party imports
import nltk
import pandas as pd
import contractions
from tqdm import tqdm
from nltk import pos_tag, word_tokenize

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load required resources
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)


def load_imdb_data(base_path: str) -> pd.DataFrame:
    """
    Load IMDB data from the given base path.
    
    Args:
        base_path (str): The base path to the IMDB dataset.
    
    Returns:
        pd.DataFrame: A DataFrame containing the IMDB dataset.
        
    Raises:
        FileNotFoundError: If the base path does not exist.
        Exception: If there is an error loading the data.
    """
    logger.info("Creating IMDB dataset...")
    datasets = []

    try:
        # Loading positive and negative reviews from training and test sets
        logger.info("Loading positive and negative reviews...")
        total_processed = 0
        
        for dataset_type in ['train', 'test']:
            for sentiment in ['pos', 'neg']:
                folder_path = os.path.join(base_path, dataset_type, sentiment)
                label = 1 if sentiment == 'pos' else 0
                
                # Count files for progress display
                files = os.listdir(folder_path)
                total_files = len(files)
                logger.info(f"Loading {total_files} {sentiment} reviews from {dataset_type} folder...")
                
                for file_name in tqdm(files, desc=f"{dataset_type}/{sentiment}"):
                    file_path = os.path.join(folder_path, file_name)
                    score = int(file_name.split('_')[1].split('.')[0])

                    with open(file_path, 'r', encoding='utf-8') as file:
                        review = file.read()
                        datasets.append({'review': review, 'sentiment': label, 'score': score})
                
                total_processed += total_files
                logger.info(f"Loaded {total_files} {sentiment} reviews from {dataset_type} folder")
        
        # Loading unsupervised reviews
        logger.info("Loading unsupervised reviews...")
        unsup_folder_path = os.path.join(base_path, 'train', 'unsup')
        unsup_files = os.listdir(unsup_folder_path)
        unsup_count = len(unsup_files)
        logger.info(f"Found {unsup_count} unsupervised reviews...")
        
        for file_name in tqdm(unsup_files, desc="unsup"):
            file_path = os.path.join(unsup_folder_path, file_name)

            with open(file_path, 'r', encoding='utf-8') as file:
                review = file.read()
                datasets.append({'review': review, 'sentiment': 'unsupervised', 'score': 'unsupervised'})
        
        total_processed += unsup_count
        logger.info(f"Loaded {unsup_count} unsupervised reviews")
        logger.info(f"Total files processed: {total_processed}")

    except Exception as e:
        logger.error(f"Error loading IMDB data: {e}")
        return None

    logger.info(f"IMDB dataset created with {len(datasets)} reviews")
    return pd.DataFrame(datasets)


def remove_duplicates(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Removes duplicates from the DataFrame.
    
    Args:
        df (pd.DataFrame): The DataFrame to remove duplicates from
    
    Returns:
        Optional[pd.DataFrame]: The DataFrame with duplicates removed or None if an error occurs
        
    Raises:
        Exception: If there is an error removing duplicates
    """
    try:
        initial_rows = df.shape[0]
        df = df.drop_duplicates()
        removed = initial_rows - df.shape[0]
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        return df
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        return None


def expand_contractions(text: str) -> str:
    """
    Expands contractions in the text.
    
    Args:
        text (str): Text with contractions
        
    Returns:
        str: Text with expanded contractions
        
    Raises:
        Exception: If there is an error expanding contractions
    """
    return contractions.fix(text)


def handle_negations(text: str) -> str:
    """
    Handles negations in the text by joining negation words with following words.
    
    Args:
        text (str): The text to handle negations in
    
    Returns:
        str: The text with negations handled
        
    Raises:
        Exception: If there is an error handling negations
    """
    negations = {"not", "no", "n't", "never"}
    
    try:
        words = word_tokenize(text)
        processed_words = []
        i = 0

        while i < len(words):
            if words[i].lower() in negations and i + 1 < len(words):
                if words[i + 1].isalpha():
                    processed_words.append(words[i] + "_" + words[i + 1])
                    i += 2
                else:
                    processed_words.append(words[i])
                    i += 1
            else:
                processed_words.append(words[i])
                i += 1

        return " ".join(processed_words)
    except Exception as e:
        logger.error(f"Error handling negations: {e}")
        return text


def clean_text(text: str, remove_numbers: bool = True) -> str:
    """
    Cleans the text by removing HTML tags, special characters, and extra whitespace.
    
    Args:
        text (str): The text to clean
        remove_numbers (bool): Whether to remove numbers from the text
    
    Returns:
        str: The cleaned text
        
    Raises:
        Exception: If there is an error cleaning the text
    """
    try:
        text = re.sub(r"<.*?>", " ", text)
        text = re.sub(r"&[a-z]+;", " ", text)
        if remove_numbers:
            text = re.sub(r"\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return text


def extract_aspects(text: str) -> List[str]:
    """
    Extracts aspects (nouns) from the text using POS tagging.
    
    Args:
        text (str): The text to extract aspects from
    
    Returns:
        List[str]: List of extracted aspects (nouns)
        
    Raises:
        Exception: If there is an error extracting aspects
    """
    try:
        words = word_tokenize(text)
        pos_tags = pos_tag(words)
        aspects = [word for word, tag in pos_tags if tag in ['NN', 'NNS', 'NNP', 'NNPS']]
        return aspects
    except Exception as e:
        logger.error(f"Error extracting aspects: {e}")
        return []


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the data by applying multiple text processing steps.
    
    Steps include:
    1. Removing duplicates
    2. Expanding contractions
    3. Handling negations
    4. Cleaning text
    5. Extracting aspects
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess
        
    Returns:
        pd.DataFrame: The preprocessed DataFrame
        
    Raises:
        Exception: If there is an error during preprocessing
    """
    logger.info("Preprocessing data...")
    total_rows = len(df)
    copy = df.copy()
    
    # Step 1: Remove duplicates
    logger.info("Step 1/6: Removing duplicates...")
    copy = remove_duplicates(copy)
    processed_rows = len(copy)
    logger.info(f"Processing {processed_rows} unique rows")
    
    # Step 2: Expand contractions
    logger.info("Step 2/6: Expanding contractions...")
    tqdm.pandas(desc="Expanding contractions")
    copy["review"] = copy["review"].progress_apply(expand_contractions)
    
    # Step 3: Handle negations
    logger.info("Step 3/6: Handling negations...")
    tqdm.pandas(desc="Handling negations")
    copy["review"] = copy["review"].progress_apply(handle_negations)
    
    # Step 4: Clean text
    logger.info("Step 4/6: Cleaning text...")
    tqdm.pandas(desc="Cleaning text")
    copy["review"] = copy["review"].progress_apply(clean_text)
    
    # Step 5: Extract aspects
    logger.info("Step 5/6: Extracting aspects...")
    tqdm.pandas(desc="Extracting aspects")
    copy["aspects"] = copy["review"].progress_apply(extract_aspects)

    # Step 6: Remove duplicates and empty rows
    logger.info("Step 6/6: Removing duplicates and empty rows...")
    copy = remove_duplicates(copy)
    copy = copy[copy["review"].notna()]

    logger.info(f"Preprocessing complete for {copy.shape[0]} reviews")
    return copy