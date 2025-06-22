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
    Safely removes duplicates from the DataFrame. For DataFrames containing list columns,
    duplicates are identified based on non-list columns only to avoid hashing issues.
    
    Args:
        df (pd.DataFrame): The DataFrame to remove duplicates from
    
    Returns:
        Optional[pd.DataFrame]: The DataFrame with duplicates removed or None if an error occurs
        
    Raises:
        Exception: If there is an error removing duplicates
    """
    try:
        initial_rows = df.shape[0]
        
        # Identify columns that contain lists
        list_columns = []
        non_list_columns = []
        
        for col in df.columns:
            # Check if any cell in the column contains a list
            has_lists = df[col].apply(lambda x: isinstance(x, list)).any()
            if has_lists:
                list_columns.append(col)
            else:
                non_list_columns.append(col)
        
        # Remove duplicates based on strategy
        if list_columns and non_list_columns:
            # If we have both list and non-list columns, use only non-list columns for duplicate detection
            logger.info(f"Detected list columns: {list_columns}. Using subset {non_list_columns} for duplicate removal.")
            result_df = df.drop_duplicates(subset=non_list_columns, keep='first')
        elif list_columns and not non_list_columns:
            # If all columns contain lists, convert to string representation temporarily
            logger.info("All columns contain lists. Converting to string representation for duplicate removal.")
            df_temp = df.copy().reset_index(drop=True)
            for col in list_columns:
                df_temp[col] = df_temp[col].apply(
                    lambda x: '|'.join(sorted(x)) if isinstance(x, list) and x else ''
                )
            
            # Get unique indices
            unique_indices = df_temp.drop_duplicates().index
            result_df = df.iloc[unique_indices].copy()
        else:
            # No list columns, standard duplicate removal
            result_df = df.drop_duplicates(keep='first')
        
        removed = initial_rows - result_df.shape[0]
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return result_df.reset_index(drop=True)
    
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


def preprocess_data(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Preprocesses the data by applying multiple text processing steps.
    
    Steps include:
    1. Removing duplicates
    2. Expanding contractions
    3. Handling negations
    4. Cleaning text
    5. Extracting aspects
    6. Final cleanup (removing duplicates and empty rows)
    
    Args:
        df (pd.DataFrame): The DataFrame to preprocess
        
    Returns:
        Optional[pd.DataFrame]: The preprocessed DataFrame or None if an error occurs
        
    Raises:
        Exception: If there is an error during preprocessing
    """
    logger.info("Preprocessing data...")
    
    if df is None or df.empty:
        logger.error("Input DataFrame is None or empty")
        return None
        
    total_rows = len(df)
    copy = df.copy()
    
    # Step 1: Remove duplicates
    logger.info("Step 1/6: Removing duplicates...")
    copy = remove_duplicates(copy)
    if copy is None:
        logger.error("Failed to remove duplicates in step 1")
        return None
    processed_rows = len(copy)
    logger.info(f"Processing {processed_rows} unique rows")
    
    # Step 2: Expand contractions
    logger.info("Step 2/6: Expanding contractions...")
    try:
        tqdm.pandas(desc="Expanding contractions")
        copy["review"] = copy["review"].progress_apply(expand_contractions)
    except Exception as e:
        logger.error(f"Error expanding contractions: {e}")
        return None
    
    # Step 3: Handle negations
    logger.info("Step 3/6: Handling negations...")
    try:
        tqdm.pandas(desc="Handling negations")
        copy["review"] = copy["review"].progress_apply(handle_negations)
    except Exception as e:
        logger.error(f"Error handling negations: {e}")
        return None
    
    # Step 4: Clean text
    logger.info("Step 4/6: Cleaning text...")
    try:
        tqdm.pandas(desc="Cleaning text")
        copy["review"] = copy["review"].progress_apply(clean_text)
    except Exception as e:
        logger.error(f"Error cleaning text: {e}")
        return None
    
    # Step 5: Extract aspects
    logger.info("Step 5/6: Extracting aspects...")
    try:
        tqdm.pandas(desc="Extracting aspects")
        copy["aspects"] = copy["review"].progress_apply(extract_aspects)
    except Exception as e:
        logger.error(f"Error extracting aspects: {e}")
        return None

    # Step 6: Final cleanup - remove duplicates (now with aspects column) and empty rows
    logger.info("Step 6/6: Removing duplicates and empty rows...")
    try:
        copy = remove_duplicates(copy)
        if copy is None:
            logger.error("Failed to remove duplicates in step 6")
            return None
            
        # Remove rows with empty reviews
        initial_count = len(copy)
        copy = copy[copy["review"].notna() & (copy["review"].str.strip() != "")]
        empty_removed = initial_count - len(copy)
        if empty_removed > 0:
            logger.info(f"Removed {empty_removed} rows with empty reviews")
            
    except Exception as e:
        logger.error(f"Error in final cleanup: {e}")
        return None

    logger.info(f"Preprocessing complete for {copy.shape[0]} reviews")
    return copy