# Standard library imports
import logging
from typing import Tuple, Optional

# Third-party imports
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def filter_binary_classes(df: pd.DataFrame, sentiment_col: str = 'generated_sentiment') -> pd.DataFrame:
    """
    Filters the dataset to keep only positive and negative classes.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        sentiment_col (str): The name of the column with class labels.
    
    Returns:
        pd.DataFrame: Filtered DataFrame containing only positive and negative classes.
    
    Raises:
        Exception: If there is an error during filtering.
    """
    binary_df = df[df[sentiment_col].isin(['positive', 'negative'])].copy()
    logger.info(f"Filtered dataset shape: {binary_df.shape}")
    logger.info(f"Class distribution after filtering:\n{binary_df[sentiment_col].value_counts()}")
    return binary_df


def balance_binary_classes(
    df: pd.DataFrame,
    sentiment_col: str = 'generated_sentiment',
    method: str = 'stratified',
    random_state: Optional[int] = 42
    ) -> pd.DataFrame:
    """
    Balances binary classes in the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        sentiment_col (str): The name of the column with class labels.
        method (str): Balancing method ('stratified' or 'random').
        random_state (Optional[int]): Seed for reproducibility.
    
    Returns:
        pd.DataFrame: Balanced DataFrame.
    
    Raises:
        ValueError: If method is not 'stratified' or 'random'.
        Exception: If there is an error during balancing.
    """
    if method not in ['stratified', 'random']:
        raise ValueError("method must be 'stratified' or 'random'")
    
    class_counts = df[sentiment_col].value_counts()
    min_class_size = class_counts.min()
    
    balanced_dfs = []
    for sentiment in ['positive', 'negative']:
        class_df = df[df[sentiment_col] == sentiment]
        if len(class_df) > min_class_size:
            balanced_class = class_df.sample(n=min_class_size, random_state=random_state)
        else:
            balanced_class = class_df
        
        balanced_dfs.append(balanced_class)
    
    balanced_df = pd.concat(balanced_dfs, axis=0).sample(frac=1, random_state=random_state)
    
    logger.info(f"Balanced dataset shape: {balanced_df.shape}")
    logger.info(f"Class distribution after balancing:\n{balanced_df[sentiment_col].value_counts()}")
    
    return balanced_df


def process_and_balance_dataset(
    input_path: str,
    output_path: str,
    sentiment_col: str = 'generated_sentiment',
    balance_method: str = 'stratified',
    random_state: Optional[int] = 42
    ) -> Tuple[pd.DataFrame, dict]:
    """
    Full pipeline for processing and balancing the dataset.
    
    Args:
        input_path (str): Path to the input file.
        output_path (str): Path for saving the result.
        sentiment_col (str): The name of the column with class labels.
        balance_method (str): Balancing method.
        random_state (Optional[int]): Seed for reproducibility.
    
    Returns:
        Tuple[pd.DataFrame, dict]: Balanced dataset and statistics.
    
    Raises:
        FileNotFoundError: If the input file does not exist.
        ValueError: If balance_method is invalid.
        Exception: If there is an error during processing.
    """
    df = pd.read_csv(input_path)
    initial_stats = {
        'initial_shape': df.shape,
        'initial_distribution': df[sentiment_col].value_counts().to_dict()
    }
    
    binary_df = filter_binary_classes(df, sentiment_col)
    balanced_df = balance_binary_classes(binary_df, sentiment_col=sentiment_col, method=balance_method, random_state=random_state)
    
    balanced_df.to_csv(output_path, index=False)
    
    final_stats = {
        'final_shape': balanced_df.shape,
        'final_distribution': balanced_df[sentiment_col].value_counts().to_dict()
    }
    
    stats = {**initial_stats, **final_stats}
    return balanced_df, stats 