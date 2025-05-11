# Standard library imports
import os
import sys
import logging
from typing import Optional

# Third-party imports
import pandas as pd
from sklearn.model_selection import train_test_split

# Local imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def split_dataset(load_path: str, save_path: str, test_size: float = 0.2, val_size: float = 0.25, stratify_col: Optional[str] = None) -> None:
    """
    Split any dataset into train, validation, and test sets.

    Args:
        load_path (str): Path to the input .csv file.
        save_path (str): Directory to save the split files.
        test_size (float): Proportion of total data for test set (e.g., 0.2 → 20%).
        val_size (float): Proportion of train_val to use for validation (e.g., 0.25 → 25% of train_val).
        stratify_col (Optional[str]): Optional column to stratify on. Set to None to disable stratification.

    Returns:
        None
        
    Raises:
        ValueError: If failed to load dataset or stratify_col does not exist.
        FileNotFoundError: If load_path does not exist.
        Exception: If there is an error during splitting.
    """
    try:
        logger.info(f"Splitting dataset from {load_path}...")

        df = pd.read_csv(load_path)
        logger.info(f"Loaded data from {load_path} ({df.shape[0]} rows)")
        
        if df is None or df.empty:
            raise ValueError("Failed to load dataset.")

        stratify_values = df[stratify_col] if stratify_col and stratify_col in df.columns else None

        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42, stratify=stratify_values
        )

        stratify_values_tv = train_val_df[stratify_col] if stratify_col and stratify_col in df.columns else None

        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size, random_state=42, stratify=stratify_values_tv
        )

        logger.info(f"Split complete. Train={len(train_df)} | Val={len(val_df)} | Test={len(test_df)}")

        # Save data
        train_path = os.path.join(save_path, "train_data.csv")
        train_df.to_csv(train_path, index=False)
        logger.info(f"Data saved to {train_path}")

        val_path = os.path.join(save_path, "val_data.csv")
        val_df.to_csv(val_path, index=False)
        logger.info(f"Data saved to {val_path}")
        
        test_path = os.path.join(save_path, "test_data.csv")
        test_df.to_csv(test_path, index=False)
        logger.info(f"Data saved to {test_path}")

        logger.info("Saved all splits to disk.")
    except Exception as e:
        logger.error(f"Error splitting dataset: {e}")
        raise e