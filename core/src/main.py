# Standard library imports
import logging
import os
import sys
import time
from typing import Any, Dict

# Local imports
from config.config import CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def phase1() -> None:
    """
    Phase 1: Load and preprocess full IMDB dataset.

    This function:
    1. Loads IMDB reviews from raw folder structure (train/test/unsup)
    2. Applies preprocessing: text normalization, negation handling, cleaning
    3. Extracts noun-based aspects from each review
    4. Splits and saves the data into supervised and unsupervised CSVs
    5. Logs basic properties of the supervised data
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If the IMDB dataset directory does not exist.
        KeyError: If expected columns are missing.
        Exception: If there is an error loading or preprocessing data.
    """
    import pandas as pd
    from data_processing import load_imdb_data, preprocess_data
    
    start_time = time.time()
    logger.info("Phase 1: Starting IMDB data loading and preprocessing...")

    # Step 1: Load data
    load_start = time.time()
    df = load_imdb_data(CONFIG['base_path'])
    load_end = time.time()
    load_duration = (load_end - load_start) / 60
    logger.info(f"Data loaded in {load_duration:.2f} minutes. Loaded {len(df)} reviews.")

    # Step 2: Preprocess data
    preprocess_start = time.time()
    logger.info("Starting data preprocessing. This may take some time...")
    df = preprocess_data(df)
    preprocess_end = time.time()
    preprocess_duration = (preprocess_end - preprocess_start) / 60
    logger.info(f"Preprocessing completed in {preprocess_duration:.2f} minutes.")
    
    # Split data into supervised and unsupervised
    logger.info("Splitting data into labeled and unlabeled sets...")
    supervised = df[df["sentiment"] != "unsupervised"]
    unsupervised = df[df["sentiment"] == "unsupervised"]
    
    logger.info(f"Supervised data: {len(supervised)} rows")
    logger.info(f"Unsupervised data: {len(unsupervised)} rows")
    
    # Save data
    logger.info("Saving processed data...")
    supervised.to_csv(CONFIG['supervised_path'], index=False)
    logger.info(f"Data saved to {CONFIG['supervised_path']}")
    
    unsupervised.to_csv(CONFIG['unsupervised_path'], index=False)
    logger.info(f"Data saved to {CONFIG['unsupervised_path']}")

    supervised_df = pd.read_csv(CONFIG['supervised_path'])
    logger.info(f"Loaded data from {CONFIG['supervised_path']} ({supervised_df.shape[0]} rows)")
    logger.info(f"Supervised data shape: {supervised_df.shape}")
    logger.info(f"Data sample: {supervised_df.sample(5)}")

    total_duration = (time.time() - start_time) / 60
    logger.info(f"Phase 1 complete: Raw IMDB data loaded, preprocessed, and saved. Total time: {total_duration:.2f} minutes")


def phase2() -> None:
    """
    Phase 2: Detect and remove duplicates in both supervised and unsupervised data.

    This function:
    1. Loads the cleaned supervised and unsupervised datasets
    2. Detects duplicated rows and logs the count
    3. Removes duplicates
    4. Saves the cleaned versions back to disk
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If datasets do not exist.
        Exception: If there is an error processing the data.
    """
    import pandas as pd
    
    logger.info("Phase 2: Checking and removing duplicates...")

    supervised_df = pd.read_csv(CONFIG['supervised_path'])
    logger.info(f"Loaded data from {CONFIG['supervised_path']} ({supervised_df.shape[0]} rows)")
    
    unsupervised_df = pd.read_csv(CONFIG['unsupervised_path'])
    logger.info(f"Loaded data from {CONFIG['unsupervised_path']} ({unsupervised_df.shape[0]} rows)")

    duplicates_count = supervised_df.duplicated().sum() + unsupervised_df.duplicated().sum()
    logger.info(f"{duplicates_count} duplicates found")

    supervised_df = supervised_df.drop_duplicates()
    unsupervised_df = unsupervised_df.drop_duplicates()

    logger.info(f"Supervised data shape: {supervised_df.shape}")
    logger.info(f"Unsupervised data shape: {unsupervised_df.shape}")
    logger.info(f"Total duplicates dropped: {duplicates_count}")

    supervised_df.to_csv(CONFIG['supervised_path'], index=False)
    logger.info(f"Data saved to {CONFIG['supervised_path']}")
    
    unsupervised_df.to_csv(CONFIG['unsupervised_path'], index=False)
    logger.info(f"Data saved to {CONFIG['unsupervised_path']}")

    logger.info("Phase 2 complete: Duplicates removed and cleaned data saved.")


def phase3() -> None:
    """
    Phase 3: Split supervised dataset into train/validation/test sets.

    This function:
    1. Loads the cleaned supervised dataset
    2. Splits it into training, validation and test subsets
    3. Saves each split into separate files for downstream usage
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If supervised dataset does not exist.
        ValueError: If there is an issue with splitting the data.
        Exception: If there is an error during splitting.
    """
    from data_split import split_dataset
    
    logger.info("Phase 3: Splitting supervised data into train/val/test...")
    split_dataset(
        load_path=CONFIG['supervised_path'],
        save_path=CONFIG["unprocessed_aspects_path"],
        stratify_col="sentiment"
    )
    logger.info("Phase 3 complete: Supervised data split and saved.")


def phase4() -> None:
    """
    Phase 4: Process extracted aspects for each review with comprehensive pipeline.

    This function:
    1. Loads train/val/test datasets with raw extracted aspects
    2. Filters rare or generic aspects
    3. Normalizes synonyms and merges similar terms
    4. Extracts context for each aspect and analyzes potential sentiment contrast
    5. Filters aspects using the provided whitelist
    6. Saves the processed aspect-enriched datasets
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If datasets do not exist.
        KeyError: If required columns are missing.
        Exception: If there is an error processing aspects.
    """
    from aspect_preparation import run_aspect_preparation
    
    logger.info("Phase 4: Starting aspect processing pipeline...")
    run_aspect_preparation()
    logger.info("Phase 4 complete: Aspects processed successfully.")


def phase5() -> None:
    """
    Phase 5: Generation of synthetic sentiments using SOLAR-10.7B-Instruct-v1.0 with neutral class filtering.
    
    This function:
    1. Uses pre-processed aspect data from previous phases
    2. Generates synthetic sentiments for each review-aspect pair using SOLAR-10.7B-Instruct-v1.0
    3. Classifies sentiments as positive, negative, or neutral
    4. Filters out neutral sentiments to create high-quality binary datasets
    5. Uses 4-bit quantization for memory efficiency with larger model
    6. Controls the number of samples per split for training efficiency
    
    Returns:
        None
        
    Raises:
        FileNotFoundError: If input files do not exist.
        RuntimeError: If there is an issue with the model.
        Exception: If there is an error generating sentiments.
    """
    logger.info("Phase 5: Starting the synthetic sentiment generation pipeline with SOLAR-10.7B-Instruct-v1.0...")

    # Local import to prevent parallel loading conflicts
    from synth_sentiment_generation import generate_synthetic_data

    # Low number of samples for testing
    max_samples = {
        "train": 20000, # Change to 10000 for full dataset
        "val": 2000, # Change to 2000 for full dataset
        "test": 2000 # Change to 2000 for full dataset
    }

    generate_synthetic_data(max_samples_per_split=max_samples)
    
    logger.info("Phase 5 complete: Aspect sentiments successfully generated with SOLAR-10.7B-Instruct-v1.0 model including neutral filtering.")


def phase6() -> Dict[str, Dict[str, Any]]:
    """
    Phase 6: Balance classes in filtered synthetic data.
    
    This function:
    1. Loads filtered synthetic data for each split (train/val/test) 
    2. Balances positive and negative sentiment classes using stratified sampling
    3. Saves balanced datasets for BERT-ABSC model training
    4. Returns statistics about the class distribution before and after balancing
    
    Returns:
        Dict[str, Dict[str, Any]]: Statistics about the class distribution before and after balancing
        
    Raises:
        FileNotFoundError: If input files do not exist.
        KeyError: If required columns are missing.
        Exception: If there is an error balancing classes.
    """
    import pandas as pd
    from balance_classes import process_and_balance_dataset
    
    logger.info("Phase 6: Starting class balancing for filtered sentiment data...")
    
    splits = ['train', 'val', 'test']
    stats = {}
    
    for split in splits:
        logger.info(f"Processing {split} split...")
        # Use filtered datasets (without neutral sentiments)
        input_path = os.path.join(CONFIG['synthetic_sentiments_path'], f"{split}_data_filtered.csv")
        output_path = os.path.join(CONFIG['synthetic_sentiments_path'], f"{split}_synthetic_balanced.csv")
        
        _, split_stats = process_and_balance_dataset(
            input_path=input_path,
            output_path=output_path,
            sentiment_col='sentiment',  # Use 'sentiment' which is the column name in filtered data
            balance_method='stratified',
            random_state=42
        )
        
        stats[split] = split_stats
        logger.info(f"Completed processing {split} split")
        logger.info(f"Initial distribution: {split_stats['initial_distribution']}")
        logger.info(f"Final distribution: {split_stats['final_distribution']}")
    
    logger.info("Phase 6 complete: Classes balanced and saved.")
    return stats


def phase7() -> None:
    """
    Phase 7: Prepare data and train LCF-ATEPC model for aspect-based sentiment analysis.
    
    This function:
    1. Prepares balanced data in format appropriate for LCF-ATEPC
    2. Creates datasets for training and validation
    3. Trains the model with optimized parameters for aspect-based sentiment analysis:
       - Uses cosine learning rate schedule with warmup
       - Implements specialized attention mechanisms for aspects
       - Utilizes gradient accumulation for efficient training
       - Applies label smoothing and early stopping
    4. Saves the best model based on F1 score directly in the models/lcf_atepc directory
    5. Prepares the model for deployment by saving tokenizer and inference components
    
    Returns:
        None
    
    Raises:
        FileNotFoundError: If input files do not exist.
        RuntimeError: If there is an issue with training.
        Exception: If there is an error training the model.
    """
    import pandas as pd
    from train_model import prepare_data_format, ABSADataset, train_model
    
    logger.info("Phase 7: Starting LCF-ATEPC training pipeline for ABSA...")
    
    lcf_atepc_deployment_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "lcf_atepc")
    os.makedirs(lcf_atepc_deployment_path, exist_ok=True)
    
    # Format our balanced data for LCF-ATEPC
    for split in ['train', 'val', 'test']:
        input_path = os.path.join(CONFIG["synthetic_sentiments_path"], f"{split}_synthetic_balanced.csv")
        output_path = os.path.join(CONFIG["lcf_atepc_data_path"], f"{split}_lcf_atepc.csv")
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input file not found: {input_path}")
            
        prepare_data_format(input_path, output_path)
    
    # Create datasets for training
    train_df = pd.read_csv(os.path.join(CONFIG["lcf_atepc_data_path"], "train_lcf_atepc.csv"))
    val_df = pd.read_csv(os.path.join(CONFIG["lcf_atepc_data_path"], "val_lcf_atepc.csv"))
    
    required_cols = ["review", "aspect", "sentiment"]
    for df, name in [(train_df, "train"), (val_df, "val")]:
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in {name} dataset: {missing_cols}")
    
    # Enable data augmentation for training (30% of samples will be augmented)
    train_dataset = ABSADataset(train_df, augment=True)
    val_dataset = ABSADataset(val_df)
    
    # Train model
    try:
        model_path = train_model(train_dataset, val_dataset)
        logger.info(f"Model trained successfully and saved to {model_path}")
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        raise
    
    logger.info(f"Phase 7 complete: LCF-ATEPC model trained, saved and prepared for deployment at {lcf_atepc_deployment_path}")
    return model_path


if __name__ == '__main__':
    """
    Main execution flow of the IMDB aspect-based sentiment analysis pipeline.
    
    This script orchestrates the entire pipeline from data loading to model training.
    Each phase can be run independently or as part of the complete flow.
    
    Usage:
        python -m core.src.main            # Run all phases sequentially
        python -m core.src.main --phase N  # Run specific phase N (1-9)
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="IMDB Aspect-Based Sentiment Analysis Pipeline")
    parser.add_argument('--phase', type=int, help='Specific phase to run (1-9)')
    args = parser.parse_args()
    
    # Mapping phases to corresponding functions
    phases = {
        1: phase1,    # Load and preprocess IMDB dataset
        2: phase2,    # Remove duplicates
        3: phase3,    # Split dataset
        4: phase4,    # Process aspects
        5: phase5,    # Generate synthetic sentiments
        6: phase6,    # Balance classes
        7: phase7     # Train LCF-ATEPC model
    }
    
    if args.phase:
        # Running specific phase
        if args.phase in phases:
            logger.info(f"Running phase {args.phase}...")
            phases[args.phase]()
            logger.info(f"Phase {args.phase} completed successfully.")
        else:
            logger.error(f"Phase {args.phase} not found. Available phases: {', '.join(map(str, sorted(phases.keys())))}")
    else:
        # Running all phases sequentially
        logger.info("Starting full IMDB ABSA pipeline...")
        
        # Executing phases in the correct order
        for phase_num in sorted(phases.keys()):
            try:
                logger.info(f"\n{'='*50}\nStarting Phase {phase_num}\n{'='*50}")
                phases[phase_num]()
                logger.info(f"Phase {phase_num} completed successfully.")
            except Exception as e:
                logger.error(f"Error in Phase {phase_num}: {str(e)}")
                logger.error("Pipeline stopped due to error.")
                break
                
        logger.info("\nIMDB ABSA pipeline completed!")