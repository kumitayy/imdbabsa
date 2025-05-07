# IMDB Aspect-Based Sentiment Analysis (ABSA)

A project for aspect-based sentiment analysis of movie reviews from the IMDB database using various deep learning approaches.

## Project Description

This project develops and compares several models for the Aspect-Based Sentiment Analysis (ABSA) task on movie reviews. Unlike traditional sentiment analysis, ABSA allows determining sentiment polarity in relation to specific aspects mentioned in the text (e.g., "plot", "acting", "special effects", etc.)

### Main Features:

1. Loading and preprocessing IMDB data
2. Extracting aspects from review texts
3. Generating synthetic data for model training
4. Training various models for the ABSA task:
   - LCF-ATEPC (Local Context Focus model)
   - RoBERTa-base
   - RoBERTa with pseudo-labeling
5. Evaluating and comparing models on test data
6. Providing an interface for analyzing arbitrary texts

## Project Structure

```
core/
├── config/             # Project configuration files
├── data/               # Datasets (raw and processed)
│   ├── raw/            # Original IMDB data
│   └── processed/      # Processed data
├── exploratory/        # Research notebooks and scripts
├── models/             # Trained models
│   ├── lcf_atepc/      # LCF-ATEPC model
│   ├── roberta_absa/   # Base RoBERTa model
│   └── roberta_absa_final/ # Final RoBERTa model
└── src/                # Source code
    ├── aspect_preparation.py  # Aspect preparation
    ├── balance_classes.py     # Class balancing
    ├── bot_execution.py       # Telegram bot integration
    ├── data_processing.py     # Data processing
    ├── data_split.py          # Data splitting
    ├── main.py               # Main project script
    ├── synth_sentiment_generation.py # Synthetic data generation
    └── train_model.py        # Model training
```

## Installation and Running

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.15+
- Pandas, NumPy, Matplotlib, Scikit-learn
- CUDA (optional, for accelerated training on GPU)

### Installing Dependencies

```bash
pip install -r requirements.txt
```

### Running the Complete Processing and Training Pipeline

```bash
python -m core.src.main
```

### Running Individual Phases

```bash
# Loading and preprocessing IMDB data
python -m core.src.main --phase 1

# Removing duplicates
python -m core.src.main --phase 2

# Splitting data into training/test sets
python -m core.src.main --phase 3

# Processing aspects
python -m core.src.main --phase 4

# Generating synthetic data
python -m core.src.main --phase 7

# Balancing classes
python -m core.src.main --phase 8

# Training the LCF-ATEPC model
python -m core.src.main --phase 9
```

### Model Comparison

To compare all trained models, run:

```bash
python -m core.exploratory.model_comparison
```

## Models

### LCF-ATEPC

A BERT-based model with a local context focus mechanism that considers the importance of words in the vicinity of the aspect. This model is optimized for the ABSA task with high sentiment determination accuracy.

### RoBERTa-base

A basic model based on the RoBERTa architecture, adapted for the ABSA task by combining text and aspect through a special [SEP] token.

### RoBERTa with Pseudo-labeling

An improved version of RoBERTa using pseudo-labeling technique to enrich training data.

## Results

Model comparison results are available in the `core/data/processed/` directory after running the model comparison script.

## Future Development

- Integration with a web interface for online analysis
- Expanding the set of supported languages
- Improving aspect extraction using neural network approaches
- Optimizing models to increase inference speed

## License

MIT 