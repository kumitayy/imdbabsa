# Raw Data Directory

This directory contains the original IMDB dataset that serves as input for the ABSA pipeline.

## Data Structure

The raw IMDB dataset should follow this structure:

```
raw/
├── train/             # Training data directory
│   ├── pos/           # Positive reviews (text files)
│   └── neg/           # Negative reviews (text files)
├── test/              # Test data directory
│   ├── pos/           # Positive reviews (text files)
│   └── neg/           # Negative reviews (text files)
├── imdbEr.txt         # Extended reviews file
└── imdb.vocab         # Vocabulary file
```

## Data Download Instructions

If the data files are not present, you can download the original IMDB dataset from:
http://ai.stanford.edu/~amaas/data/sentiment/

After downloading, extract the contents to this directory maintaining the folder structure described above.

## Data Description

- **train/pos** and **train/neg**: Contains positive and negative reviews for training (12,500 each)
- **test/pos** and **test/neg**: Contains positive and negative reviews for testing (12,500 each)
- **imdbEr.txt**: Extended reviews file containing additional metadata
- **imdb.vocab**: Vocabulary file containing all unique words in the dataset

## Data Format

Each review is stored as a separate text file with the review content. The polarity (positive/negative) is indicated by the directory it's stored in rather than explicit labels in the files themselves. 