# Setting Up and Running the Project from Scratch

This document provides step-by-step instructions for setting up and running the IMDB ABSA project from scratch.

## Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/imdb-absa.git
   cd imdb-absa
   ```

2. **Create a virtual environment**
   ```bash
   # For Windows
   python -m venv .venv
   .venv\Scripts\activate
   
   # For Linux/Mac
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Full Pipeline

The project's workflow is divided into several phases that must be executed in sequence:

1. **Data loading and preprocessing**
   ```bash
   python -m core.src.main --phase 1
   ```
   This loads the IMDB dataset from raw files and applies preprocessing.

2. **Removing duplicates**
   ```bash
   python -m core.src.main --phase 2
   ```
   Detects and removes duplicate entries in the datasets.

3. **Data splitting**
   ```bash
   python -m core.src.main --phase 3
   ```
   Splits the supervised data into training, validation, and test sets.

4. **Aspect processing**
   ```bash
   python -m core.src.main --phase 4
   ```
   Extracts and processes aspects from each review.

5. **Synthetic data generation**
   ```bash
   python -m core.src.main --phase 7
   ```
   Generates synthetic sentiment data for training.

6. **Class balancing**
   ```bash
   python -m core.src.main --phase 8
   ```
   Balances the classes for more effective model training.

7. **Model training (LCF-ATEPC)**
   ```bash
   python -m core.src.main --phase 9
   ```
   Trains the LCF-ATEPC model for aspect-based sentiment analysis.

## Running Model Comparisons

To compare models on challenging examples where aspect sentiment contradicts overall review sentiment:

```bash
python -m core.exploratory.model_comparison
```

This script will:
1. Load the trained models
2. Run inference on standard test cases
3. Evaluate models on challenging contradictory sentiment examples
4. Generate visualizations comparing model performance

## Troubleshooting

- **Disk Space**: The full pipeline requires approximately 5GB of disk space, primarily for the models and processed data.
- **GPU Memory**: Training with GPU acceleration requires at least 4GB of VRAM. If you encounter GPU memory errors, reduce batch size in config.py.
- **CPU Training**: If GPU is not available, models will automatically train on CPU but will be significantly slower.
- **Path Issues**: If you encounter import errors, ensure the project root is in your Python path:
  ```python
  import sys
  import os
  sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
  ``` 