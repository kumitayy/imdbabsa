# IMDB Aspect-Based Sentiment Analysis: LCF-ATEPC vs RoBERTa Comparison

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> **Portfolio Project**: Demonstrating the superiority of LCF-ATEPC over RoBERTa in Aspect-Based Sentiment Analysis using synthetic data generation for model enhancement.

## 🎯 Project Objective

This project demonstrates a **comparative analysis** between two deep learning approaches for Aspect-Based Sentiment Analysis (ABSA) on IMDB movie reviews:

- **LCF-ATEPC** (Local Context Focus - Aspect Term Extraction and Polarity Classification)
- **RoBERTa-based** models

The key innovation showcased is the use of **synthetic data generation** to improve model performance on datasets that were originally unsuitable for ABSA tasks.

## 🚀 Key Findings & Demonstrations

### 1. **Model Superiority**: LCF-ATEPC vs RoBERTa
- LCF-ATEPC consistently outperforms RoBERTa variants in ABSA tasks
- Better handling of local context around aspect terms
- More accurate sentiment classification for specific aspects

### 2. **Synthetic Data Enhancement**
- Original IMDB dataset lacks aspect-level annotations
- Synthetic sentiment generation bridges this gap
- Demonstrates how unsuitable data can be transformed for specialized tasks

### 3. **Comprehensive Evaluation**
- Multiple metrics comparison (F1, Precision, Recall, Accuracy)
- Analysis of challenging cases where aspect sentiment contradicts overall review sentiment
- Performance evaluation on both synthetic and real test data

## 📊 Results Overview

All detailed results, visualizations, and model comparisons are available in:
**`core/exploratory/model_comparison.ipynb`**

This notebook contains:
- Performance metrics comparison
- Visualization of model strengths/weaknesses  
- Analysis of synthetic vs. real data impact
- Case studies of complex sentiment scenarios

## 🛠 Technical Implementation

### Architecture
```
core/
├── config/                    # Configuration management
│   ├── __init__.py
│   └── config.py             # Main project configuration
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Original IMDB data
│   ├── processed/            # Transformed data for ABSA
│   └── whitelist.txt         # Aspect filtering whitelist
├── exploratory/              # 📈 Main results and analysis
│   ├── model_comparison.ipynb # 🔍 CORE FINDINGS HERE
│   └── EDA.ipynb             # Exploratory Data Analysis
├── models/                   # Trained model artifacts
│   ├── lcf_atepc/           # LCF-ATEPC model
│   ├── roberta_absa/        # Base RoBERTa model
│   ├── roberta_absa_pseudo/ # RoBERTa with pseudo-labeling
│   └── model_comparison_results.csv # Model performance metrics
└── src/                      # Implementation pipeline
    ├── __init__.py
    ├── main.py              # Pipeline orchestration
    ├── data_processing.py   # Data preprocessing
    ├── data_split.py        # Data splitting
    ├── aspect_preparation.py # Aspect extraction
    ├── synth_sentiment_generation.py # 🔑 Synthetic data creation
    ├── balance_classes.py   # Class balancing
    ├── train_model.py       # Model training
    └── bot_execution.py     # Telegram bot integration
```

### Models Implemented

1. **LCF-ATEPC**: State-of-the-art ABSA model with local context focusing
2. **RoBERTa Base**: Standard transformer approach adapted for ABSA
3. **RoBERTa + Pseudo-labeling**: Enhanced with semi-supervised techniques

## 🔧 Quick Start

### Installation
```bash
git clone <repository-url>
cd imdb-absa
pip install -r requirements.txt
```

### Run Complete Pipeline
```bash
python -m core.src.main --all
```

### View Results
Open `core/exploratory/model_comparison.ipynb` to see:
- Performance comparisons
- Model analysis
- Key insights and conclusions

## 💡 Key Insights Demonstrated

1. **Specialized architectures matter**: LCF-ATEPC's local context mechanism provides significant advantages over general-purpose transformers for ABSA
2. **Data transformation viability**: Synthetic data generation can successfully adapt unsuitable datasets for specialized tasks
3. **Performance metrics**: Quantifiable improvements across multiple evaluation criteria
4. **Real-world applicability**: Analysis of edge cases and challenging scenarios

## 🔬 Project Conclusions

This research demonstrates three critical insights that extend beyond ABSA to general deep learning methodology:

### 1. **Small LLMs with Quantization are Powerful Data Transformers**
Even relatively small Language Models (LLMs) running with 4-bit quantization demonstrate **excellent results** in transforming data from completely unsuitable formats for ABSA tasks into highly suitable ones. This finding suggests that sophisticated data preprocessing using quantized models can democratize access to advanced NLP techniques without requiring massive computational resources.

### 2. **Local Context Mechanisms Reduce Overconfidence**
The **LCF-ATEPC model's local context focus mechanism** proves superior on challenging examples and shows significantly **less tendency toward overconfidence** compared to general transformers. This architectural advantage becomes particularly evident in edge cases where aspect sentiment contradicts overall review sentiment - precisely the scenarios where robust, reliable predictions matter most.

### 3. **Two Pillars of Deep Learning Success**
The project identifies two fundamental factors that had the **greatest impact on final model performance**:
- **Appropriate data generation**: Transforming unsuitable data into task-appropriate format
- **Suitable tool selection**: Choosing architectures with mechanisms aligned to the task (LCF for local context)

This reflects a **generalized approach to any deep learning task**: success depends equally on data quality/suitability and architectural alignment with the problem domain, rather than simply scaling model size or training time.

## 🎓 Educational Value

This project demonstrates:
- Advanced NLP model comparison methodologies
- Synthetic data generation techniques
- Comprehensive evaluation frameworks
- Data transformation strategies for domain adaptation

## 📄 License

MIT License - feel free to use this project as a reference for your own ABSA implementations. 