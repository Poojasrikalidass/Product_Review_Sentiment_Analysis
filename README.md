# Product_Review_Sentiment_Analysis

# Embeddings to Transformers: Product Review Sentiment Analysis

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![scikit--learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![sentence--transformers](https://img.shields.io/badge/sentence--transformers-MiniLM--L6--v2-purple)
![License](https://img.shields.io/badge/license-MIT-green)

A case-study notebook that builds a sentiment classification system for
e-commerce product reviews using pre-trained sentence embeddings and classical
ML classifiers.

## Table of Contents

- [Business Context](#business-context)
- [Objective](#objective)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [Usage](#usage)
- [Repository Structure](#repository-structure)
- [License](#license)

## Business Context

Customer reviews are a direct channel between consumers and businesses, carrying
insight into satisfaction, product performance, and perceived value. Given the
volume and unstructured nature of review text, manual analysis doesn't scale —
this project automates that classification.

## Objective

Classify product reviews into three categories — **positive**, **neutral**, and
**negative** — to support targeted business actions: reinforcing strengths found
in positive reviews, surfacing unmet expectations from neutral feedback, and
flagging issues raised in negative reviews.

## Dataset

| Column | Description |
|---|---|
| `Product ID` | Unique identifier for each product |
| `Product Review` | Free-text customer review |
| `Sentiment` | Label: positive / neutral / negative |

- **Source rows:** 1,007
- **After cleaning:** 1,005 (2 duplicate rows dropped)
- **Unique products:** 66
- **Missing values:** none
- **Class balance:** heavily skewed toward positive (~855 positive vs. ~75 each
  for neutral and negative)

## Pipeline

1. **Clean data** — drop duplicates, reset index.
2. **Generate sentence embeddings** — encode each review with the pre-trained
   `all-MiniLM-L6-v2` SentenceTransformer model, producing a 1,005 × 384
   embedding matrix.
3. **Feature/label split** — `X` = embedding matrix, `y` = `Sentiment`.
4. **Train/test split** — 80:20, stratified by sentiment, `random_state=42`
   → 804 training rows, 201 test rows.
5. **Model building** — train two classifiers on the embeddings.
6. **Evaluation** — accuracy and weighted F1 on train and test sets.

<p align="center"><img src="https://i.ibb.co/wN0NQW2D/preprocessing-table-impressive.jpg" width="720" alt="Data preprocessing pipeline"></p>

<p align="center"><img src="https://i.ibb.co/G44m8yKc/model-building-table.jpg" width="720" alt="Model building summary"></p>

## Results

| Model | Train Acc | Train F1 | Test Acc | Test F1 | Generalization gap |
|---|---|---|---|---|---|
| Random Forest | 100% | 100% | 86.6% | 81.8% | -13.4% acc / -18.2% F1 |
| Gradient Boosting | ~99.9% | ~99.9% | 84.1% | 80.3% | -15.8% acc / -19.6% F1 |

<p align="center"><img src="https://i.ibb.co/v4tG8C0X/model-comparison.jpg" width="600" alt="Model comparison chart"></p>

**Final model: Random Forest + Transformer embeddings**, chosen for its
smaller train-to-test generalization gap and higher test accuracy/F1.

<p align="center"><img src="https://i.ibb.co/F4Zz4ptc/final-selection-table.jpg" width="720" alt="Final model selection"></p>

## Conclusion

Customer reviews were categorized into positive, neutral, and negative
sentiment using sentence embeddings from a pre-trained Transformer model.
Random Forest and Gradient Boosting classifiers were trained and evaluated on
these embeddings; Random Forest generalized better and was selected as the
final model. Suggested next steps include exploring XGBoost, SVMs, sequential
neural networks, or fine-tuning a Transformer directly on the review text for
further performance gains.

## Requirements

```
pandas
numpy
sentence-transformers
scikit-learn
matplotlib
seaborn
```

Install with:

```bash
pip install pandas numpy sentence-transformers scikit-learn matplotlib seaborn
```

## Usage

1. Clone the repo and install dependencies (see above).
2. Place `Product_Reviews.csv` in the expected data path (the notebook loads it
   from Google Drive by default — update the path for local use).
3. Run the notebook cells sequentially from top to bottom.

```bash
jupyter notebook Case_Study_Product_Review_Sentiment_Analysis_Transformers.ipynb
```

## Repository Structure

```
.
├── Case_Study_Product_Review_Sentiment_Analysis_Transformers.ipynb
├── README.md
└── images/
    ├── data_preprocessing.jpg
    ├── model_building.jpg
    ├── model_comparison.jpg
    └── final_model_selection.jpg
```
