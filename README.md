# Estimating-Income

A machine learning course project that predicts whether a person’s income is above **$50K** using the [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) dataset.

## Purpose

The goal is not only to train models, but to **understand the data**, build a **fair comparison pipeline**, and explain **what works, what does not, and why**—in terms of both model assumptions and dataset properties (class imbalance, missing values, correlated features, etc.).

## What this project does

1. **Exploratory data analysis (EDA)** on the training set to guide preprocessing and evaluation choices.
2. **Shared preprocessing** so every model sees the same cleaned features (imputation, one-hot encoding, scaling, optional log transform).
3. **Shared evaluation** with stratified cross-validation and multiple metrics (accuracy alone is misleading on imbalanced data), plus a Dummy baseline.
4. **Model notebooks** for algorithms covered in class, for example:
   - Linear / polynomial regression (with a threshold for classification)
   - Perceptron
   - Support vector machines (linear)
   - k-nearest neighbors
   - decision trees
   - Naive Bayes 
5. **Saved results** (`results/*.json`) for later comparison in a report.

Common code lives under `src/` (`data.py`, `preprocess.py`, `evaluate.py`). Experiments live under `notebooks/`. Learning notes are in `learn/` (indexed by `what I learn.md`).

## What you can learn from this project

- How to turn EDA findings into concrete preprocessing and metric decisions
- Why imbalanced classification needs more than accuracy (F1, ROC-AUC, balanced accuracy, confusion matrices)
- How Pipelines and ColumnTransformers prevent data leakage
- How the same problem behaves under different model assumptions (linear boundaries, soft-margin SVM, lazy learners, trees, etc.)
- How to compare models fairly with one shared train/test split, one preprocessor, and one evaluation script
- How to write up “what works / what doesn’t” with evidence, not just leaderboard scores
