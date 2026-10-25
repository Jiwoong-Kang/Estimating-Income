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
   - Logistic regression
5. **Saved results** (`results/*.json`) for later comparison in a report.

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Open notebooks with Jupyter or VS Code/Cursor using this `venv` kernel.

## Data

Training and test files live under `data/adult/` (`adult.data`, `adult.test`, `adult.names`).

- Source: [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult)
- Loaded by `src/data.py` (`load_adult("train")` / `load_adult("test")`)

No extra download step is needed if those files are already in the repo.

## How to run

1. Activate the virtual environment (see Setup).
2. Open any notebook under `notebooks/` and run it.
3. Pick the model you want (for example `06_svm.ipynb` or `10_logistic_regression.ipynb`) and run its cells top to bottom.

Shared code is imported from `src/`, so run notebooks from the `notebooks/` folder (or keep `sys.path.append("..")` as in the existing cells).

Optional starting points:

- `01_eda.ipynb` — exploratory analysis and preprocessing decisions
- `02_dummy.ipynb` — baseline before other models

## Reproducing results

- Each model notebook’s final cell calls `save_results(...)` and writes a JSON file under `results/`.
- Re-run that notebook to regenerate the corresponding `results/*.json`.
- Hyperparameter choices (when tuned) are stored in the JSON `extra` field (candidates, selection metric, selected value).

## Config / seeds

Defined mainly in `src/evaluate.py`:

| Setting | Value |
|---|---|
| `RANDOM_STATE` | `42` |
| CV | 5-fold `StratifiedKFold` (shuffle=True) |
| Metrics | accuracy, balanced_accuracy, f1, roc_auc, average_precision (+ confusion matrix on test) |

Model notebooks also pass `random_state=42` where the estimator supports it. Preprocessing is shared via `src/preprocess.py` (`make_model_pipeline`).

## Project layout

```
Estimating-Income/
├── data/adult/          # UCI Adult files
├── notebooks/           # EDA + one notebook per model
├── results/             # saved metrics JSON
├── src/
│   ├── data.py          # load train/test
│   ├── preprocess.py    # shared preprocessing pipeline
│   └── evaluate.py      # CV, test metrics, save/load results
├── learn/               # dated learning notes
├── what I learn.md      # index of learn/
├── requirements.txt
└── README.md
```

## What you can learn from this project

- How to turn EDA findings into concrete preprocessing and metric decisions
- Why imbalanced classification needs more than accuracy (F1, ROC-AUC, balanced accuracy, confusion matrices)
- How Pipelines and ColumnTransformers prevent data leakage
- How the same problem behaves under different model assumptions (linear boundaries, soft-margin SVM, lazy learners, trees, etc.)
- How to compare models fairly with one shared train/test split, one preprocessor, and one evaluation script
- How to write up “what works / what doesn’t” with evidence, not just leaderboard scores
