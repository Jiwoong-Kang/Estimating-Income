# Estimating-Income

A machine learning course project (**Project Type 1**) that predicts whether a person’s income is above **$50K** using the [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult) dataset.

## Purpose

The goal is not only to train models, but to **understand the data**, build a **fair comparison pipeline**, and explain **what works, what does not, and why**—linking results to model assumptions and dataset properties (class imbalance, missing values, correlated features, etc.).


## What this project does

1. **EDA** on the training set to guide preprocessing and evaluation choices.
2. **Shared preprocessing** (`src/preprocess.py`) built once for all models. Column handling is shared, but options can differ by model (e.g. `scale=False` for trees/ensembles, `dense=True` for GaussianNB).
3. **Shared evaluation** (`src/evaluate.py`): stratified 5-fold CV, multiple metrics, Dummy baseline, test-set confusion matrix.
4. **Supervised models** in dedicated notebooks, with selective hyperparameter tuning where it matters.
5. **Ensembles**: bagging, random forest, AdaBoost, gradient boosting.
6. **Extra analyses**: k-means clustering, bias–variance curves (train vs CV).
7. **Saved results** in `results/*.json` for the final comparison / report.

## Notebooks

| Notebook | Topic |
|---|---|
| `01_eda.ipynb` | Exploratory analysis & preprocessing decisions |
| `02_dummy.ipynb` | Dummy baseline |
| `03_linear_regression.ipynb` | Linear regression + threshold |
| `04_polynomial.ipynb` | Polynomial regression + threshold |
| `05_perceptron.ipynb` | Perceptron |
| `06_svm.ipynb` | Linear SVM (`LinearSVC`) |
| `07_knn.ipynb` | k-NN |
| `08_decision_tree.ipynb` | Decision tree |
| `09_naive_bayes.ipynb` | Gaussian Naive Bayes (`dense=True`) |
| `10_logistic_regression.ipynb` | Logistic regression |
| `11_clustering.ipynb` | K-means (Elbow / Silhouette, then income rates) |
| `12_bias_variance.ipynb` | Bias–variance via train vs CV F1 |
| `13_bagging.ipynb` | Bagging vs Random Forest |
| `14_boosting.ipynb` | AdaBoost vs Gradient Boosting |

## Setup

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use this `venv` as the Jupyter / VS Code / Cursor kernel.

## Data

Files under `data/adult/` (`adult.data`, `adult.test`, `adult.names`).

- Loaded by `src/data.py`: `load_adult("train")` / `load_adult("test")`
- Roughly 32k train / 16k test rows; label imbalance ~76% `<=50K` vs ~24% `>50K`

## How to run

1. Activate the virtual environment.
2. Open a notebook under `notebooks/` and run top to bottom.
3. Notebooks use `sys.path.append("..")` to import `src/`.

Suggested order: `01_eda` → `02_dummy` → model notebooks → `13` / `14` ensembles → comparison.

## Evaluation

From `src/evaluate.py`:

| Setting | Value |
|---|---|
| `RANDOM_STATE` | `42` |
| CV | 5-fold `StratifiedKFold` (shuffle=True) |
| Metrics | accuracy, balanced_accuracy, f1, roc_auc, average_precision |
| Test | full train fit + confusion matrix on `adult.test` |

Scores use `predict_proba` when available, otherwise `decision_function`.

## Tuning vs fixed settings

- **Base models** (e.g. tree depth, k-NN `k`, SVM `C`): candidates + CV where useful; choices stored in JSON `extra`.
- **Ensembles**: often fix `n_estimators=100` (and GB `learning_rate=0.1`) to compare *methods* fairly.
- **Clustering**: choose `k` with Elbow / Silhouette (not a classification grid).
- **Bias–variance**: vary complexity on purpose; the curve matters more than a single “best” setting.

## Project layout

```
Estimating-Income/
├── data/adult/          # UCI Adult files
├── notebooks/           # EDA, models, clustering, bias–variance, ensembles
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

- Turning EDA into preprocessing and metric decisions
- Why imbalanced data needs more than accuracy
- Pipelines / ColumnTransformers and avoiding leakage
- How linear models, trees, ensembles, and Naive Bayes behave on the same task
- Illustrating bias–variance with train vs CV curves (not a full numeric decomposition)
- Writing “what works / what doesn’t” with evidence for a course report
