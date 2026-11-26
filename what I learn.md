# What I Learn

Notes on what I learned while working on this project. Later used as a reference for the report's "Lessons Learnt" section.
Details are split by date under the `learn/` folder.

| Date | Topic | Key points |
|---|---|---|
| [2026-10-12](learn/2026-10-12.md) | Project setup | Project direction, dataset traits, virtual env, `src/data.py`, mistakes that fail without raising errors |
| [2026-10-13](learn/2026-10-13.md) | EDA (Exploratory Data Analysis) | Meaning of blanks, accuracy trap, distributions and caps, correlation metrics, analysis should end in a decision |
| [2026-10-14](learn/2026-10-14.md) | Preprocessing module (`src/preprocess.py`) | One-hot encoding, log transform and scaling, Pipeline and ColumnTransformer, fit/transform and data leakage |
| [2026-10-16](learn/2026-10-16.md) | Evaluation module + Dummy baseline | StratifiedKFold, multiple metrics, confusion matrix, results json, Dummy(most_frequent) floor |
| [2026-10-17](learn/2026-10-17.md) | Linear & polynomial regression | Threshold wrapper, inner Pipeline (poly→linear), deg=2 results, import/kernel |
| [2026-10-19](learn/2026-10-19.md) | Perceptron | decision_function, random_state, better than Dummy but unstable and weaker than regression |
| [2026-10-20](learn/2026-10-20.md) | SVM ~ Naive Bayes | LinearSVC, k-NN, Tree, NB(dense; accuracy below Dummy can still be fine) |
| [2026-10-21](learn/2026-10-21.md) | Logistic Regression | Probabilistic model, similar strength to SVM and linear regression, added for comparison |
| [2026-10-23](learn/2026-10-23.md) | Hyperparameter tuning (Poly~kNN) | deg=2, Perceptron `alpha`, SVM `C=1`, kNN `k=35` |
| [2026-10-24](learn/2026-10-24.md) | Tree / NB / Logistic tuning | Tree depth=15, NB smoothing=0.1, Logistic `C=10` (low sensitivity) |
| [2026-10-26](learn/2026-10-26.md) | K-Means clustering | Elbow+Silhouette, k=2, high-income rates by cluster (20.6% vs 62%) |
| [2026-10-28](learn/2026-10-28.md) | Bias–Variance | Indirect view via train vs CV, Tree/kNN/Poly, sklearn key names, F1 |
| [2026-11-02](learn/2026-11-02.md) | Bagging & Random Forest | Ensemble > single tree, Bagging vs RF, compare at depth 15 |
| [2026-11-04](learn/2026-11-04.md) | Boosting (Ada + GB) | stump vs depth 3, n/lr fixed, GB best overall |
| [2026-11-18](learn/2026-11-18.md) | MLP | ReLU·adam·dense/scale, max_iter 300>500 (Test), below ensembles |
| [2026-11-25](learn/2026-11-25.md) | All-model score table | Test + CV metrics for every `results/*.json`, metric definitions |
