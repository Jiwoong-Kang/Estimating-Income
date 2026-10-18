"""
평가 모듈
모든 모델 노트북이 같은 방식으로 점수를 매기도록 한다.
EDA 7.4절 결정: 여러 지표 + Dummy 기준선 + StratifiedKFold
"""

from pathlib import Path
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import json
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score
)
from sklearn.model_selection import StratifiedKFold, cross_validate

CV_SCORING = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "roc_auc",
    "average_precision",
]

N_SPLITS = 5
RANDOM_STATE = 42
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

#------------------------------------------------------------
# 교차 검증으로 점수 재기
#------------------------------------------------------------

def cross_val_scores(pipe, X, y, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    
    raw = cross_validate(
        pipe, X, y,
        cv = cv,
        scoring = CV_SCORING,
        n_jobs=-1,
        return_train_score=False,
    )

    summary = {}
    for name in CV_SCORING:
        scores = raw[f"test_{name}"]
        summary[name] = {
            "mean": float(scores.mean()),
            "std": float(scores.std()),
        }
    return summary

#------------------------------------------------------------
# 시험 데이터 최종 채점 + 혼동 행렬
#------------------------------------------------------------

def _positive_scores(pipe, X):
    if hasattr(pipe, "predict_proba"):
        return pipe.predict_proba(X)[:, 1]   
    if hasattr(pipe, "decision_function"):
        return pipe.decision_function(X)    
    raise AttributeError("None of predict_proba or decision_function is available.")

def evaluate_on_test(pipe, X_train, y_train, X_test, y_test):
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_score = _positive_scores(pipe, X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "average_precision": float(average_precision_score(y_test, y_score)),
    }
   
    cm = confusion_matrix(y_test, y_pred)
    return {
        "metrics": metrics,
        "confusion_matrix": cm.tolist(),  
    }

#------------------------------------------------------------
# 회귀 -> 분류용 래퍼 (임계값)
#------------------------------------------------------------

class ThresholdingRegressor(BaseEstimator, ClassifierMixin):
    def __init__(self, regressor, threshold=0.5):
        self.regressor = regressor
        self.threshold = threshold

    def fit(self, X, y):
        self.regressor.fit(X, y)
        self.classes_ = np.array([0, 1])
        return self
    
    def decision_function(self, X):
        return np.asarray(self.regressor.predict(X)).ravel()
    
    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= self.threshold).astype(int)

#------------------------------------------------------------
# 결과 저장 / 불러오기
#------------------------------------------------------------

def save_results(model_name, cv_summary, test_result, extra=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)  

    payload = {
        "model": model_name,
        "cv": cv_summary,                 
        "test": test_result["metrics"],
        "confusion_matrix": test_result["confusion_matrix"],
    }
    if extra is not None:
        payload["extra"] = extra

    path = RESULTS_DIR / f"{model_name}.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    return path


def load_results(model_name):
    path = RESULTS_DIR / f"{model_name}.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)