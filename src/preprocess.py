"""
전처리 모듈
EDA(notebooks/01_eda.ipynb) 7.3절의 전처리 결정을 코드로 옮긴 것이다.
모든 모델이 같은 전처리를 거치도록, 모델 노트북에서는 이 파일의 함수만 사용한다.
"""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

TARGET = "income"
DROP_COLS = ["fnlwgt", "education"]

PLAIN_NUMERIC_COLS = ["age", "education-num", "hours-per-week"]
CAPITAL_COLS = ["capital-gain", "capital-loss"]
CATEGORICAL_COLS = [
    "workclass", 
    "marital-status", 
    "occupation", 
    "relationship", 
    "race", 
    "sex", 
    "native-country"]

RARE_MIN_FREQUENCY = 50
#------------------------------------------------------------
# 손질 1단계: 입력 항목(X)과 정답(y) 나누기
#------------------------------------------------------------

def split_features_target(df):
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = (df[TARGET] == ">50K").astype(int)

    return X, y

#------------------------------------------------------------
# 손질 2-3단계 빈칸 채우기, one-hot encoding
#------------------------------------------------------------

def _categorical_steps(min_frequency=RARE_MIN_FREQUENCY):
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),

        ("onehot", OneHotEncoder(
            min_frequency=min_frequency, 
            handle_unknown="infrequent_if_exist"
            )),
    ])

#------------------------------------------------------------
# 손질 4단계: 숫자 항목 다음기(로그 변환, 크기 맞추기)

def _plain_numeric_steps(scale=True):
    if scale:
        return StandardScaler()
    return "passthrough"

def _capital_steps(scale=True, log_capital=True):
    steps = []
    
    if log_capital:
        steps.append(("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")))
    
    if scale:
        steps.append(("scale", StandardScaler()))
    
    if not steps:
        return "passthrough"
    return Pipeline(steps)

#------------------------------------------------------------
# 묶기: 열 묶음마다 알맞은 손질 도구를 배정한다
#------------------------------------------------------------

def build_preprocessor(scale=True, log_capital=True, min_frequency=RARE_MIN_FREQUENCY):

    return ColumnTransformer([
        ("numeric", _plain_numeric_steps(scale), PLAIN_NUMERIC_COLS),
        ("capital", _capital_steps(scale, log_capital), CAPITAL_COLS),
        ("categorical", _categorical_steps(min_frequency), CATEGORICAL_COLS),
    ])

#------------------------------------------------------------
# 최종 조립: 전처리 + 모델을 하나의 파이프라인으로 잇는다
#------------------------------------------------------------

def make_model_pipeline(model, **preprocess_options):
    return Pipeline([
        ("preprocess", build_preprocessor(**preprocess_options)),
        ("model", model),
    ])