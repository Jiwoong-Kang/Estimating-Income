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
# Split features (X) and target (y)
#------------------------------------------------------------

def split_features_target(df):
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = (df[TARGET] == ">50K").astype(int)

    return X, y

#------------------------------------------------------------
# Categoricals: fill missing + one-hot encode
#------------------------------------------------------------

def _categorical_steps(min_frequency=RARE_MIN_FREQUENCY, dense=False):
    return Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),

        ("onehot", OneHotEncoder(
            min_frequency=min_frequency, 
            handle_unknown="infrequent_if_exist",
            sparse_output=not dense,
            )),
    ])

#------------------------------------------------------------
# Numerics: optional log (capital) + scaling
#------------------------------------------------------------

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
# Assign the right steps to each column group
#------------------------------------------------------------

def build_preprocessor(scale=True, log_capital=True, dense=False,min_frequency=RARE_MIN_FREQUENCY):

    return ColumnTransformer([
        ("numeric", _plain_numeric_steps(scale), PLAIN_NUMERIC_COLS),
        ("capital", _capital_steps(scale, log_capital), CAPITAL_COLS),
        ("categorical", _categorical_steps(min_frequency, dense), CATEGORICAL_COLS),
    ])

#------------------------------------------------------------
# Preprocess + model in one pipeline
#------------------------------------------------------------

def make_model_pipeline(model, **preprocess_options):
    return Pipeline([
        ("preprocess", build_preprocessor(**preprocess_options)),
        ("model", model),
    ])
