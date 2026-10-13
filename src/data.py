from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "adult"

COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income"
]

def load_adult(split="train"):
    is_test = split == "test"
    path = DATA_DIR / ("adult.test" if is_test else "adult.data")
    df = pd.read_csv(
        path, 
        header=None, 
        names=COLUMNS, 
        skipinitialspace=True, 
        na_values="?", 
        skiprows=1 if is_test else 0,
    )

    df["income"] = df["income"].str.strip('.')
    return df