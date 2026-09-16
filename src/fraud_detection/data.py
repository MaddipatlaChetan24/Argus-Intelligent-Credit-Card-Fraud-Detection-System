"""Data loading and splitting utilities."""

import pandas as pd
from sklearn.model_selection import train_test_split

from fraud_detection.config import (
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_NAME,
    TEST_SIZE,
    VAL_SIZE,
)


def load_dataset(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw Kaggle credit card transactions CSV and drop duplicate rows."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download 'creditcard.csv' from "
            "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud and place it there."
        )

    df = pd.read_csv(path)
    return df.drop_duplicates()


def split_data(df: pd.DataFrame):
    """Stratified split into train/validation/test sets (60/20/20 of the full data).

    Returns (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    X = df.drop(columns=[TARGET_NAME])
    y = df[TARGET_NAME]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=VAL_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
