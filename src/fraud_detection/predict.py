"""Inference helpers backed by the saved model and scaler."""

from functools import lru_cache
from typing import Sequence

import joblib
import pandas as pd
from tensorflow import keras

from fraud_detection.config import DECISION_THRESHOLD, FEATURE_NAMES, MODEL_PATH, SCALER_PATH


@lru_cache(maxsize=1)
def _load_artifacts():
    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Model/scaler not found at {MODEL_PATH} / {SCALER_PATH}. "
            "Train the model first with `python -m fraud_detection.train`."
        )
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_batch(df: pd.DataFrame, threshold: float = DECISION_THRESHOLD) -> pd.DataFrame:
    """Score a DataFrame of transactions (columns must match FEATURE_NAMES).

    Returns a copy of `df` with two added columns: `fraud_probability` and `prediction`.
    """
    missing = set(FEATURE_NAMES) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")

    model, scaler = _load_artifacts()

    X = df[FEATURE_NAMES]
    X_scaled = scaler.transform(X)
    probabilities = model.predict(X_scaled, verbose=0).ravel()

    result = df.copy()
    result["fraud_probability"] = probabilities
    result["prediction"] = pd.Series(probabilities >= threshold).map(
        {True: "Fraud", False: "Legitimate"}
    ).values
    return result


def predict_transaction(
    transaction: Sequence[float], threshold: float = DECISION_THRESHOLD
) -> tuple[str, float]:
    """Score a single transaction given as a sequence in FEATURE_NAMES order.

    Returns (result, probability) where result is "Fraud" or "Legitimate".
    """
    transaction_df = pd.DataFrame([transaction], columns=FEATURE_NAMES)
    scored = predict_batch(transaction_df, threshold=threshold)
    row = scored.iloc[0]
    return row["prediction"], float(row["fraud_probability"])
