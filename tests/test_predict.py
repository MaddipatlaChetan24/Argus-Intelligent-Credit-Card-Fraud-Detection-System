import numpy as np
import pandas as pd
import pytest

from fraud_detection import predict as predict_module
from fraud_detection.config import FEATURE_NAMES


class FakeScaler:
    def transform(self, X):
        return np.asarray(X)


class FakeModel:
    """Returns a fixed probability per row, taken from the 'Amount' column / 1000."""

    def predict(self, X, verbose=0):
        amounts = X[:, -1]
        return (amounts / 1000).clip(0, 1).reshape(-1, 1)


@pytest.fixture(autouse=True)
def fake_artifacts(monkeypatch):
    predict_module._load_artifacts.cache_clear()
    monkeypatch.setattr(
        predict_module, "_load_artifacts", lambda: (FakeModel(), FakeScaler())
    )


def _make_transaction(amount: float):
    return [0.0] * (len(FEATURE_NAMES) - 1) + [amount]


def test_predict_transaction_legitimate_below_threshold():
    transaction = _make_transaction(amount=100.0)  # probability 0.1
    result, probability = predict_module.predict_transaction(transaction)

    assert result == "Legitimate"
    assert probability == pytest.approx(0.1)


def test_predict_transaction_fraud_above_threshold():
    transaction = _make_transaction(amount=999.0)  # probability 0.999 >= default 0.99
    result, probability = predict_module.predict_transaction(transaction)

    assert result == "Fraud"
    assert probability == pytest.approx(0.999)


def test_predict_transaction_custom_threshold():
    transaction = _make_transaction(amount=500.0)  # probability 0.5
    result, _probability = predict_module.predict_transaction(transaction, threshold=0.4)

    assert result == "Fraud"


def test_predict_batch_adds_expected_columns():
    df = pd.DataFrame(
        [_make_transaction(100.0), _make_transaction(999.0)], columns=FEATURE_NAMES
    )

    scored = predict_module.predict_batch(df)

    assert list(scored["prediction"]) == ["Legitimate", "Fraud"]
    assert scored["fraud_probability"].tolist() == pytest.approx([0.1, 0.999])
    # Original columns/rows are preserved, not mutated in place.
    assert len(scored) == len(df)
    assert set(FEATURE_NAMES).issubset(scored.columns)


def test_predict_batch_missing_columns_raises():
    df = pd.DataFrame([{"Time": 0.0, "Amount": 100.0}])

    with pytest.raises(ValueError, match="missing required columns"):
        predict_module.predict_batch(df)
