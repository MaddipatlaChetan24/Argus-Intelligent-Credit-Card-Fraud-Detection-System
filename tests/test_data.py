import numpy as np
import pandas as pd
import pytest

from fraud_detection.data import split_data


@pytest.fixture
def synthetic_df():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "Time": rng.uniform(0, 1000, n),
            **{f"V{i}": rng.normal(size=n) for i in range(1, 29)},
            "Amount": rng.uniform(0, 500, n),
        }
    )
    # ~5% fraud, matching the real dataset's heavy imbalance.
    df["Class"] = (rng.random(n) < 0.05).astype(int)
    return df


def test_split_sizes_sum_to_total(synthetic_df):
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(synthetic_df)

    total = len(X_train) + len(X_val) + len(X_test)
    assert total == len(synthetic_df)
    assert len(X_train) == len(y_train)
    assert len(X_val) == len(y_val)
    assert len(X_test) == len(y_test)


def test_split_is_stratified(synthetic_df):
    _X_train, _X_val, _X_test, y_train, y_val, y_test = split_data(synthetic_df)

    overall_rate = synthetic_df["Class"].mean()
    for y in (y_train, y_val, y_test):
        assert y.mean() == pytest.approx(overall_rate, abs=0.05)


def test_split_no_target_leakage(synthetic_df):
    X_train, X_val, X_test, _y_train, _y_val, _y_test = split_data(synthetic_df)

    for X in (X_train, X_val, X_test):
        assert "Class" not in X.columns
