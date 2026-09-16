"""Evaluate the trained model on the held-out test set.

Searches for the F1-optimal decision threshold on the validation set, then reports
final metrics on the test set and saves the ROC / precision-recall curve plots.

Usage:
    python -m fraud_detection.evaluate
"""

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from tensorflow import keras

from fraud_detection.config import (
    IMAGES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    PR_CURVE_PATH,
    RESULTS_DIR,
    ROC_CURVE_PATH,
    SCALER_PATH,
)
from fraud_detection.data import load_dataset, split_data


def find_best_threshold(y_val, y_val_prob, thresholds=np.arange(0.50, 1.00, 0.01)) -> float:
    """Return the threshold in `thresholds` that maximizes validation F1."""
    best_threshold, best_f1 = 0.5, -1.0
    for threshold in thresholds:
        y_pred = (y_val_prob >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return float(best_threshold)


def main() -> None:
    model = keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    df = load_dataset()
    _X_train, X_val, X_test, _y_train, y_val, y_test = split_data(df)

    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    y_val_prob = model.predict(X_val_scaled, verbose=0).ravel()
    y_test_prob = model.predict(X_test_scaled, verbose=0).ravel()

    threshold = find_best_threshold(y_val, y_val_prob)
    y_test_pred = (y_test_prob >= threshold).astype(int)

    metrics = {
        "Threshold": threshold,
        "Precision": precision_score(y_test, y_test_pred, zero_division=0),
        "Recall": recall_score(y_test, y_test_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_test_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, y_test_prob),
        "PR-AUC": average_precision_score(y_test, y_test_prob),
    }
    print(metrics)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(METRICS_PATH, index=False)

    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ANN (AUC = {metrics['ROC-AUC']:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.grid()
    plt.savefig(ROC_CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    precision, recall, _ = precision_recall_curve(y_test, y_test_prob)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f"ANN (AP = {metrics['PR-AUC']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.grid()
    plt.savefig(PR_CURVE_PATH, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved plots to {IMAGES_DIR}")


if __name__ == "__main__":
    main()
