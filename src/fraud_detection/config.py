"""Shared paths, feature schema, and constants used across the pipeline."""

from pathlib import Path

# Project root (two levels up from this file: src/fraud_detection/config.py -> project root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = BASE_DIR / "data" / "raw"
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"

MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "fraud_detection_ann.keras"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

RESULTS_DIR = BASE_DIR / "results"
METRICS_PATH = RESULTS_DIR / "evaluation_metrics.csv"

IMAGES_DIR = BASE_DIR / "images"
ROC_CURVE_PATH = IMAGES_DIR / "roc_curve.png"
PR_CURVE_PATH = IMAGES_DIR / "precision_recall_curve.png"

# Feature order used everywhere the model expects a fixed-shape input.
FEATURE_NAMES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)
TARGET_NAME = "Class"

RANDOM_STATE = 42

# Fraction held out for the test set, and (of what remains) for the validation set.
TEST_SIZE = 0.2
VAL_SIZE = 0.2

# Threshold selected via F1-optimal search on the validation set (see notebooks/04).
DECISION_THRESHOLD = 0.99
