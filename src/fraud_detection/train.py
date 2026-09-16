"""Train the fraud detection model and save the model + scaler artifacts.

Usage:
    python -m fraud_detection.train
"""

import joblib
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import MODEL_PATH, MODELS_DIR, SCALER_PATH
from fraud_detection.data import load_dataset, split_data
from fraud_detection.model import build_model, compute_class_weights


def main(epochs: int = 20, batch_size: int = 2048) -> None:
    df = load_dataset()
    X_train, X_val, _X_test, y_train, y_val, _y_test = split_data(df)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    class_weight = compute_class_weights(y_train)

    model = build_model(n_features=X_train.shape[1])
    model.fit(
        X_train_scaled,
        y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight,
        verbose=1,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"Saved model to {MODEL_PATH}")
    print(f"Saved scaler to {SCALER_PATH}")


if __name__ == "__main__":
    main()
