"""Stub out `tensorflow` so the test suite doesn't need a real (heavy) TF install.

Tests never exercise actual model training/inference against TensorFlow directly —
they either test pure data-splitting logic or mock the model/scaler that
`fraud_detection.predict` loads. Stubbing here keeps the suite fast and avoids
environment-specific TensorFlow install issues.
"""

import sys
import types
from unittest.mock import MagicMock


def _install_tensorflow_stub() -> None:
    if getattr(sys.modules.get("tensorflow"), "__fraud_detection_stub__", False):
        return

    layers_stub = types.ModuleType("tensorflow.keras.layers")
    layers_stub.Input = MagicMock(name="Input")
    layers_stub.Dense = MagicMock(name="Dense")
    layers_stub.Dropout = MagicMock(name="Dropout")

    models_stub = types.SimpleNamespace(load_model=MagicMock(name="load_model"))

    keras_stub = types.ModuleType("tensorflow.keras")
    keras_stub.Sequential = MagicMock(name="Sequential")
    keras_stub.Model = MagicMock(name="Model")
    keras_stub.models = models_stub
    keras_stub.layers = layers_stub
    keras_stub.metrics = types.SimpleNamespace(
        Precision=MagicMock(name="Precision"), Recall=MagicMock(name="Recall")
    )

    tf_stub = types.ModuleType("tensorflow")
    tf_stub.__fraud_detection_stub__ = True
    tf_stub.keras = keras_stub

    sys.modules["tensorflow"] = tf_stub
    sys.modules["tensorflow.keras"] = keras_stub
    sys.modules["tensorflow.keras.layers"] = layers_stub


_install_tensorflow_stub()
