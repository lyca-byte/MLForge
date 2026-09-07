"""
MLForge — ml/builder.py
Constructs a Keras model from the structured JSON layer configuration
sent by the frontend. Never executes arbitrary user code.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras

# Maximum allowed parameter count to prevent runaway builds
# MAX_PARAMS = 5_000_000
MAX_PARAMS = 2_000_000

SUPPORTED_LAYERS = {
    "Conv2D", "MaxPooling2D", "AveragePooling2D",
    "Flatten", "Dense", "Dropout", "BatchNormalization",
}

SUPPORTED_ACTIVATIONS = {"relu", "sigmoid", "tanh", "softmax", "linear"}
SUPPORTED_PADDINGS    = {"same", "valid"}
SUPPORTED_OPTIMIZERS  = {"adam", "sgd", "rmsprop"}


# ── Validation ────────────────────────────────────────────────

def validate_config(
    layers: List[Dict[str, Any]],
    input_shape: Tuple[int, ...],
    num_classes: int,
    task: str,
    training_cfg: Dict[str, Any],
) -> None:
    """Raise ValueError describing any configuration problem."""
    if not layers:
        raise ValueError("Model must have at least one layer.")

    for i, layer in enumerate(layers):
        ltype = layer.get("type")
        if ltype not in SUPPORTED_LAYERS:
            raise ValueError(f"Layer {i}: unsupported type '{ltype}'.")

        if ltype == "Conv2D":
            filters = layer.get("filters", 32)
            if not (1 <= filters <= 512):
                raise ValueError(f"Layer {i} Conv2D: filters must be 1–512, got {filters}.")
            k = layer.get("kernel_size", 3)
            if not (1 <= k <= 7):
                raise ValueError(f"Layer {i} Conv2D: kernel_size must be 1–7, got {k}.")
            act = layer.get("activation", "relu")
            if act not in SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Layer {i} Conv2D: unsupported activation '{act}'.")

        elif ltype == "Dense":
            units = layer.get("units", 128)
            if not (1 <= units <= 1024):
                raise ValueError(f"Layer {i} Dense: units must be 1–1024, got {units}.")
            act = layer.get("activation", "relu")
            if act not in SUPPORTED_ACTIVATIONS:
                raise ValueError(f"Layer {i} Dense: unsupported activation '{act}'.")

        elif ltype == "Dropout":
            rate = layer.get("rate", 0.25)
            if not (0 < rate < 1):
                raise ValueError(f"Layer {i} Dropout: rate must be between 0 and 1, got {rate}.")

    # Architecture rules
    has_conv    = any(l["type"] == "Conv2D" for l in layers)
    has_dense   = any(l["type"] == "Dense" for l in layers)
    has_flatten = any(l["type"] == "Flatten" for l in layers)
    flat_idx    = next((i for i, l in enumerate(layers) if l["type"] == "Flatten"), None)
    first_dense = next((i for i, l in enumerate(layers) if l["type"] == "Dense"), None)

    if has_conv and has_dense and not has_flatten:
        raise ValueError(
            "A Flatten layer is required between convolutional and Dense layers."
        )

    if has_flatten and has_conv:
        last_conv = max(i for i, l in enumerate(layers) if l["type"] == "Conv2D")
        if last_conv > flat_idx:
            raise ValueError("Conv2D layers must appear before Flatten.")

    if has_flatten and has_dense and first_dense < flat_idx:
        raise ValueError("Dense layers must appear after Flatten.")

    # Training limits
    epochs = training_cfg.get("epochs", 10)
    if not (1 <= epochs <= 50):
        raise ValueError(f"Epochs must be between 1 and 50, got {epochs}.")

    lr = training_cfg.get("learning_rate", 0.001)
    if not (1e-6 <= lr <= 1.0):
        raise ValueError(f"Learning rate must be between 1e-6 and 1.0, got {lr}.")

    optimizer = training_cfg.get("optimizer", "adam")
    if optimizer not in SUPPORTED_OPTIMIZERS:
        raise ValueError(f"Unsupported optimizer '{optimizer}'.")


# ── Model builder ─────────────────────────────────────────────

def build_model(
    layers_cfg:  List[Dict[str, Any]],
    input_shape: Tuple[int, ...],
    num_classes: int,
    task:        str,
) -> keras.Model:
    """
    Construct and return a Keras Sequential model from the layer config.
    Automatically appends an output Dense layer with the correct
    activation (sigmoid for binary, softmax for multiclass).
    """
    model = keras.Sequential(name="MLForge_Model")
    model.add(keras.Input(shape=input_shape))

    for layer_cfg in layers_cfg:
        ltype = layer_cfg["type"]

        if ltype == "Conv2D":
            model.add(keras.layers.Conv2D(
                filters=int(layer_cfg.get("filters", 32)),
                kernel_size=int(layer_cfg.get("kernel_size", 3)),
                activation=layer_cfg.get("activation", "relu"),
                padding=layer_cfg.get("padding", "same"),
            ))

        elif ltype == "MaxPooling2D":
            model.add(keras.layers.MaxPooling2D(
                pool_size=int(layer_cfg.get("pool_size", 2)),
            ))

        elif ltype == "AveragePooling2D":
            model.add(keras.layers.AveragePooling2D(
                pool_size=int(layer_cfg.get("pool_size", 2)),
            ))

        elif ltype == "Flatten":
            model.add(keras.layers.Flatten())

        elif ltype == "Dense":
            model.add(keras.layers.Dense(
                units=int(layer_cfg.get("units", 128)),
                activation=layer_cfg.get("activation", "relu"),
            ))

        elif ltype == "Dropout":
            model.add(keras.layers.Dropout(
                rate=float(layer_cfg.get("rate", 0.25)),
            ))

        elif ltype == "BatchNormalization":
            model.add(keras.layers.BatchNormalization())

    # ── Auto output layer ─────────────────────────────────────
    if task == "binary":
        model.add(keras.layers.Dense(1, activation="sigmoid"))
    else:
        model.add(keras.layers.Dense(num_classes, activation="softmax"))

    # ── Parameter check ───────────────────────────────────────
    total = model.count_params()
    if total > MAX_PARAMS:
        raise ValueError(
            f"Model has {total:,} parameters, which exceeds the limit of {MAX_PARAMS:,}. "
            "Reduce the number of filters or Dense units."
        )

    return model


def get_model_info(model: keras.Model) -> Dict[str, Any]:
    total          = int(model.count_params())
    # TF 2.16+ may return variable objects or an empty list — use count_params breakdown
    trainable      = int(sum(w.numpy().size for w in model.trainable_weights)
                        if model.trainable_weights else 0)
    non_trainable  = total - trainable
    return {
        "num_layers":           len(model.layers),
        "total_params":         total,
        "trainable_params":     trainable,
        "non_trainable_params": max(non_trainable, 0),
    }
