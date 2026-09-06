"""
MLForge — ml/datasets.py
Dataset loading, filtering, and metadata.
"""

from __future__ import annotations

import numpy as np
from typing import List, Dict, Any, Tuple

# ── Dataset registry ──────────────────────────────────────────
DATASET_REGISTRY: Dict[str, Dict[str, Any]] = {
    "mnist": {
        "name":          "MNIST",
        "input_shape":   (28, 28, 1),
        "num_classes":   10,
        "train_samples": 60000,
        "test_samples":  10000,
    },
}


def get_dataset_info(dataset_id: str) -> Dict[str, Any]:
    info = DATASET_REGISTRY.get(dataset_id.lower())
    if not info:
        raise ValueError(f"Unsupported dataset: {dataset_id}")
    return info


def load_dataset(
    dataset_id: str,
    selected_classes: List[int],
    task: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and filter a dataset to the selected classes.

    Returns:
        x_train, y_train, x_test, y_test
        y labels are remapped to 0..N-1 for the selected classes.
    """
    dataset_id = dataset_id.lower()

    if dataset_id == "mnist":
        from tensorflow.keras.datasets import mnist  # type: ignore
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # Add channel dim → (N, 28, 28, 1)
        x_train = x_train[..., np.newaxis].astype(np.float32) / 255.0
        x_test  = x_test[...,  np.newaxis].astype(np.float32) / 255.0
    else:
        raise ValueError(f"Unsupported dataset: {dataset_id}")

    # ── Filter to selected classes ────────────────────────────
    selected = sorted(selected_classes)
    label_map = {orig: new for new, orig in enumerate(selected)}

    train_mask = np.isin(y_train, selected)
    test_mask  = np.isin(y_test,  selected)

    x_train = x_train[train_mask]
    y_train = np.array([label_map[y] for y in y_train[train_mask]], dtype=np.int32)
    x_test  = x_test[test_mask]
    y_test  = np.array([label_map[y] for y in y_test[test_mask]],  dtype=np.int32)

    return x_train, y_train, x_test, y_test
