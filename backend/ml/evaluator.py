"""
MLForge — ml/evaluator.py
Computes evaluation metrics on the test set after training.
"""

from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from tensorflow import keras


def evaluate_model(
    model: keras.Model,
    x_test: np.ndarray,
    y_test: np.ndarray,
    ds_cfg: Dict[str, Any],
    num_classes: int,
) -> Dict[str, Any]:
    """
    Run inference on x_test, compute metrics, and return a metrics dict
    that can be serialised to JSON and returned to the frontend.
    """
    task = ds_cfg.get("task", "multiclass")

    # Get predictions
    # raw_preds = model.predict(x_test, verbose=0)
    raw_preds = model.predict(x_test, batch_size=32, verbose=0,)

    if task == "binary":
        # sigmoid output → threshold at 0.5; ensure 1-D array even for single sample
        squeezed = raw_preds.squeeze()
        y_pred   = (np.atleast_1d(squeezed) >= 0.5).astype(int)
        average  = "binary"
    else:
        y_pred  = np.argmax(raw_preds, axis=1)
        average = "macro"

    y_true = np.atleast_1d(y_test.astype(int))

    acc       = float(accuracy_score(y_true, y_pred))
    precision = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    recall    = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    f1        = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    cm        = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    classes      = ds_cfg.get("classes", list(range(num_classes)))
    class_labels = ds_cfg.get("class_labels") or {}

    per_class: Dict[str, Any] = {}
    for i, orig_cls in enumerate(classes):
        # class_labels keys may be int or str depending on JSON round-trip
        label = str(
            class_labels.get(str(orig_cls))
            or class_labels.get(int(orig_cls) if str(orig_cls).isdigit() else orig_cls)
            or orig_cls
        )
        row   = report.get(str(i), {})
        per_class[str(orig_cls)] = {
            "precision": float(row.get("precision", 0)),
            "recall":    float(row.get("recall",    0)),
            "f1":        float(row.get("f1-score",  0)),
            "support":   int(row.get("support",     0)),
            "label":     label,
        }

    return {
        "accuracy":         acc,
        "precision":        precision,
        "recall":           recall,
        "f1":               f1,
        "confusion_matrix": cm,
        "per_class":        per_class,
    }
