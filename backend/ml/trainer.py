"""
MLForge — ml/trainer.py
Manages training jobs in memory with thread-based background execution.
Each job is identified by a UUID and its progress is polled by the frontend.
"""

from __future__ import annotations

import threading
import uuid
import time
import logging
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras

from pathlib import Path

# from ml.datasets import load_dataset
# from ml.builder  import build_model, validate_config, get_model_info
# from ml.evaluator import evaluate_model

from backend.ml.datasets import load_dataset
from backend.ml.builder import build_model, validate_config, get_model_info
from backend.ml.evaluator import evaluate_model

logger = logging.getLogger("mlforge.trainer")

# Root directory project MLForge
BASE_DIR = Path(__file__).resolve().parents[2]

# Directory untuk menyimpan model hasil training
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ── In-memory job store ───────────────────────────────────────
# Maps job_id → job dict
_JOBS: Dict[str, Dict[str, Any]] = {}

# Limit concurrent jobs (free resource constraint)
# MAX_CONCURRENT_JOBS = 2
MAX_CONCURRENT_JOBS = 1


def _get_active_count() -> int:
    return sum(1 for j in _JOBS.values() if j["status"] in ("queued", "training"))


# ── Progress callback ─────────────────────────────────────────

class _ProgressCallback(keras.callbacks.Callback):  # type: ignore[misc]
    def __init__(self, job_id: str, total_epochs: int) -> None:
        super().__init__()
        self.job_id       = job_id
        self.total_epochs = total_epochs

    def on_epoch_end(self, epoch: int, logs: Optional[Dict] = None) -> None:
        if logs is None:
            logs = {}
        job = _JOBS.get(self.job_id)
        if not job:
            return

        history = job.setdefault("history", {
            "accuracy": [], "val_accuracy": [],
            "loss": [],     "val_loss": [],
        })
        history["accuracy"].append(float(logs.get("accuracy", 0)))
        history["val_accuracy"].append(float(logs.get("val_accuracy", 0)))
        history["loss"].append(float(logs.get("loss", 0)))
        history["val_loss"].append(float(logs.get("val_loss", 0)))

        job["message"] = (
            f"Epoch {epoch + 1}/{self.total_epochs} — "
            f"acc: {logs.get('accuracy', 0):.4f}, "
            f"val_acc: {logs.get('val_accuracy', 0):.4f}"
        )
        logger.info(job["message"])

        # Cancellation check
        if job.get("cancelled"):
            self.model.stop_training = True


# ── Training thread ───────────────────────────────────────────

def _run_training(job_id: str, payload: Dict[str, Any]) -> None:
    job = _JOBS[job_id]

    try:
        job["status"]  = "training"
        job["message"] = "Loading dataset…"

        ds_cfg   = payload["dataset"]
        model_cfg = payload["model"]
        train_cfg = payload["training"]

        # Load data
        x_train, y_train, x_test, y_test = load_dataset(
            dataset_id=ds_cfg["id"],
            selected_classes=ds_cfg["classes"],
            task=ds_cfg["task"],
        )

        num_classes = len(ds_cfg["classes"])
        input_shape = tuple(ds_cfg["input_shape"])

        # Build & compile model
        model = build_model(
            layers_cfg=model_cfg["layers"],
            input_shape=input_shape,
            num_classes=num_classes,
            task=ds_cfg["task"],
        )

        optimizer_map = {
            "adam":    keras.optimizers.Adam(learning_rate=train_cfg["learning_rate"]),
            "sgd":     keras.optimizers.SGD(learning_rate=train_cfg["learning_rate"]),
            "rmsprop": keras.optimizers.RMSprop(learning_rate=train_cfg["learning_rate"]),
        }
        optimizer = optimizer_map[train_cfg["optimizer"]]

        if ds_cfg["task"] == "binary":
            loss = "binary_crossentropy"
            y_train_fit = y_train.astype(np.float32)
            y_test_fit  = y_test.astype(np.float32)
        else:
            loss = "sparse_categorical_crossentropy"
            y_train_fit = y_train
            y_test_fit  = y_test

        model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        job["message"] = "Model compiled. Starting training…"

        callbacks = [
            _ProgressCallback(job_id=job_id, total_epochs=train_cfg["epochs"]),
        ]

        model.fit(
            x_train, y_train_fit,
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            validation_split=train_cfg["validation_split"],
            callbacks=callbacks,
            verbose=0,
        )

        if job.get("cancelled"):
            job["status"]  = "failed"
            job["message"] = "Training cancelled."
            return

        job["message"] = "Training complete. Evaluating…"

        # Evaluate
        metrics = evaluate_model(model, x_test, y_test_fit, ds_cfg, num_classes)
        metrics["model_info"] = get_model_info(model)

        # Save model
        import os
        os.makedirs("models", exist_ok=True)
        model_path = f"models/{job_id}.keras"
        model.save(model_path)

        job["model_path"]        = model_path
        job["metrics"]           = metrics
        job["status"]            = "completed"
        job["message"]           = f"Completed — {train_cfg['epochs']} epochs"
        # Store dataset config so prediction endpoint knows input shape & classes
        job["_payload_dataset"]  = payload["dataset"]
        logger.info("Job %s completed.", job_id)

    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        logger.error("Job %s failed:\n%s", job_id, tb)
        job["status"]  = "failed"
        job["message"] = str(exc)


# ── Public API ────────────────────────────────────────────────

def create_job(payload: Dict[str, Any]) -> str:
    if _get_active_count() >= MAX_CONCURRENT_JOBS:
        raise RuntimeError(
            "Too many concurrent training jobs. Please wait for an existing job to finish."
        )

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "job_id":     job_id,
        "status":     "queued",
        "message":    "Queued — waiting to start…",
        "metrics":    None,
        "model_path": None,
        "cancelled":  False,
        # history is intentionally absent so setdefault initialises it correctly
    }

    thread = threading.Thread(target=_run_training, args=(job_id, payload), daemon=True)
    thread.start()
    return job_id


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _JOBS.get(job_id)


def cancel_job(job_id: str) -> bool:
    job = _JOBS.get(job_id)
    if not job:
        return False
    job["cancelled"] = True
    return True
