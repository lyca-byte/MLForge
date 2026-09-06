"""
MLForge — ml/exporter.py
Exports trained models to .keras and .tflite formats.
"""

from __future__ import annotations

import os
import logging
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow import keras

logger = logging.getLogger("mlforge.exporter")


def get_keras_path(job_id: str) -> str:
    return f"models/{job_id}.keras"


def get_tflite_path(job_id: str) -> str:
    return f"models/{job_id}.tflite"


def export_tflite(job_id: str) -> str:
    """
    Convert the saved .keras model to TFLite.
    Returns the path to the .tflite file.
    """
    keras_path  = get_keras_path(job_id)
    tflite_path = get_tflite_path(job_id)

    if not os.path.exists(keras_path):
        raise FileNotFoundError(f"Trained model not found for job {job_id}.")

    # If already converted, return cached
    if os.path.exists(tflite_path):
        return tflite_path

    model      = keras.models.load_model(keras_path)
    converter  = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_buf = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_buf)

    logger.info("TFLite export saved: %s", tflite_path)
    return tflite_path
