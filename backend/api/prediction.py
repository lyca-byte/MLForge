"""
MLForge — api/prediction.py
Image upload and model inference.
"""

from __future__ import annotations

import io
import logging

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image
from tensorflow import keras

# from ml.trainer import get_job
from backend.ml.trainer import get_job

router  = APIRouter()
logger  = logging.getLogger("mlforge.prediction")

# Cache loaded models to avoid re-loading on every request
_MODEL_CACHE: dict = {}


def _load_model(job_id: str) -> keras.Model:
    if job_id not in _MODEL_CACHE:
        import os
        path = f"models/{job_id}.keras"
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found for job {job_id}.")
        _MODEL_CACHE[job_id] = keras.models.load_model(path)
    return _MODEL_CACHE[job_id]


def _preprocess_image(
    image_bytes: bytes,
    input_shape: tuple,
) -> np.ndarray:
    """
    Convert raw image bytes to a normalised numpy array matching input_shape.
    input_shape = (H, W, C)
    """
    img = Image.open(io.BytesIO(image_bytes))

    h, w, c = input_shape
    mode = "L" if c == 1 else "RGB"
    img  = img.convert(mode).resize((w, h), Image.LANCZOS)

    arr = np.array(img, dtype=np.float32) / 255.0
    if c == 1:
        arr = arr[..., np.newaxis]
    # Add batch dimension
    return arr[np.newaxis, ...]


@router.post("/predict")
async def predict(
    image:  UploadFile = File(...),
    job_id: str        = Form(...),
):
    # ── Validate job ─────────────────────────────────────────
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Model training is not complete.")

    # ── Validate image ────────────────────────────────────────
    allowed = {"image/png", "image/jpeg", "image/bmp", "image/gif", "image/webp"}
    if image.content_type not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported image format '{image.content_type}'. "
                   f"Please upload PNG, JPG, BMP, or WebP."
        )

    content = await image.read()
    if len(content) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large. Maximum size is 10 MB.")

    # ── Load model & infer ────────────────────────────────────
    try:
        model = _load_model(job_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    ds_cfg      = job.get("_payload_dataset") or {}
    raw_shape   = ds_cfg.get("input_shape", [28, 28, 1])
    input_shape = tuple(raw_shape)

    try:
        arr = _preprocess_image(content, input_shape)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not process image: {e}")

    raw = model.predict(arr, verbose=0)

    classes = list(range(raw.shape[-1])) if raw.shape[-1] > 1 else [0, 1]

    if raw.shape[-1] == 1:
        # binary
        conf = float(raw[0, 0])
        pred_idx = 1 if conf >= 0.5 else 0
        probs = {str(0): 1.0 - conf, str(1): conf}
    else:
        probs_arr = raw[0]
        pred_idx  = int(np.argmax(probs_arr))
        conf      = float(probs_arr[pred_idx])
        probs     = {str(i): float(p) for i, p in enumerate(probs_arr)}

    return {
        "predicted_class": pred_idx,
        "confidence":      conf,
        "probabilities":   probs,
    }
