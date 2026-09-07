"""
MLForge — api/export.py
Model and metadata export endpoints.
Serves files as downloads without exposing filesystem paths.
"""

from __future__ import annotations

import json
import logging
import os

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

# from ml.trainer  import get_job
# from ml.exporter import export_tflite, get_keras_path

from backend.ml.trainer import get_job
from backend.ml.exporter import export_tflite, get_keras_path

router = APIRouter()
logger = logging.getLogger("mlforge.export")

EXPORT_TYPES = {"keras", "tflite", "architecture", "history", "evaluation"}


@router.get("/{export_type}/{job_id}")
async def export_model(export_type: str, job_id: str):
    if export_type not in EXPORT_TYPES:
        raise HTTPException(status_code=400, detail=f"Unknown export type '{export_type}'.")

    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail="Training not complete yet.")

    # ── .keras file ───────────────────────────────────────────
    if export_type == "keras":
        path = get_keras_path(job_id)
        if not os.path.exists(path):
            raise HTTPException(status_code=404, detail="Model file not found.")
        return FileResponse(
            path=path,
            media_type="application/octet-stream",
            filename=f"mlforge_model_{job_id[:8]}.keras",
        )

    # ── .tflite file ──────────────────────────────────────────
    elif export_type == "tflite":
        try:
            path = export_tflite(job_id)
        except FileNotFoundError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            logger.exception("TFLite conversion failed for job %s", job_id)
            raise HTTPException(status_code=500, detail=f"TFLite conversion failed: {e}")

        return FileResponse(
            path=path,
            media_type="application/octet-stream",
            filename=f"mlforge_model_{job_id[:8]}.tflite",
        )

    # ── JSON exports ──────────────────────────────────────────
    elif export_type == "architecture":
        # The model architecture is stored inside the job payload
        # Re-derive from the keras model's config
        try:
            from tensorflow import keras as _keras
            kpath = get_keras_path(job_id)
            model = _keras.models.load_model(kpath)
            data  = json.loads(model.to_json())
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not load model: {e}")

        return Response(
            content=json.dumps(data, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="mlforge_architecture_{job_id[:8]}.json"'},
        )

    elif export_type == "history":
        history = job.get("history")
        if not history:
            raise HTTPException(status_code=404, detail="Training history not available.")
        return Response(
            content=json.dumps(history, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="mlforge_history_{job_id[:8]}.json"'},
        )

    elif export_type == "evaluation":
        metrics = job.get("metrics")
        if not metrics:
            raise HTTPException(status_code=404, detail="Evaluation results not available.")
        return Response(
            content=json.dumps(metrics, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f'attachment; filename="mlforge_evaluation_{job_id[:8]}.json"'},
        )
