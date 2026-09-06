"""
MLForge — api/training.py
Training job management endpoints.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ml.builder import validate_config
from ml.trainer import create_job, get_job, cancel_job

router = APIRouter()


# ── Request models ────────────────────────────────────────────

class DatasetConfig(BaseModel):
    id:            str
    name:          str
    task:          str               # "binary" | "multiclass"
    classes:       List[int]
    class_labels:  Dict[str, str]    = Field(default_factory=dict)
    input_shape:   List[int]
    train_samples: int               = 60000
    test_samples:  int               = 10000


class LayerConfig(BaseModel):
    type: str
    # All other fields are optional and passed through
    class Config:
        extra = "allow"


class ModelConfig(BaseModel):
    layers: List[Dict[str, Any]]


class TrainingConfig(BaseModel):
    epochs:           int   = Field(default=10, ge=1, le=50)
    batch_size:       int   = Field(default=64, ge=8, le=256)
    learning_rate:    float = Field(default=0.001, ge=1e-6, le=1.0)
    optimizer:        str   = "adam"
    validation_split: float = Field(default=0.1, ge=0.05, le=0.3)


class StartTrainingRequest(BaseModel):
    dataset:  DatasetConfig
    model:    ModelConfig
    training: TrainingConfig


# ── Endpoints ─────────────────────────────────────────────────

@router.post("/start")
async def start_training(req: StartTrainingRequest):
    """Validate configuration and start a background training job."""
    ds       = req.dataset
    model    = req.model
    training = req.training

    # Validate before spawning job
    try:
        validate_config(
            layers=model.layers,
            input_shape=tuple(ds.input_shape),
            num_classes=len(ds.classes),
            task=ds.task,
            training_cfg=training.dict(),
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    try:
        job_id = create_job({
            "dataset":  ds.dict(),
            "model":    model.dict(),
            "training": training.dict(),
        })
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return {"job_id": job_id, "status": "queued"}


@router.get("/status/{job_id}")
async def training_status(job_id: str):
    """Poll the status and partial results of a training job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job


@router.post("/cancel/{job_id}")
async def cancel_training(job_id: str):
    """Request cancellation of a running job."""
    if not cancel_job(job_id):
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return {"job_id": job_id, "cancelled": True}
