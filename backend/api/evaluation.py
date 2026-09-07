# """
# MLForge — api/evaluation.py
# Returns evaluation results for a completed job.
# """

# from fastapi import APIRouter, HTTPException
# # from ml.trainer import get_job
# from backend.ml.trainer import get_job

# router = APIRouter()


# @router.get("/{job_id}")
# async def get_evaluation(job_id: str):
#     """Return evaluation metrics for a completed training job."""
#     job = get_job(job_id)
#     if not job:
#         raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
#     if job["status"] != "completed":
#         raise HTTPException(status_code=400, detail=f"Job is not completed (status: {job['status']}).")
#     if not job.get("metrics"):
#         raise HTTPException(status_code=404, detail="Evaluation results not available.")
#     return job["metrics"]



"""
MLForge — api/evaluation.py
Returns evaluation results for a completed job.
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = BASE_DIR / "models"


@router.get("/{job_id}")
async def get_evaluation(job_id: str):
    """Return evaluation metrics for a completed training job."""

    evaluation_path = (
        MODEL_DIR
        / job_id
        / "evaluation.json"
    )

    if not evaluation_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Evaluation results for job '{job_id}' not found."
        )

    try:
        with open(
            evaluation_path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read evaluation results: {e}"
        )