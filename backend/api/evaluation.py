"""
MLForge — api/evaluation.py
Returns evaluation results for a completed job.
"""

from fastapi import APIRouter, HTTPException
from ml.trainer import get_job

router = APIRouter()


@router.get("/{job_id}")
async def get_evaluation(job_id: str):
    """Return evaluation metrics for a completed training job."""
    job = get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    if job["status"] != "completed":
        raise HTTPException(status_code=400, detail=f"Job is not completed (status: {job['status']}).")
    if not job.get("metrics"):
        raise HTTPException(status_code=404, detail="Evaluation results not available.")
    return job["metrics"]
