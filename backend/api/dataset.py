"""
MLForge — api/dataset.py
Dataset metadata endpoints.
"""

from fastapi import APIRouter, HTTPException
from ml.datasets import DATASET_REGISTRY, get_dataset_info

router = APIRouter()


@router.get("/list")
async def list_datasets():
    """Return available datasets."""
    return {
        "datasets": [
            {"id": k, **v}
            for k, v in DATASET_REGISTRY.items()
        ]
    }


@router.get("/{dataset_id}/info")
async def dataset_info(dataset_id: str):
    """Return metadata for a specific dataset."""
    try:
        info = get_dataset_info(dataset_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return info
