"""
MLForge Backend — main.py
FastAPI application entry point.
"""

import os
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# from api.dataset    import router as dataset_router
# from api.training   import router as training_router
# from api.evaluation import router as evaluation_router
# from api.prediction import router as prediction_router
# from api.export     import router as export_router

from backend.api.dataset import router as dataset_router
from backend.api.training import router as training_router
from backend.api.evaluation import router as evaluation_router
from backend.api.prediction import router as prediction_router
from backend.api.export import router as export_router

# ── Logging ────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("mlforge")

# ── App ────────────────────────────────────────────────────────
app = FastAPI(
    title="MLForge API",
    description="Backend API for the MLForge No-Code Machine Learning Platform",
    version="1.0.0",
)

# ── CORS — allow frontend dev server / file:// ─────────────────
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
    "https://ml-forge-eta.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



# ── Routers ────────────────────────────────────────────────────
app.include_router(dataset_router,    prefix="/api/dataset",    tags=["Dataset"])
app.include_router(training_router,   prefix="/api/training",   tags=["Training"])
app.include_router(evaluation_router, prefix="/api/evaluation", tags=["Evaluation"])
app.include_router(prediction_router, prefix="/api/prediction", tags=["Prediction"])
app.include_router(export_router,     prefix="/api/export",     tags=["Export"])

# ── Serve model artefacts (for internal API use only, no path leak) ─
os.makedirs("models", exist_ok=True)

@app.get("/health")
async def health():
    return {"status": "ok", "service": "MLForge API"}
