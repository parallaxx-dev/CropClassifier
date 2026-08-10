"""GET /api/v1/model-info — static, precomputed validation metrics.

Per CLAUDE.md's Stage 4 plan: "static/precomputed accuracy metrics... this
is a one-time computation, not per-request." See app/data/model_stats.py.
"""

from fastapi import APIRouter

from app.data.model_stats import MODEL_INFO

router = APIRouter(prefix="/api/v1", tags=["model-info"])


@router.get("/model-info")
def model_info() -> dict:
    return MODEL_INFO
