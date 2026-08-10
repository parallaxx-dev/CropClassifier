"""GET /api/v1/demo/parcels — a small, spatially-clustered sample of parcels
per trained region with both true and (cached) predicted labels, for the
"Live Validation" map tab. POST /api/v1/demo/repredict re-runs prediction
live for that same set on demand, so a professor can confirm the displayed
predictions are real rather than hardcoded."""

from fastapi import APIRouter, Request

from app.services import demo_parcels

router = APIRouter(prefix="/api/v1", tags=["demo"])


@router.get("/demo/parcels")
def get_demo_parcels() -> dict:
    return demo_parcels.get_demo_geojson()


@router.post("/demo/repredict")
def repredict_demo_parcels(request: Request) -> dict:
    parcels = demo_parcels.select_demo_parcels()
    demo_parcels.run_predictions(parcels, request.app.state.model)
    return demo_parcels.get_demo_geojson()
