"""GET /api/v1/parcels — real, pre-outlined field boundaries for the parcel-
browser page. GET /api/v1/parcels/regions lists what's available."""

from fastapi import APIRouter

from app.services import parcels as parcels_service

router = APIRouter(prefix="/api/v1", tags=["parcels"])


@router.get("/parcels/regions")
def regions() -> list[dict]:
    return parcels_service.list_regions()


@router.get("/parcels")
def parcels(region: str | None = None, limit: int = 40) -> dict:
    return parcels_service.get_parcels_geojson(region=region, limit=limit)
