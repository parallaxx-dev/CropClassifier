"""Request/response models for the prediction API."""

from datetime import date
from typing import Literal

from pydantic import BaseModel, field_validator


class AOIPolygon(BaseModel):
    """A field boundary as GeoJSON — coordinates in [lon, lat], WGS84."""

    type: Literal["Polygon"]
    coordinates: list[list[list[float]]]


class PredictRequest(BaseModel):
    aoi: AOIPolygon
    start_date: date
    end_date: date

    @field_validator("end_date")
    @classmethod
    def end_after_start(cls, end_date: date, info) -> date:
        start_date = info.data.get("start_date")
        if start_date is not None and end_date <= start_date:
            raise ValueError("end_date must be after start_date")
        return end_date


class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]
    observations_used: int
    start_date: date
    end_date: date
    partial_range_warning: str | None = None
    area_hectares: float
