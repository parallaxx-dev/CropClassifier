"""POST /api/v1/predict — classify a field parcel's crop type from its
Sentinel-2 growing-season time series."""

from fastapi import APIRouter, HTTPException, Request

from app.config import TARGET_SEQ_LEN
from app.models.inference import predict as run_prediction
from app.schemas import PredictRequest, PredictResponse
from app.services import sentinel_fetch, validation

router = APIRouter(prefix="/api/v1", tags=["predict"])


@router.post("/predict", response_model=PredictResponse)
def predict_crop(payload: PredictRequest, request: Request) -> PredictResponse:
    polygon = validation.validate_aoi(payload.aoi.model_dump())
    validation.validate_date_range(payload.start_date, payload.end_date)

    x_raw, dates = sentinel_fetch.fetch_time_series(
        polygon, payload.start_date, payload.end_date
    )
    if len(dates) == 0:
        raise HTTPException(
            422,
            "No usable Sentinel-2 scenes found for this AOI and date range — "
            "try widening the date range or lowering the cloud-cover filter.",
        )

    result = run_prediction(request.app.state.model, x_raw)

    return PredictResponse(
        predicted_class=result["pred_label"],
        confidence=result["confidence"],
        probabilities=result["probs"],
        observations_used=min(len(dates), TARGET_SEQ_LEN),
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
