"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import CHECKPOINT_PATH, DEVICE, INPUT_DIM, MODEL_PARAMS, NUM_CLASSES
from app.models.architecture import load_model
from app.routers.predict import router as predict_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loaded once here, kept in app.state for the life of the process — never
    # reloaded per-request.
    app.state.model = load_model(
        checkpoint_path=CHECKPOINT_PATH,
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        device=DEVICE,
        model_params=MODEL_PARAMS,
    )
    yield
    del app.state.model


app = FastAPI(title="Crop Classifier API", lifespan=lifespan)
app.include_router(predict_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
