"""FastAPI application entrypoint."""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CHECKPOINT_PATH, DEVICE, INPUT_DIM, MODEL_PARAMS, NUM_CLASSES
from app.models.architecture import load_model
from app.routers.demo import router as demo_router
from app.routers.model_info import router as model_info_router
from app.routers.parcels import router as parcels_router
from app.routers.predict import router as predict_router
from app.routers.upload import router as upload_router


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

# Defaults to local dev origins only. Set CORS_ALLOWED_ORIGINS (comma-
# separated) in the deployed environment to the real frontend URL -- e.g.
# an ECS task definition's environment block -- or every request from a
# deployed frontend gets silently blocked by the browser.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
allowed_origins = [
    origin.strip() for origin in os.environ.get("CORS_ALLOWED_ORIGINS", _default_origins).split(",")
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(model_info_router)
app.include_router(parcels_router)
app.include_router(upload_router)
app.include_router(demo_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
