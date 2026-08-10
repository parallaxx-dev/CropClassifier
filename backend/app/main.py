"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import CHECKPOINT_PATH, DEVICE, INPUT_DIM, MODEL_PARAMS, NUM_CLASSES
from app.models.architecture import load_model
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

# Local dev only: Vite's default dev server port. Tighten this before any
# real deployment -- wide open just so the frontend can reach the API while
# both run locally.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict_router)
app.include_router(model_info_router)
app.include_router(parcels_router)
app.include_router(upload_router)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
