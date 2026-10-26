# GDAL-inclusive base image -- rasterio/geopandas (parcel polygon handling,
# GeoTIFF AOI uploads) need real GDAL underneath and are painful to build
# from scratch. Matches the base image already decided on in CLAUDE.md's
# Tech Stack Decisions.
FROM ghcr.io/osgeo/gdal:ubuntu-small-3.8.0

# The base image ships GDAL's own Python bindings but not pip -- installed
# explicitly rather than assumed.
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, app code after -- so `docker build` only re-installs
# packages when requirements.txt actually changes, not on every code edit.
COPY requirements.txt .

# Plain `pip install torch` pulls the default CUDA-inclusive build (2GB+)
# even though this model is tiny and runs CPU-only in production (see
# INFERENCE_ENSEMBLE_SIZE-draw inference in app/models/inference.py -- no
# GPU needed for a few dozen ms of transformer forward passes). The
# CPU-only wheel index cuts both image size and build time substantially.
# Installed before the rest of requirements.txt so that install is a no-op
# (already satisfies torch>=2.0.0) instead of overwriting this with the
# default build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

# App code + the one checkpoint actually loaded at startup (app/config.py's
# CHECKPOINT_PATH). The other legacy checkpoints in backend/ are excluded
# via .dockerignore -- not needed to serve a prediction.
COPY app/ ./app/
COPY multicountry_india_custom_18class.pth .

# Parcel geometry data for the Parcel Browser / Live Validation pages
# (training/eurocrops_output/.../*_sampled_geoms.geojson and local_parcels/)
# is deliberately NOT baked in: both paths are gitignored (local_parcels/ is
# private), so COPYing them made the image unbuildable from a clean clone
# (CI, a fresh EC2 box). docker-compose.yml bind-mounts them at runtime
# instead; services/parcels.py and services/custom_parcels.py already skip
# missing files, so the API still starts (with fewer parcel regions) when
# they're absent.

# CDSE_SH_CLIENT_ID / CDSE_SH_CLIENT_SECRET are read from the environment
# (app/config.py's load_dotenv() is a no-op if backend/.env doesn't exist,
# which it deliberately won't in this image -- see .dockerignore). Inject
# them at run time (docker-compose.yml's env_file: backend/.env, or
# docker run --env-file) -- never bake real credentials into the image.
EXPOSE 8000

# --proxy-headers: requests arrive via the frontend's nginx reverse proxy,
# so trust its X-Forwarded-* headers for client IP / scheme.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
