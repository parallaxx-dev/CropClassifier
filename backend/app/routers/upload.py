"""POST /api/v1/upload-aoi — turn an uploaded geospatial file (vector or
raster, whatever GDAL/geopandas/rasterio can read) into a single GeoJSON AOI
polygon, for the Draw AOI page's "or upload a file" input. Does not run
inference itself -- the frontend feeds the returned polygon into the same
POST /predict flow a hand-drawn AOI already uses."""

from fastapi import APIRouter, HTTPException, UploadFile

from app.services.aoi_upload import MAX_UPLOAD_BYTES, extract_aoi_from_file, save_upload_to_tempfile

router = APIRouter(prefix="/api/v1", tags=["upload"])


@router.post("/upload-aoi")
async def upload_aoi(file: UploadFile) -> dict:
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB upload limit")
    if len(contents) == 0:
        raise HTTPException(400, "Uploaded file is empty")

    tmp_dir, tmp_path = save_upload_to_tempfile(contents, file.filename or "upload")
    try:
        return extract_aoi_from_file(tmp_path, file.filename or "upload")
    finally:
        tmp_dir.cleanup()
