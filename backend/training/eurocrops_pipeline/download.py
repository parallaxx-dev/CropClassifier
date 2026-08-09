"""
download.py — per-country zip download + extraction + cleanup.

_download_with_resume is copied verbatim from train_eurocrops_local.py (already
solid — handles HTTP Range resume and transient connection drops, empirically
necessary per that script's own comment about a ~2.6GB download dropping mid-
stream). Everything else here is new: routes raw downloads through SCRATCH_DIR
(not the 20GB persistent tier — see config.py), and cleanup is a single function
called immediately after a country's parcels are sampled, not deferred.
"""

import shutil
import time
import zipfile
from pathlib import Path

import requests

from config import CountryEntry, SCRATCH_DIR, zenodo_url


def _download_with_resume(url: str, dest_path: Path, max_retries: int = 12) -> None:
    for attempt in range(1, max_retries + 1):
        existing_size = dest_path.stat().st_size if dest_path.exists() else 0
        headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}
        try:
            with requests.get(url, stream=True, headers=headers, timeout=60) as resp:
                if existing_size and resp.status_code == 416:
                    return
                if existing_size and resp.status_code == 200:
                    existing_size = 0
                resp.raise_for_status()
                mode = "ab" if existing_size and resp.status_code == 206 else "wb"
                with open(dest_path, mode) as out:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        out.write(chunk)
                content_length = resp.headers.get("Content-Length")

            if content_length is not None:
                expected_size = existing_size + int(content_length) if mode == "ab" else int(content_length)
                actual_size = dest_path.stat().st_size
                if actual_size < expected_size:
                    raise IOError(f"download incomplete: got {actual_size} bytes, expected {expected_size}")
            return
        except Exception as e:
            print(f"  download attempt {attempt}/{max_retries} failed: {e}")
            if attempt == max_retries:
                raise
            wait = min(60, 2**attempt)
            print(f"  retrying in {wait}s...")
            time.sleep(wait)


def download_and_extract(country: CountryEntry) -> Path:
    """Download + extract one country's zip into scratch space. Returns the path
    to the .shp file. Caller is responsible for calling cleanup() once this
    country's parcels have been sampled — leaving extracted data around across
    countries is exactly the gap this pipeline fixes relative to the single-
    country script (which never cleaned up the extracted directory at all)."""
    country_dir = SCRATCH_DIR / country.code
    zip_path = SCRATCH_DIR / f"{country.code}.zip"
    country_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{country.code}] downloading {country.zenodo_filename} (~{country.approx_size_mb}MB)...")
    _download_with_resume(zenodo_url(country.zenodo_filename), zip_path)

    print(f"[{country.code}] extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(country_dir)

    shp_path = None
    for p in country_dir.rglob("*.shp"):
        shp_path = p
        break
    if shp_path is None:
        raise FileNotFoundError(f"[{country.code}] no .shp found after extracting {zip_path}")
    return shp_path


def cleanup(country: CountryEntry) -> None:
    """Delete this country's extracted shapefile + zip. Called immediately after
    sampling, inside the per-country loop — bounds peak disk usage to roughly one
    country's footprint regardless of how many countries get processed in total."""
    country_dir = SCRATCH_DIR / country.code
    zip_path = SCRATCH_DIR / f"{country.code}.zip"
    if country_dir.exists():
        shutil.rmtree(country_dir)
    if zip_path.exists():
        zip_path.unlink()
    print(f"[{country.code}] cleaned up scratch space")
