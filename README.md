# Text-to-Polygon Pipeline

Zero-shot text-to-polygon pipeline for Bavarian DOP20 orthophotos. The system
downloads raw RGB/NIR imagery via LDBV WCS, runs SAM 3.1 segmentation per tile,
optionally filters masks with NDVI/nDSM, merges duplicate polygons, and exposes
the result in a local FastAPI review UI.

## Hardware

For reproducible pipeline runs:

- NVIDIA GPU with CUDA support.
- At least 12 GB VRAM for small AOIs; 16 GB or more is recommended.
- 32 GB system RAM recommended.
- 10 GB free disk space for model cache, DOP tiles, DOM/DGM cache, and job output.
- Windows 11 or Linux with Python 3.12.

The unit tests run without GPU, SAM weights, or LDBV credentials.

## External Access

You need:

- LDBV WCS credentials for DOP20 download.
- Hugging Face access to `facebook/sam3`.
- A local or cached SAM 3.1 model snapshot when `SAM3_LOCAL_FILES_ONLY=true`.

## Install

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .[dev]
```

Install CUDA PyTorch for your CUDA runtime. Example for CUDA 12.8:

```powershell
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Install or refresh the SAM/Transformers dependencies if needed:

```powershell
python -m pip install transformers accelerate
huggingface-cli login
```

On systems where Fiona/GDAL wheels are unavailable, install geospatial packages
through conda-forge first, then run `pip install -e .[dev]`:

```powershell
conda create -n ki-geodaten python=3.12 fiona rasterio geopandas -c conda-forge
conda activate ki-geodaten
python -m pip install -e .[dev]
```

## Configure

Create `.env` from `.env.example`:

```powershell
Copy-Item .env.example .env
```

Set at least these values:

```text
WCS_USERNAME=...
WCS_PASSWORD=...
SAM3_MODEL_ID=facebook/sam3
SAM3_LOCAL_FILES_ONLY=false
```

Use `SAM3_LOCAL_FILES_ONLY=true` only after the model exists in the local
Hugging Face cache or when `SAM3_MODEL_ID` points to a local snapshot directory.

Important runtime defaults:

```text
DOP_SOURCE=wcs
WCS_COVERAGE_ID=OI.OrthoimageCoverage
MODALITY_USE_NDVI=true
MODALITY_USE_NDSM=false
GLOBAL_POLYGON_NMS_IOU=0.5
GLOBAL_POLYGON_CONTAINMENT_RATIO=0.85
```

## Start The System

Run server and worker in two separate terminals from the repository root.

Terminal 1, review UI and API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn ki_geodaten.app.main:app --host 127.0.0.1 --port 8000 --reload --reload-dir ki_geodaten --reload-exclude "data/*" --reload-exclude "*.db*"
```

Terminal 2, GPU worker:

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
.\.venv\Scripts\python.exe -m ki_geodaten.worker.loop
```

Open:

```text
http://127.0.0.1:8000
```

On Linux, Git Bash, or WSL, the wrapper scripts are equivalent:

```bash
./scripts/run-server.sh
./scripts/run-worker.sh
```

## Run A Pipeline Job

Use the UI at `http://127.0.0.1:8000`, draw/select an AOI in Bavaria, enter a
prompt such as `building`, choose the tile preset, and submit the job. The API
stores job state in `data/jobs.db`, downloads imagery under `data/dop/`, and
writes exports under `data/results/`.

The worker processes one job at a time. Pending jobs stay in `PENDING` until the
worker claims them.

## Tests

Run the reproducible test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q -p no:cacheprovider
```

Run the gated SAM smoke test only after Hugging Face authentication and model
access are configured:

```powershell
$env:RUN_SAM3_SMOKE="1"
$env:SAM3_MODEL_ID="facebook/sam3"
.\.venv\Scripts\python.exe -m pytest tests\pipeline\test_sam3_adapter_smoke.py -q -p no:cacheprovider
```

## Reproducibility Notes

- Runtime configuration is read from `.env` through `ki_geodaten.config.Settings`.
- WCS is the model input path; WMS is only the Leaflet review basemap.
- Job metadata stores the effective run configuration for later inspection.
- Global polygon NMS is deterministic: candidates are sorted by score, area,
  tile position, and original position before suppression.
- `data/` is intentionally ignored by Git because it contains local rasters,
  SQLite state, model outputs, and experiment artifacts.

## Known Limits

- AOIs are restricted to Bavaria.
- Default maximum AOI size is 1 km2.
- The production pipeline expects one active GPU worker.
- nDSM requires DOM/DGM OpenData downloads and increases runtime.
