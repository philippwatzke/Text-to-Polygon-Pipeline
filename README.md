# Text-to-Polygon Pipeline (Bayern DOP20 + SAM 3.1)

Local prototype that extracts georeferenced polygons from Bayerische DOP20
orthophotos using a free-text prompt and SAM 3.1 zero-shot segmentation.

See `docs/superpowers/specs/2026-04-22-text-to-polygon-design.md` for the
full design rationale.

## Requirements

- Python 3.12
- CUDA-capable NVIDIA GPU + CUDA PyTorch
- `rasterio` for GeoTIFF/VRT IO
- `fiona` for explicit-schema GeoPackage export
- Hugging Face access to `facebook/sam3`
- SQLite 3.35+

## Install

```bash
conda create -n ki-geodaten python=3.12 fiona -c conda-forge
conda activate ki-geodaten
pip install -e .[dev]
```

On Windows this project uses the official Transformers SAM3 API because the
direct `facebookresearch/sam3` package imports Triton kernels that are not
published for Windows. Install CUDA PyTorch from the official index, then log in
to Hugging Face:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install transformers accelerate
.\.venv\Scripts\hf.exe auth login
.\.venv\Scripts\hf.exe auth whoami
```

## Configure

Copy `.env.example` to `.env`. Defaults point at the LDBV OpenData WMS:

```text
WMS_URL=https://geoservices.bayern.de/od/wms/dop/v1/dop20
WMS_LAYER=by_dop20c
WMS_VERSION=1.1.1
WMS_FORMAT=image/png
WMS_CRS=EPSG:25832
WMS_MAX_PIXELS=6000
SAM3_MODEL_ID=facebook/sam3
```

The WMS verification evidence is in `docs/wms-verification.md`.

## Run

Two processes in separate terminals:

```bash
./scripts/run-server.sh
./scripts/run-worker.sh
```

On Windows, use Git Bash/WSL or run the commands directly:

```powershell
python -m uvicorn ki_geodaten.app.main:app --reload `
  --reload-dir ki_geodaten --reload-exclude "data/*" --reload-exclude "*.db*"
python -m ki_geodaten.worker.loop
```

## Tests

```bash
pytest
```

Tests do not require SAM or GPU. The standard suite no longer requires
`osgeo.gdal`; VRT mosaics are written directly and validated through Rasterio.

After Hugging Face login, run the gated SAM smoke test explicitly:

```powershell
$env:RUN_SAM3_SMOKE="1"
$env:SAM3_MODEL_ID="facebook/sam3"
.\.venv\Scripts\python.exe -m pytest tests\pipeline\test_sam3_adapter_smoke.py -q -p no:cacheprovider
```

## Known Limits

- Bayern only, via LDBV DOP20 WMS.
- 1 km2 max AOI.
- `tile_preset` selects max object diameter: 64 m, 128 m, or 192 m.
- `large` is expensive and intended for offline exploration.
