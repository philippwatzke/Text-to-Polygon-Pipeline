# LDBV DOP20 WCS Verification

**Verified on:** 2026-04-28
**Performed by:** Philipp Watzke
**Endpoint:** `https://geoservices.bayern.de/pro/wcs/dop/v1/wcs_inspire_dop20`

This is the empirical verification that the LDBV DOP20 WCS endpoint behaves
as the pipeline assumes. It replaces `wms-verification.md` (now scoped to the
UI basemap only — see `wms-display-verification.md`).

---

## 1. Service identity

| Field | Value |
|---|---|
| `WCS_URL` | `https://geoservices.bayern.de/pro/wcs/dop/v1/wcs_inspire_dop20` |
| `WCS_VERSION` | `2.0.1` |
| `WCS_COVERAGE_ID` | **`OI.OrthoimageCoverage`** (INSPIRE naming, **not** `by_dop20c`) |
| `WCS_FORMAT` | `image/tiff` |
| `WCS_CRS` | `EPSG:25832` |
| `WCS_MAX_PIXELS` | `6000` per side (default; server appears to allow more) |
| Auth | HTTP Basic, `WCS_USERNAME` / `WCS_PASSWORD` in `.env` |
| Native pixel size | 0.2 m |
| **Grid origin (EPSG:25832)** | **(492346.1, 5618159.9)** → **0.1 mod 0.2** on both axes |
| Coverage extent (EPSG:25832) | minx=492346, miny=5214282, maxx=867438, maxy=5618160 |
| Bands | 4: `r`, `g`, `b`, **`ir`** (NIR) — all uint8, 0..255 |

## 2. GetCapabilities

```bash
curl -u "$WCS_USERNAME:$WCS_PASSWORD" \
  "https://geoservices.bayern.de/pro/wcs/dop/v1/wcs_inspire_dop20?SERVICE=WCS&REQUEST=GetCapabilities"
```

Response: HTTP 200, 8750 bytes, `text/xml`.

- Service title: **INSPIRE-WCS BY Orthofotografie DOP20**
- `ows:ServiceTypeVersion`: `2.0.1`
- Profiles include:
  - `http://www.opengis.net/spec/WCS/2.0/conf/core`
  - `http://www.opengis.net/spec/WCS_protocol-binding_get-kvp/1.0/conf/get-kvp`
  - `http://www.opengis.net/spec/GMLCOV_geotiff-coverages/1.0/conf/geotiff-coverage`
  - `http://www.opengis.net/spec/WCS_service-extension_crs/1.0/conf/crs`
- Operations: `GetCapabilities`, `DescribeCoverage`, `GetCoverage`
- Supported `wcs:formatSupported`: `image/png`, **`image/tiff`** (used), `image/jpeg`, `image/png; mode=8bit`, `image/vnd.jpeg-png`, `image/vnd.jpeg-png8`
- One coverage offered: `OI.OrthoimageCoverage` (Subtype `RectifiedGridCoverage`)

## 3. DescribeCoverage

```bash
curl -u "$WCS_USERNAME:$WCS_PASSWORD" \
  "https://geoservices.bayern.de/pro/wcs/dop/v1/wcs_inspire_dop20?SERVICE=WCS&VERSION=2.0.1&REQUEST=DescribeCoverage&COVERAGEID=OI.OrthoimageCoverage"
```

Key fields from the response:

```xml
<gml:Envelope srsName=".../EPSG/0/25832" axisLabels="x y" uomLabels="m m">
  <gml:lowerCorner>492346 5214282</gml:lowerCorner>
  <gml:upperCorner>867438 5618160</gml:upperCorner>
</gml:Envelope>

<gml:RectifiedGrid dimension="2">
  <gml:limits><gml:GridEnvelope>
    <gml:low>0 0</gml:low>
    <gml:high>1875459 2019389</gml:high>
  </gml:GridEnvelope></gml:limits>
  <gml:origin>
    <gml:Point srsName=".../EPSG/0/25832">
      <gml:pos>492346.100000 5618159.900000</gml:pos>
    </gml:Point>
  </gml:origin>
  <gml:offsetVector>0.200000 0</gml:offsetVector>
  <gml:offsetVector>0 -0.200000</gml:offsetVector>
</gml:RectifiedGrid>

<swe:DataRecord>
  <swe:field name="r">…<swe:interval>0 255</swe:interval>…</swe:field>
  <swe:field name="g">…<swe:interval>0 255</swe:interval>…</swe:field>
  <swe:field name="b">…<swe:interval>0 255</swe:interval>…</swe:field>
  <swe:field name="ir">…<swe:interval>0 255</swe:interval>…</swe:field>
</swe:DataRecord>
```

- **Origin (492346.1, 5618159.9):** both ≡ 0.1 mod 0.2. Pixel corners therefore land on `x ∈ {…, 492346.1, 492346.3, 492346.5, …}` — the integer-metre grid is **off** by 0.1 m. The pipeline's `WCS_GRID_ORIGIN_X = WCS_GRID_ORIGIN_Y = 0.1` reflects this.
- **Offset vectors (0.2, 0)** and **(0, −0.2)**: confirms 0.2 m grid, y axis pointing down.
- **Range type:** four uint8 fields named `r`, `g`, `b`, `ir`. **Band 4 is NIR**, not Alpha. This is significant for later NDVI work — no separate CIR coverage needed.
- `swe:nilValues` is present but **empty** for every band: the server declares no NoData sentinel.

## 4. Adjacent-chunk source-pixel consistency test

The previously documented "compare A's last column with B's first column" is
**not the right test** — those are physically adjacent but distinct pixels (at
x=…0.0 and …0.2 respectively), so a non-zero diff is just real-world image
content variation, not server resampling.

The actual seamless-mosaic property is: **the server addresses source pixels
deterministically, regardless of the request window**. We verify this with
two checks.

### 4a. Determinism

Request the same chunk twice; bytes must be identical.

```bash
curl -u … "?…&SUBSET=x(690000.1,690100.1)&SUBSET=y(5334000.1,5334100.1)" -o A1.tif
curl -u … "?…&SUBSET=x(690000.1,690100.1)&SUBSET=y(5334000.1,5334100.1)" -o A2.tif
```

```python
import rasterio, numpy as np
with rasterio.open("A1.tif") as a1, rasterio.open("A2.tif") as a2:
    diff = np.abs(a1.read().astype(int) - a2.read().astype(int))
print(diff.max())  # → 0
```

**Result:** `diff.max() == 0` → byte-identical. ✓

### 4b. Source-pixel consistency across windows

Request chunk **A** for `x ∈ [690000.1, 690100.1]` and chunk **C** for
`x ∈ [690050.1, 690150.1]`. They overlap at `x ∈ [690050.1, 690100.1]` (250 px).
Compare `A[:, :, 250:500]` with `C[:, :, 0:250]` — these cover **the same
physical pixels**.

```python
sub_a = arr_a[:, :, 250:500]
sub_c = arr_c[:, :, 0:250]
diff = np.abs(sub_a.astype(int) - sub_c.astype(int))
print(diff.max(), diff.mean())  # → 0, 0.0000
```

**Result:** `diff.max() == 0`, `diff.mean() == 0.0000` for all four bands. ✓

The server addresses source pixels by an absolute grid index that is independent
of the request window. The pipeline's grid-snap + chunked download therefore
produces seamless mosaics by construction — no overlap stitching needed.

## 5. Datatype, band order, and Alpha vs IR mismatch

Per DescribeCoverage, band 4 is **`ir`** (Near-Infrared). However, when rasterio
opens the returned GeoTIFF, it reports:

```python
src.colorinterp = [Red, Green, Blue, Alpha]
src.dtypes      = ['uint8', 'uint8', 'uint8', 'uint8']
src.nodatavals  = (None, None, None, None)
```

**Implication:** the server tags band 4 as Alpha in the GeoTIFF
ColorInterpretation block, even though the data is NIR. This means:

- `rasterio.DatasetReader.dataset_mask()` will treat band 4 as a transparency
  mask. Where IR=0 (rare in real DOP), pixels would be reported as NoData;
  where IR>0 (the overwhelming majority of pixels), all are "valid".
- For our pipeline this is acceptable: the tiler reads bands `[1, 2, 3]` for
  SAM 3, and `dataset_mask` is used only for NoData detection, which the
  current LDBV DOP20 effectively does not produce inside Bayern (the coverage
  is gap-free within the federal-state extent). At AOIs that cross the Bayern
  border the server would presumably return either a 4xx or zero-fill bands —
  to be tested at the time we run an edge-of-state AOI.
- For future NDVI work, **read band 4 explicitly** as a data band rather than
  relying on `dataset_mask`. The IR signal is genuinely there and usable.

## 6. Authentication

- HTTP Basic, credentials supplied per request (`-u "$USER:$PASS"`).
- HTTP 200 with valid credentials.
- Without credentials: HTTP 401 `Unauthorized` (not retried by `urllib3.Retry`).
- No observed rate limit during verification (10+ rapid requests succeeded).
- Credentials user-bound (`WCS_USERNAME=26A177839`); no expiration documented.

## 7. End-to-end pipeline probe

After the migration, we ran the actual pipeline client against the live endpoint
at the verified Munich AOI:

```python
prepared = prepare_download_bbox(
    minx=690000.1, miny=5334000.1, maxx=690100.1, maxy=5334100.1,
    preset=TilePreset.MEDIUM, origin_x=0.1, origin_y=0.1,
)
vrt = download_dop20(
    bbox_utm=prepared.download_bbox, out_dir=Path("..."),
    wcs_url=settings.WCS_URL,
    coverage_id=settings.WCS_COVERAGE_ID,
    wcs_version=settings.WCS_VERSION,
    fmt=settings.WCS_FORMAT, crs=settings.WCS_CRS,
    max_pixels=settings.WCS_MAX_PIXELS,
    origin_x=settings.WCS_GRID_ORIGIN_X, origin_y=settings.WCS_GRID_ORIGIN_Y,
    username=settings.WCS_USERNAME, password=settings.WCS_PASSWORD,
)
# -> AOI       (690000.1, 5334000.1, 690100.1, 5334100.1)
# -> Download  (689936.1, 5333936.1, 690164.1, 5334164.1)  (medium-preset margin)
# -> VRT       1140 x 1140, 4 bands, CRS=EPSG:25832
# -> bounds    matches download bbox exactly
# -> sample    center pixel mean per band [69, 84, 60, 173] (vegetation, IR high)
```

✓ The complete `prepare → download → write GeoTIFF chunks → build VRT → reopen
with rasterio` chain works against the live LDBV WCS.

## 8. Outcome — verification gate

- [x] Auth confirmed (HTTP 200 with credentials, 401 without).
- [x] CoverageID identified (`OI.OrthoimageCoverage`) and propagated to Settings.
- [x] Grid origin identified ((0.1, 0.1) mod 0.2) and propagated to Settings.
- [x] Native datatype (uint8 ×4 incl. NIR band) confirmed.
- [x] Source-pixel consistency: byte-identical across overlapping windows.
- [x] Determinism: byte-identical for repeated identical requests.
- [x] End-to-end `download_dop20()` produces a valid VRT with correct bounds.
- [x] All 119 unit/integration tests pass against the new client.

**The WCS pivot is empirically viable. Pipeline assumptions hold.**

## 9. Notes / follow-ups

- The server-tagged `Alpha` band 4 vs DescribeCoverage's `ir` field: not a
  correctness issue today, but worth a note when implementing NDVI later. Read
  band 4 explicitly with `src.read(4, ...)` rather than relying on
  `dataset_mask` semantics.
- NoData behaviour at the Bayern border is untested. Open question for a later
  edge-of-state AOI run.
- The chunk GeoTIFFs ship a non-standard `LOCAL_CS["ETRS89 / UTM zone 32N", ...]`
  CRS block. `rasterio.CRS.to_epsg()` returns `None` for this. `_verify_chunk`
  therefore checks bounds + linear units instead of strict EPSG equality; the
  canonical `EPSG:25832` is set explicitly in the VRT we build on top.
- `WCS_MAX_PIXELS = 6000` was inherited from the WMS path. The server may
  allow larger windows; if Munich-area test runs are slow, raising this value
  reduces the number of HTTP round-trips.
