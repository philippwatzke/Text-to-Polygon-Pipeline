# Heuristic Baseline Protocol

Goal: test the null hypothesis that a simple raster rule is sufficient and SAM
does not add material value.

Compare the same AOIs with three systems:

1. `SAM only`: normal prompt run, no modality thresholds.
2. `Heuristic only`: raster thresholding without SAM.
   - Buildings: `nDSM >= 3m`, connected components, morphology, minimum area.
   - Vegetation: `NDVI >= 0.3`, optionally `nDSM >= 2m`.
3. `SAM + modalities`: normal SAM run plus nDSM/NDVI post-filter.

Do not hand-pick one weak heuristic. Tune a small grid on validation AOIs, then
freeze the best heuristic for the test AOIs.

Recommended first grid for buildings:

| Parameter | Values |
| --- | --- |
| nDSM min | `2.0`, `2.5`, `3.0`, `3.5`, `4.0` |
| min area m2 | `5`, `10`, `20` |
| closing m | `0`, `1`, `2` |

Example baseline job:

```powershell
.\.venv\Scripts\python.exe scripts\run_heuristic_baseline.py `
  --prompt building `
  --label heuristic_building_ndsm3 `
  --bbox-wgs84 11.55 48.13 11.56 48.14 `
  --ndsm-min 3.0 `
  --min-area-m2 10 `
  --close-m 1
```

The script creates a normal review job in `data/jobs.db`, so the output can be
opened in the app, manually rejected/marked, compared, and exported through the
same workflow as SAM jobs.

For significance, aggregate per AOI. Report precision, recall, and F1 for each
system, then use a paired bootstrap or Wilcoxon signed-rank test over AOIs.
Report the effect size in F1 points in addition to any p-value.
