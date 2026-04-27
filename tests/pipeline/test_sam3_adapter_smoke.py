import os

import pytest

pytest.importorskip("transformers")

if os.environ.get("RUN_SAM3_SMOKE") != "1":
    pytest.skip("RUN_SAM3_SMOKE=1 not set", allow_module_level=True)


def test_sam3_token_counter_does_not_require_cuda():
    from ki_geodaten.pipeline.segmenter import Sam3TextTokenCounter

    model_ref = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")

    counter = Sam3TextTokenCounter(model_ref)
    assert counter("building") > 0


def test_sam3_segmenter_smoke_prediction_shape():
    import numpy as np
    from affine import Affine

    from ki_geodaten.models import TilePreset
    from ki_geodaten.pipeline.segmenter import MaskResult, Sam3Segmenter
    from ki_geodaten.pipeline.tiler import Tile, TileConfig

    model_ref = os.environ.get("SAM3_MODEL_ID", "facebook/sam3")

    cfg = TileConfig.from_preset(TilePreset.MEDIUM)
    tile = Tile(
        array=np.zeros((1024, 1024, 3), dtype=np.uint8),
        pixel_origin=(0, 0),
        size=1024,
        center_margin=cfg.center_margin,
        affine=Affine(0.2, 0, 0, 0, -0.2, 204.8),
        tile_row=0,
        tile_col=0,
        nodata_mask=np.zeros((1024, 1024), dtype=bool),
    )
    segmenter = Sam3Segmenter(model_ref)
    out = segmenter.predict(tile, "building")
    assert isinstance(out, list)
    for item in out:
        assert isinstance(item, MaskResult)
        assert item.mask.shape == (1024, 1024)
        assert item.mask.dtype == bool
