import numpy as np
import pytest

from ki_geodaten.pipeline.segmenter import (
    MaskResult,
    Sam3Segmenter,
    Sam3TextTokenCounter,
    SegmenterUnavailableError,
    _preprocess_rgb_for_sam,
    local_mask_nms,
)


def _mask(bbox, shape=(100, 100)):
    m = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = bbox
    m[r0:r1, c0:c1] = True
    return m


def test_local_nms_drops_high_iou_duplicate():
    a = MaskResult(mask=_mask((10, 10, 50, 50)), score=0.9, box_pixel=(10, 10, 50, 50))
    b = MaskResult(mask=_mask((11, 11, 49, 49)), score=0.8, box_pixel=(11, 11, 49, 49))
    kept = local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [a]


def test_local_nms_drops_contained_mask():
    a = MaskResult(mask=_mask((0, 0, 80, 80)), score=0.9, box_pixel=(0, 0, 80, 80))
    b = MaskResult(mask=_mask((10, 10, 30, 30)), score=0.8, box_pixel=(10, 10, 30, 30))
    kept = local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [a]


def test_local_nms_keeps_spatially_separate():
    a = MaskResult(mask=_mask((0, 0, 20, 20)), score=0.9, box_pixel=(0, 0, 20, 20))
    b = MaskResult(mask=_mask((70, 70, 90, 90)), score=0.8, box_pixel=(70, 70, 90, 90))
    kept = local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [a, b]


def test_local_nms_score_descending_priority():
    high = MaskResult(mask=_mask((0, 0, 50, 50)), score=0.9, box_pixel=(0, 0, 50, 50))
    low = MaskResult(mask=_mask((0, 0, 50, 50)), score=0.3, box_pixel=(0, 0, 50, 50))
    kept = local_mask_nms([low, high], iou_threshold=0.6, containment_ratio=0.9)
    assert kept == [high]


def test_local_nms_bbox_prefilter_skips_disjoint_dense_masks(monkeypatch):
    a = MaskResult(mask=_mask((0, 0, 20, 20)), score=0.9, box_pixel=(0, 0, 20, 20))
    b = MaskResult(mask=_mask((70, 70, 90, 90)), score=0.8, box_pixel=(70, 70, 90, 90))

    def fail_dense_compare(*args, **kwargs):
        raise AssertionError("dense mask comparison should not run for disjoint boxes")

    import ki_geodaten.pipeline.segmenter as segmenter

    monkeypatch.setattr(segmenter, "_iou", fail_dense_compare)
    monkeypatch.setattr(segmenter, "_containment", fail_dense_compare)
    assert local_mask_nms([a, b], iou_threshold=0.6, containment_ratio=0.9) == [a, b]


def test_sam3_segmenter_missing_checkpoint_fails_fast(tmp_path):
    segmenter = Sam3Segmenter(tmp_path / "missing.pt")

    with pytest.raises(SegmenterUnavailableError, match="checkpoint not found"):
        segmenter.load()


class _FakeModel:
    @classmethod
    def from_pretrained(cls, model_ref, **kwargs):
        instance = cls()
        instance.model_ref = model_ref
        instance.from_pretrained_kwargs = kwargs
        return instance

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True
        return self

    def __call__(self, **inputs):
        self.calls = getattr(self, "calls", 0) + 1
        return {"raw": inputs}


class _FakeTokenizer:
    def __call__(self, prompt, **kwargs):
        import torch

        return {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
        }


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    @classmethod
    def from_pretrained(cls, model_ref, **kwargs):
        instance = cls()
        instance.model_ref = model_ref
        instance.from_pretrained_kwargs = kwargs
        return instance

    def __call__(self, images, text, return_tensors):
        import torch

        self.calls = getattr(self, "calls", 0) + 1
        assert images.size == (100, 100)
        assert text == "building"
        assert return_tensors == "pt"
        return {"pixel_values": torch.zeros((1, 3, 100, 100))}

    def post_process_instance_segmentation(
        self,
        outputs,
        threshold,
        mask_threshold,
        target_sizes,
    ):
        import torch

        mask = torch.zeros((1, 100, 100), dtype=torch.bool)
        mask[:, 10:30, 20:40] = True
        return [
            {
                "masks": mask,
                "boxes": torch.tensor([[20.2, 10.4, 39.8, 29.6]]),
                "scores": torch.tensor([0.75]),
            }
        ]


def test_sam3_segmenter_converts_transformers_results():
    segmenter = Sam3Segmenter(
        "facebook/sam3",
        device="cpu",
        model_cls=_FakeModel,
        processor_cls=_FakeProcessor,
    )
    tile = type(
        "Tile",
        (),
        {"array": np.zeros((100, 100, 3), dtype=np.uint8)},
    )()

    out = segmenter.predict(tile, "building")

    assert len(out) == 1
    assert out[0].score == pytest.approx(0.75)
    assert out[0].mask.dtype == bool
    assert out[0].mask.shape == (100, 100)
    assert out[0].box_pixel == (10, 20, 30, 40)
    assert segmenter._processor.from_pretrained_kwargs == {"local_files_only": True}
    assert segmenter._model.from_pretrained_kwargs == {"local_files_only": True}


def test_sam3_segmenter_ensemble_runs_original_and_clahe_then_deduplicates():
    segmenter = Sam3Segmenter(
        "facebook/sam3",
        device="cpu",
        image_preprocess="ensemble",
        model_cls=_FakeModel,
        processor_cls=_FakeProcessor,
    )
    tile = type(
        "Tile",
        (),
        {"array": np.zeros((100, 100, 3), dtype=np.uint8)},
    )()

    out = segmenter.predict(tile, "building")

    assert len(out) == 1
    assert segmenter._processor.calls == 2
    assert segmenter._model.calls == 2


def test_preprocess_rgb_for_sam_gamma_lifts_shadows_more_than_highlights():
    arr = np.asarray([[[40, 80, 200]]], dtype=np.uint8)

    out = _preprocess_rgb_for_sam(arr, mode="gamma", gamma=0.9)

    assert out[0, 0, 0] > arr[0, 0, 0]
    assert out[0, 0, 1] > arr[0, 0, 1]
    assert out[0, 0, 2] > arr[0, 0, 2]
    assert int(out[0, 0, 0]) - int(arr[0, 0, 0]) > int(out[0, 0, 2]) - int(arr[0, 0, 2])


def test_preprocess_rgb_for_sam_identity_returns_uint8_view():
    arr = np.asarray([[[40, 80, 200]]], dtype=np.uint8)

    out = _preprocess_rgb_for_sam(arr, mode="none", gamma=0.9, brightness=1.2)

    assert out.dtype == np.uint8
    assert np.array_equal(out, arr)


def test_preprocess_rgb_for_sam_clahe_changes_luminance_without_shape_or_dtype_change():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :32] = [35, 35, 35]
    arr[:, 32:] = [90, 90, 90]

    out = _preprocess_rgb_for_sam(
        arr,
        mode="clahe",
        clahe_clip_limit=2.0,
        clahe_tile_grid_size=4,
    )

    assert out.shape == arr.shape
    assert out.dtype == np.uint8
    assert not np.array_equal(out, arr)
    assert np.allclose(out[..., 0], out[..., 1], atol=1)
    assert np.allclose(out[..., 1], out[..., 2], atol=1)


def test_preprocess_rgb_for_sam_clahe_blend_preserves_some_original_signal():
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:, :32] = [35, 35, 35]
    arr[:, 32:] = [90, 90, 90]

    original = _preprocess_rgb_for_sam(arr, mode="clahe", clahe_blend=0.0)
    full = _preprocess_rgb_for_sam(arr, mode="clahe", clahe_blend=1.0)
    blended = _preprocess_rgb_for_sam(arr, mode="clahe", clahe_blend=0.5)

    assert np.array_equal(original, arr)
    assert not np.array_equal(full, arr)
    assert np.abs(blended.astype(int) - arr.astype(int)).mean() < np.abs(
        full.astype(int) - arr.astype(int)
    ).mean()


def test_sam3_text_counter_uses_processor_tokenizer():
    counter = Sam3TextTokenCounter("facebook/sam3", processor_cls=_FakeProcessor)

    assert counter("building") == 3
