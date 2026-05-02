from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class MaskResult:
    mask: np.ndarray
    score: float
    box_pixel: tuple[int, int, int, int]
    ndvi_mean: float | None = None
    ndsm_mean: float | None = None


class SegmenterUnavailableError(RuntimeError):
    """Raised when the SAM adapter cannot run in the current environment."""


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    if inter == 0:
        return 0.0
    union = int(np.logical_or(a, b).sum())
    return inter / union


def _containment(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    min_area = min(int(a.sum()), int(b.sum()))
    if min_area == 0:
        return 0.0
    return inter / min_area


def _boxes_intersect(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return min(ar1, br1) > max(ar0, br0) and min(ac1, bc1) > max(ac0, bc0)


def local_mask_nms(
    masks: Iterable[MaskResult],
    *,
    iou_threshold: float,
    containment_ratio: float,
) -> list[MaskResult]:
    sorted_masks = sorted(masks, key=lambda item: item.score, reverse=True)
    kept: list[MaskResult] = []
    for candidate in sorted_masks:
        drop = False
        for existing in kept:
            if not _boxes_intersect(candidate.box_pixel, existing.box_pixel):
                continue
            if _iou(candidate.mask, existing.mask) >= iou_threshold:
                drop = True
                break
            if _containment(candidate.mask, existing.mask) >= containment_ratio:
                drop = True
                break
        if not drop:
            kept.append(candidate)
    return kept


def _preprocess_rgb_for_sam(
    array: np.ndarray,
    *,
    mode: str = "none",
    gamma: float = 1.0,
    brightness: float = 1.0,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 8,
    clahe_blend: float = 1.0,
) -> np.ndarray:
    mode = mode.lower().strip()
    if mode not in {"none", "gamma", "clahe"}:
        raise ValueError("mode must be one of: none, gamma, clahe")
    if gamma <= 0:
        raise ValueError("gamma must be > 0")
    if brightness <= 0:
        raise ValueError("brightness must be > 0")
    if clahe_clip_limit <= 0:
        raise ValueError("clahe_clip_limit must be > 0")
    if clahe_tile_grid_size <= 0:
        raise ValueError("clahe_tile_grid_size must be > 0")
    if not 0.0 <= clahe_blend <= 1.0:
        raise ValueError("clahe_blend must be between 0 and 1")

    rgb = array.astype(np.uint8, copy=False)
    if mode == "none":
        return rgb

    if mode == "clahe":
        enhanced = _clahe_luminance(
            rgb,
            clip_limit=clahe_clip_limit,
            tile_grid_size=clahe_tile_grid_size,
        )
        if clahe_blend == 0.0:
            rgb = rgb.copy()
        elif clahe_blend == 1.0:
            rgb = enhanced
        else:
            mixed = (
                rgb.astype(np.float32) * (1.0 - clahe_blend)
                + enhanced.astype(np.float32) * clahe_blend
            )
            rgb = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)

    if gamma == 1.0 and brightness == 1.0:
        return rgb

    work = rgb.astype(np.float32) / 255.0
    if gamma != 1.0:
        work = np.power(work, gamma)
    if brightness != 1.0:
        work *= brightness
    return np.clip(np.rint(work * 255.0), 0, 255).astype(np.uint8)


def _clahe_luminance(
    rgb: np.ndarray,
    *,
    clip_limit: float,
    tile_grid_size: int,
) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:
        raise SegmenterUnavailableError(
            "SAM image preprocessing mode 'clahe' requires opencv-python-headless."
        ) from exc

    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    enhanced_l = clahe.apply(l_channel)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


class Sam3Segmenter:
    """SAM 3 adapter using the official Hugging Face Transformers API."""

    def __init__(
        self,
        model_ref: str | Path,
        *,
        device: str = "cuda",
        iou_threshold: float = 0.6,
        containment_ratio: float = 0.9,
        score_threshold: float = 0.3,
        mask_threshold: float = 0.5,
        image_preprocess: str = "none",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        clahe_blend: float = 1.0,
        image_gamma: float = 1.0,
        image_brightness: float = 1.0,
        local_files_only: bool = True,
        model_cls: Any | None = None,
        processor_cls: Any | None = None,
    ) -> None:
        self.model_ref = model_ref
        self.device = device
        self.iou_threshold = iou_threshold
        self.containment_ratio = containment_ratio
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.image_preprocess = image_preprocess
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = clahe_tile_grid_size
        self.clahe_blend = clahe_blend
        self.image_gamma = image_gamma
        self.image_brightness = image_brightness
        self.local_files_only = local_files_only
        self._model = None
        self._processor = None
        self._model_cls = model_cls
        self._processor_cls = processor_cls

    def load(self) -> None:
        if self._model is not None:
            return
        if isinstance(self.model_ref, Path) and not self.model_ref.exists():
            raise SegmenterUnavailableError(f"SAM3 checkpoint not found: {self.model_ref}")
        try:
            from transformers import Sam3Model, Sam3Processor
        except Exception as exc:  # noqa: BLE001
            raise SegmenterUnavailableError(
                "Transformers SAM3 API is unavailable; install transformers with SAM3 support."
            ) from exc

        model_cls = self._model_cls or Sam3Model
        processor_cls = self._processor_cls or Sam3Processor
        from_pretrained_kwargs = {"local_files_only": self.local_files_only}
        try:
            self._processor = processor_cls.from_pretrained(
                str(self.model_ref),
                **from_pretrained_kwargs,
            )
            self._model = model_cls.from_pretrained(
                str(self.model_ref),
                **from_pretrained_kwargs,
            )
            self._model.to(self.device)
            self._model.eval()
        except Exception as exc:  # noqa: BLE001
            raise SegmenterUnavailableError(
                "SAM3 model could not be loaded. Verify Hugging Face auth/access "
                f"and model reference: {self.model_ref}"
            ) from exc

    def predict(self, tile, prompt: str) -> list[MaskResult]:
        if self._model is None:
            self.load()
        import torch

        converted: list[MaskResult] = []
        for mode in self._preprocess_modes():
            converted.extend(self._predict_single_pass(tile, prompt, mode, torch))

        return local_mask_nms(
            converted,
            iou_threshold=self.iou_threshold,
            containment_ratio=self.containment_ratio,
        )

    def _preprocess_modes(self) -> tuple[str, ...]:
        mode = self.image_preprocess.lower().strip()
        if mode == "ensemble":
            return ("none", "clahe")
        if mode in {"none", "gamma", "clahe"}:
            return (mode,)
        raise ValueError("image_preprocess must be one of: none, gamma, clahe, ensemble")

    def _predict_single_pass(self, tile, prompt: str, mode: str, torch) -> list[MaskResult]:
        image_array = _preprocess_rgb_for_sam(
            tile.array,
            mode=mode,
            gamma=self.image_gamma,
            brightness=self.image_brightness,
            clahe_clip_limit=self.clahe_clip_limit,
            clahe_tile_grid_size=self.clahe_tile_grid_size,
            clahe_blend=self.clahe_blend,
        )
        image = Image.fromarray(image_array, mode="RGB")
        inputs = self._processor(images=image, text=prompt, return_tensors="pt")
        inputs = {
            key: value.to(self.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        try:
            with torch.inference_mode():
                outputs = self._model(**inputs)
            results = self._processor.post_process_instance_segmentation(
                outputs,
                threshold=self.score_threshold,
                mask_threshold=self.mask_threshold,
                target_sizes=[(tile.array.shape[0], tile.array.shape[1])],
            )[0]
            converted = _convert_transformers_results(
                results.get("masks", []),
                results.get("boxes", []),
                results.get("scores", []),
                shape=(tile.array.shape[0], tile.array.shape[1]),
            )
            return converted
        finally:
            del inputs

    def encoder_token_count(self, prompt: str) -> int:
        if self._processor is None:
            self.load()
        return _processor_token_count(self._processor, prompt)


class Sam3TextTokenCounter:
    """Tokenizer-only adapter for the FastAPI process."""

    def __init__(
        self,
        model_ref: str | Path,
        *,
        local_files_only: bool = True,
        processor_cls: Any | None = None,
    ) -> None:
        self.model_ref = model_ref
        self.local_files_only = local_files_only
        self._processor = None
        self._processor_cls = processor_cls

    def load(self) -> None:
        if self._processor is not None:
            return
        try:
            from transformers import Sam3Processor
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Transformers SAM3 processor API is unavailable") from exc
        processor_cls = self._processor_cls or Sam3Processor
        self._processor = processor_cls.from_pretrained(
            str(self.model_ref),
            local_files_only=self.local_files_only,
        )

    def __call__(self, prompt: str) -> int:
        if self._processor is None:
            self.load()
        return _processor_token_count(self._processor, prompt)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def _box_xyxy_to_rc(box: np.ndarray, shape: tuple[int, int]) -> tuple[int, int, int, int]:
    height, width = shape
    x0, y0, x1, y1 = [float(v) for v in box[:4]]
    c0 = max(0, min(width, int(np.floor(x0))))
    r0 = max(0, min(height, int(np.floor(y0))))
    c1 = max(0, min(width, int(np.ceil(x1))))
    r1 = max(0, min(height, int(np.ceil(y1))))
    return (r0, c0, r1, c1)


def _mask_bbox(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return np.asarray([0, 0, 0, 0], dtype=float)
    return np.asarray(
        [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1],
        dtype=float,
    )


def _convert_transformers_results(
    masks,
    boxes,
    scores,
    *,
    shape: tuple[int, int],
) -> list[MaskResult]:
    masks_np = _to_numpy(masks)
    boxes_np = _to_numpy(boxes)
    scores_np = _to_numpy(scores)
    if masks_np.size == 0:
        return []
    if masks_np.ndim == 2:
        masks_np = masks_np[None, :, :]
    out: list[MaskResult] = []
    for idx, mask in enumerate(masks_np):
        bool_mask = mask.astype(bool, copy=True)
        box = boxes_np[idx] if idx < len(boxes_np) else _mask_bbox(bool_mask)
        score = float(scores_np[idx]) if idx < len(scores_np) else 0.0
        out.append(
            MaskResult(
                mask=bool_mask,
                score=score,
                box_pixel=_box_xyxy_to_rc(box, shape),
            )
        )
    return out


def _processor_token_count(processor, prompt: str) -> int:
    # Spec §8 requires checking the *final* templatized encoder sequence
    # length (incl. special tokens, exclusive of padding). Prefer the
    # attention_mask sum when the tokenizer pads anyway; fall back to
    # the raw input_ids shape when it doesn't.
    encoded = processor.tokenizer(
        prompt,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=True,
    )
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        return int(attention_mask.sum().item())
    return int(encoded["input_ids"].shape[-1])
