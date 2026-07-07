"""Segmentation mask helpers."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor


def _array(value: Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.numpy()
    return np.asarray(value)


def mask_iou(
    prediction: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> np.ndarray:
    """Per-class IoU for integer segmentation masks."""

    pred = _array(prediction).astype(np.int64)
    true = _array(target).astype(np.int64)
    if pred.shape != true.shape:
        raise ValueError("prediction and target masks must have matching shape")
    if num_classes is None:
        valid_pred = pred if ignore_index is None else pred[pred != ignore_index]
        valid_true = true if ignore_index is None else true[true != ignore_index]
        max_label = 0
        if valid_pred.size:
            max_label = max(max_label, int(valid_pred.max()))
        if valid_true.size:
            max_label = max(max_label, int(valid_true.max()))
        num_classes = max_label + 1
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    scores = np.zeros((num_classes,), dtype=np.float64)
    valid = np.ones_like(true, dtype=bool) if ignore_index is None else true != ignore_index
    for cls in range(num_classes):
        pred_mask = (pred == cls) & valid
        true_mask = (true == cls) & valid
        intersection = np.logical_and(pred_mask, true_mask).sum()
        union = np.logical_or(pred_mask, true_mask).sum()
        scores[cls] = 1.0 if union == 0 else float(intersection / union)
    return scores


def mean_mask_iou(
    prediction: Tensor | np.ndarray,
    target: Tensor | np.ndarray,
    num_classes: int | None = None,
    ignore_index: int | None = None,
) -> float:
    """Mean per-class mask IoU."""

    return float(np.mean(mask_iou(prediction, target, num_classes, ignore_index)))


def masks_to_boxes(masks: Tensor | np.ndarray) -> np.ndarray:
    """Convert binary masks with shape ``(N, H, W)`` to ``xyxy`` boxes."""

    values = _array(masks)
    if values.ndim == 2:
        values = values[np.newaxis, ...]
    if values.ndim != 3:
        raise ValueError("masks_to_boxes expects shape (H, W) or (N, H, W)")
    boxes = np.zeros((values.shape[0], 4), dtype=np.float32)
    for index, mask in enumerate(values):
        ys, xs = np.where(mask > 0)
        if ys.size == 0 or xs.size == 0:
            continue
        boxes[index] = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]
    return boxes


def mask_to_one_hot(mask: Tensor | np.ndarray, num_classes: int) -> np.ndarray:
    """Convert integer masks to one-hot ``(..., classes, H, W)`` arrays."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    values = _array(mask).astype(np.int64)
    if values.ndim < 2:
        raise ValueError("mask must have at least 2 dimensions")
    if values.min(initial=0) < 0 or values.max(initial=0) >= num_classes:
        raise ValueError("mask values must be within num_classes")
    eye = np.eye(num_classes, dtype=np.float32)
    one_hot = eye[values]
    return np.moveaxis(one_hot, -1, -3)


def one_hot_to_mask(one_hot: Tensor | np.ndarray) -> np.ndarray:
    """Convert one-hot/probability masks with class axis ``-3`` to labels."""

    values = _array(one_hot)
    if values.ndim < 3:
        raise ValueError("one_hot mask must have at least 3 dimensions")
    return np.argmax(values, axis=-3).astype(np.int64)


def resize_mask(mask: Tensor | np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Resize a mask with nearest-neighbor sampling."""

    out_h, out_w = _pair(size, "size")
    values = _array(mask)
    height, width = values.shape[-2], values.shape[-1]
    row_indices = np.minimum((np.arange(out_h) * height // out_h), height - 1)
    col_indices = np.minimum((np.arange(out_w) * width // out_w), width - 1)
    resized = np.take(values, row_indices, axis=-2)
    resized = np.take(resized, col_indices, axis=-1)
    return np.ascontiguousarray(resized)


def center_crop_mask(mask: Tensor | np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Center-crop masks over their last two dimensions."""

    crop_h, crop_w = _pair(size, "size")
    values = _array(mask)
    height, width = values.shape[-2], values.shape[-1]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size cannot exceed mask spatial dimensions")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return np.ascontiguousarray(values[..., top : top + crop_h, left : left + crop_w])


def random_crop_mask(
    mask: Tensor | np.ndarray,
    size: int | tuple[int, int],
    seed: int | None = None,
) -> np.ndarray:
    """Random-crop masks over their last two dimensions."""

    crop_h, crop_w = _pair(size, "size")
    values = _array(mask)
    height, width = values.shape[-2], values.shape[-1]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size cannot exceed mask spatial dimensions")
    rng = random.Random(seed)
    top = rng.randint(0, height - crop_h)
    left = rng.randint(0, width - crop_w)
    return np.ascontiguousarray(values[..., top : top + crop_h, left : left + crop_w])


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or pair")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive")
    return (first, second)


__all__ = [
    "center_crop_mask",
    "mask_iou",
    "mask_to_one_hot",
    "masks_to_boxes",
    "mean_mask_iou",
    "one_hot_to_mask",
    "random_crop_mask",
    "resize_mask",
]
