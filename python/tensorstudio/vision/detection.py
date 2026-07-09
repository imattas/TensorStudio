"""Detection utilities for boxes, anchors, and non-maximum suppression."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor


def _array(value: Tensor | np.ndarray | Sequence[Any]) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.numpy()
    return np.asarray(value)


def _boxes(value: Tensor | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    boxes = _array(value).astype(np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    return boxes


def box_area(boxes: Tensor | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Area of ``xyxy`` boxes."""

    values = _boxes(boxes)
    widths = np.clip(values[:, 2] - values[:, 0], 0.0, None)
    heights = np.clip(values[:, 3] - values[:, 1], 0.0, None)
    return widths * heights


def box_iou(
    boxes1: Tensor | np.ndarray | Sequence[Sequence[float]],
    boxes2: Tensor | np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Pairwise IoU for ``xyxy`` boxes."""

    first = _boxes(boxes1)
    second = _boxes(boxes2)
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    intersection = wh[..., 0] * wh[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


def generalized_box_iou(
    boxes1: Tensor | np.ndarray | Sequence[Sequence[float]],
    boxes2: Tensor | np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Pairwise generalized IoU for ``xyxy`` boxes."""

    first = _boxes(boxes1)
    second = _boxes(boxes2)
    iou = box_iou(first, second)
    top_left = np.minimum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.maximum(first[:, None, 2:], second[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    enclosing = wh[..., 0] * wh[..., 1]
    union = box_area(first)[:, None] + box_area(second)[None, :]
    intersection = iou * (union / (1.0 + iou))
    actual_union = union - intersection
    penalty = np.divide(
        enclosing - actual_union,
        enclosing,
        out=np.zeros_like(iou),
        where=enclosing > 0,
    )
    return iou - penalty


def distance_box_iou(
    boxes1: Tensor | np.ndarray | Sequence[Sequence[float]],
    boxes2: Tensor | np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Pairwise distance IoU for ``xyxy`` boxes."""

    first = _boxes(boxes1)
    second = _boxes(boxes2)
    iou = box_iou(first, second)
    center1 = (first[:, :2] + first[:, 2:]) / 2.0
    center2 = (second[:, :2] + second[:, 2:]) / 2.0
    center_distance = np.sum((center1[:, None, :] - center2[None, :, :]) ** 2, axis=-1)
    top_left = np.minimum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.maximum(first[:, None, 2:], second[None, :, 2:])
    diagonal = np.sum(np.clip(bottom_right - top_left, 0.0, None) ** 2, axis=-1)
    return iou - np.divide(center_distance, diagonal, out=np.zeros_like(iou), where=diagonal > 0)


def nms(
    boxes: Tensor | np.ndarray | Sequence[Sequence[float]],
    scores: Tensor | np.ndarray | Sequence[float],
    iou_threshold: float = 0.5,
    score_threshold: float | None = None,
    top_k: int | None = None,
) -> list[int]:
    """Non-maximum suppression returning kept indices."""

    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in [0, 1]")
    values = _boxes(boxes)
    score_values = _array(scores).astype(np.float64).reshape(-1)
    if values.shape[0] != score_values.shape[0]:
        raise ValueError("boxes and scores must have the same length")
    order = np.argsort(score_values)[::-1]
    if score_threshold is not None:
        order = order[score_values[order] >= score_threshold]
    keep: list[int] = []
    while order.size:
        index = int(order[0])
        keep.append(index)
        if top_k is not None and len(keep) >= top_k:
            break
        if order.size == 1:
            break
        overlaps = box_iou(values[[index]], values[order[1:]])[0]
        order = order[1:][overlaps <= iou_threshold]
    return keep


def box_xyxy_to_cxcywh(boxes: Tensor | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Convert ``xyxy`` boxes to ``center_x, center_y, width, height``."""

    values = _boxes(boxes)
    centers = (values[:, :2] + values[:, 2:]) / 2.0
    sizes = np.clip(values[:, 2:] - values[:, :2], 0.0, None)
    return np.concatenate([centers, sizes], axis=1)


def box_cxcywh_to_xyxy(boxes: Tensor | np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Convert ``center_x, center_y, width, height`` boxes to ``xyxy``."""

    values = _boxes(boxes)
    half = values[:, 2:] / 2.0
    return np.concatenate([values[:, :2] - half, values[:, :2] + half], axis=1)


def encode_boxes(
    boxes: Tensor | np.ndarray | Sequence[Sequence[float]],
    anchors: Tensor | np.ndarray | Sequence[Sequence[float]],
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Encode target boxes relative to anchors."""

    target = box_xyxy_to_cxcywh(boxes)
    anchor = box_xyxy_to_cxcywh(anchors)
    if target.shape != anchor.shape:
        raise ValueError("boxes and anchors must have matching shape")
    eps = 1e-12
    dx = (target[:, 0] - anchor[:, 0]) / np.maximum(anchor[:, 2], eps) * weights[0]
    dy = (target[:, 1] - anchor[:, 1]) / np.maximum(anchor[:, 3], eps) * weights[1]
    dw = np.log(np.maximum(target[:, 2], eps) / np.maximum(anchor[:, 2], eps)) * weights[2]
    dh = np.log(np.maximum(target[:, 3], eps) / np.maximum(anchor[:, 3], eps)) * weights[3]
    return np.stack([dx, dy, dw, dh], axis=1)


def decode_boxes(
    deltas: Tensor | np.ndarray | Sequence[Sequence[float]],
    anchors: Tensor | np.ndarray | Sequence[Sequence[float]],
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> np.ndarray:
    """Decode anchor-relative box deltas to ``xyxy`` boxes."""

    delta = _boxes(deltas)
    anchor = box_xyxy_to_cxcywh(anchors)
    if delta.shape != anchor.shape:
        raise ValueError("deltas and anchors must have matching shape")
    dx = delta[:, 0] / weights[0]
    dy = delta[:, 1] / weights[1]
    dw = delta[:, 2] / weights[2]
    dh = delta[:, 3] / weights[3]
    centers = np.stack(
        [dx * anchor[:, 2] + anchor[:, 0], dy * anchor[:, 3] + anchor[:, 1]],
        axis=1,
    )
    sizes = np.stack([np.exp(dw) * anchor[:, 2], np.exp(dh) * anchor[:, 3]], axis=1)
    return box_cxcywh_to_xyxy(np.concatenate([centers, sizes], axis=1))


def generate_anchors(
    feature_shape: tuple[int, int],
    stride: int | tuple[int, int],
    sizes: Sequence[float] = (32.0,),
    ratios: Sequence[float] = (0.5, 1.0, 2.0),
) -> np.ndarray:
    """Generate grid anchors in ``xyxy`` format."""

    height, width = int(feature_shape[0]), int(feature_shape[1])
    stride_h, stride_w = (stride, stride) if isinstance(stride, int) else stride
    if height <= 0 or width <= 0 or stride_h <= 0 or stride_w <= 0:
        raise ValueError("feature_shape and stride values must be positive")
    anchors: list[list[float]] = []
    for row in range(height):
        for col in range(width):
            center_x = (col + 0.5) * stride_w
            center_y = (row + 0.5) * stride_h
            for size in sizes:
                for ratio in ratios:
                    if size <= 0.0 or ratio <= 0.0:
                        raise ValueError("sizes and ratios must be positive")
                    anchor_w = size * math.sqrt(1.0 / ratio)
                    anchor_h = size * math.sqrt(ratio)
                    anchors.append(
                        [
                            center_x - anchor_w / 2.0,
                            center_y - anchor_h / 2.0,
                            center_x + anchor_w / 2.0,
                            center_y + anchor_h / 2.0,
                        ]
                    )
    return np.asarray(anchors, dtype=np.float32)


__all__ = [
    "box_area",
    "box_cxcywh_to_xyxy",
    "box_iou",
    "box_xyxy_to_cxcywh",
    "decode_boxes",
    "distance_box_iou",
    "encode_boxes",
    "generalized_box_iou",
    "generate_anchors",
    "nms",
]
