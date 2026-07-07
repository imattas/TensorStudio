"""Visualization helpers for TensorStudio vision."""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor

from .io import _require_pillow
from .transforms import to_numpy_image


def draw_bounding_boxes(
    image: Tensor | np.ndarray,
    boxes: Tensor | np.ndarray | list[list[float]],
    labels: list[str] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 1,
) -> np.ndarray:
    """Draw ``xyxy`` bounding boxes on an image and return an HWC uint8 array."""

    if width <= 0:
        raise ValueError("width must be positive")
    Image = _require_pillow()
    ImageDraw = _require_image_draw()
    array = to_numpy_image(image, channels_last=True)
    if array.ndim == 2:
        array = np.repeat(array[..., np.newaxis], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError("draw_bounding_boxes expects a single image")
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    pil_image = Image.fromarray(array)
    draw = ImageDraw.Draw(pil_image)
    box_array: np.ndarray = np.asarray(
        boxes.numpy() if isinstance(boxes, Tensor) else boxes,
        dtype=np.float64,
    )
    if box_array.ndim != 2 or box_array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    if labels is not None and len(labels) != box_array.shape[0]:
        raise ValueError("labels length must match boxes")
    for index, box in enumerate(box_array):
        xyxy = tuple(float(value) for value in box)
        draw.rectangle(xyxy, outline=color, width=width)
        if labels is not None:
            draw.text((xyxy[0], xyxy[1]), labels[index], fill=color)
    return np.asarray(pil_image)


def draw_predictions(
    image: Tensor | np.ndarray,
    boxes: Tensor | np.ndarray | list[list[float]],
    labels: list[str],
    scores: Tensor | np.ndarray | list[float] | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
    width: int = 1,
) -> np.ndarray:
    """Draw prediction boxes with optional scores."""

    text_labels = list(labels)
    if scores is not None:
        score_values = np.asarray(
            scores.numpy() if isinstance(scores, Tensor) else scores
        ).reshape(-1)
        if len(score_values) != len(text_labels):
            raise ValueError("scores length must match labels")
        text_labels = [
            f"{label} {score:.2f}"
            for label, score in zip(text_labels, score_values, strict=True)
        ]
    return draw_bounding_boxes(image, boxes, labels=text_labels, color=color, width=width)


def overlay_mask(
    image: Tensor | np.ndarray,
    mask: Tensor | np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.5,
) -> np.ndarray:
    """Overlay a binary/integer mask on an image."""

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must satisfy 0 <= alpha <= 1")
    array = _uint8_image(image)
    mask_array = np.asarray(mask.numpy() if isinstance(mask, Tensor) else mask)
    if mask_array.ndim == 3:
        mask_array = mask_array.argmax(axis=0)
    if mask_array.shape != array.shape[:2]:
        raise ValueError("mask spatial shape must match image")
    overlay = array.copy()
    active = mask_array > 0
    color_array = np.asarray(color, dtype=np.float32)
    overlay[active] = (
        array[active].astype(np.float32) * (1.0 - alpha) + color_array * alpha
    ).round()
    return overlay.astype(np.uint8)


def feature_map_grid(features: Tensor | np.ndarray, nrow: int = 8) -> np.ndarray:
    """Visualize CxHxW or NxCxHxW feature maps as a uint8 grid."""

    if nrow <= 0:
        raise ValueError("nrow must be positive")
    array = np.asarray(features.numpy() if isinstance(features, Tensor) else features)
    if array.ndim == 4:
        array = array[0]
    if array.ndim != 3:
        raise ValueError("features must have shape (C, H, W) or (N, C, H, W)")
    maps = []
    for channel in array:
        values = channel.astype(np.float32)
        values = values - float(values.min())
        max_value = float(values.max())
        if max_value > 0.0:
            values = values / max_value
        maps.append((values * 255.0).round().astype(np.uint8))
    rows = (len(maps) + nrow - 1) // nrow
    height, width = maps[0].shape
    grid = np.zeros((rows * height, min(nrow, len(maps)) * width), dtype=np.uint8)
    for index, fmap in enumerate(maps):
        row = index // nrow
        col = index % nrow
        grid[row * height : (row + 1) * height, col * width : (col + 1) * width] = fmap
    return grid


def _uint8_image(image: Tensor | np.ndarray) -> np.ndarray:
    array = to_numpy_image(image, channels_last=True)
    if array.ndim == 2:
        array = np.repeat(array[..., np.newaxis], 3, axis=-1)
    if array.ndim != 3:
        raise ValueError("expected a single image")
    if np.issubdtype(array.dtype, np.floating):
        return (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    if array.dtype != np.uint8:
        return np.clip(array, 0, 255).astype(np.uint8)
    return array.copy()


def _require_image_draw() -> Any:
    try:
        from PIL import ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bounding-box drawing requires Pillow: python -m pip install 'tensorstudio[vision]'"
        ) from exc
    return ImageDraw


__all__ = ["draw_bounding_boxes", "draw_predictions", "feature_map_grid", "overlay_mask"]
