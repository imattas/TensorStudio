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
    box_array = np.asarray(boxes.numpy() if isinstance(boxes, Tensor) else boxes, dtype=np.float64)
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


def _require_image_draw() -> Any:
    try:
        from PIL import ImageDraw
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Bounding-box drawing requires Pillow: python -m pip install 'tensorstudio[vision]'"
        ) from exc
    return ImageDraw


__all__ = ["draw_bounding_boxes"]
