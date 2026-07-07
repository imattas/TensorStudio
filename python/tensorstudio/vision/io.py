"""Image IO helpers for TensorStudio vision."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor
from tensorstudio.typing import PathLikeStr

from .transforms import to_numpy_image


def _require_pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Image file IO requires Pillow: python -m pip install 'tensorstudio[vision]'"
        ) from exc
    return Image


def load_image(path: PathLikeStr, mode: str | None = "RGB") -> np.ndarray:
    """Load an image file into a NumPy array.

    ``mode`` is passed to Pillow's ``convert`` method when provided. Common
    values are ``"RGB"``, ``"L"``, and ``"RGBA"``.
    """

    Image = _require_pillow()
    with Image.open(path) as image:
        if mode is not None:
            image = image.convert(mode)
        return np.asarray(image.copy())


def save_image(image: Tensor | np.ndarray, path: PathLikeStr) -> None:
    """Save a single image-like tensor or array to disk."""

    Image = _require_pillow()
    array = to_numpy_image(image, channels_last=True)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if np.issubdtype(array.dtype, np.floating):
        array = (np.clip(array, 0.0, 1.0) * 255.0).round().astype(np.uint8)
    elif array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(output_path)


def make_grid(
    images: Tensor | np.ndarray | list[Tensor | np.ndarray],
    nrow: int = 8,
    padding: int = 2,
    value: float = 0.0,
) -> np.ndarray:
    """Arrange CHW/NCHW or HWC/NHWC images into a single HWC grid."""

    if nrow <= 0:
        raise ValueError("nrow must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    batch = _image_batch(images)
    if not batch:
        raise ValueError("make_grid requires at least one image")

    heights = {image.shape[0] for image in batch}
    widths = {image.shape[1] for image in batch}
    channels = {1 if image.ndim == 2 else image.shape[2] for image in batch}
    if len(heights) != 1 or len(widths) != 1 or len(channels) != 1:
        raise ValueError("all images must have the same HWC shape")

    height = next(iter(heights))
    width = next(iter(widths))
    channel_count = next(iter(channels))
    rows = (len(batch) + nrow - 1) // nrow
    cols = min(nrow, len(batch))
    grid_shape = (
        rows * height + padding * max(rows - 1, 0),
        cols * width + padding * max(cols - 1, 0),
        channel_count,
    )
    grid = np.full(grid_shape, value, dtype=batch[0].dtype)
    for index, image in enumerate(batch):
        row = index // nrow
        col = index % nrow
        top = row * (height + padding)
        left = col * (width + padding)
        if image.ndim == 2:
            image = image[..., np.newaxis]
        grid[top : top + height, left : left + width, :] = image
    if channel_count == 1:
        return grid[..., 0]
    return grid


def _image_batch(images: Tensor | np.ndarray | list[Tensor | np.ndarray]) -> list[np.ndarray]:
    if isinstance(images, list):
        return [to_numpy_image(image, channels_last=True) for image in images]
    array = to_numpy_image(images, channels_last=True)
    if array.ndim in {2, 3}:
        return [array]
    if array.ndim == 4:
        return [array[index] for index in range(array.shape[0])]
    raise ValueError("images must be a single image or image batch")


__all__ = ["load_image", "make_grid", "save_image"]
