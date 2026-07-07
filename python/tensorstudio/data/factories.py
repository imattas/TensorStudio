"""Dataset creation factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from tensorstudio.typing import PathLikeStr

from .dataset import ArrayDataset, Dataset, TensorDataset, from_arrays, from_tensors


def from_image_folder(
    root: PathLikeStr,
    *,
    transform: Callable[[Any], Any] | None = None,
    target_transform: Callable[[int], Any] | None = None,
    mode: str | None = "RGB",
) -> Dataset:
    """Create an image-classification dataset from ``root/class_name/image`` folders."""

    from tensorstudio.vision import ImageFolder

    return ImageFolder(root, transform=transform, target_transform=target_transform, mode=mode)


__all__ = [
    "ArrayDataset",
    "Dataset",
    "TensorDataset",
    "from_arrays",
    "from_image_folder",
    "from_tensors",
]
