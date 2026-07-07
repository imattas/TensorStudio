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


def from_detection_folder(
    root: PathLikeStr,
    *,
    transform: Callable[[Any], Any] | None = None,
    target_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
    images_dir: str = "images",
    annotations_dir: str = "annotations",
    mode: str | None = "RGB",
) -> Dataset:
    """Create a detection dataset from image and JSON annotation folders."""

    from tensorstudio.vision import DetectionFolder

    return DetectionFolder(
        root,
        transform=transform,
        target_transform=target_transform,
        images_dir=images_dir,
        annotations_dir=annotations_dir,
        mode=mode,
    )


def from_segmentation_folder(
    root: PathLikeStr,
    *,
    transform: Callable[[Any], Any] | None = None,
    target_transform: Callable[[Any], Any] | None = None,
    images_dir: str = "images",
    masks_dir: str = "masks",
    mode: str | None = "RGB",
) -> Dataset:
    """Create a segmentation dataset from image and mask folders."""

    from tensorstudio.vision import SegmentationFolder

    return SegmentationFolder(
        root,
        transform=transform,
        target_transform=target_transform,
        images_dir=images_dir,
        masks_dir=masks_dir,
        mode=mode,
    )


__all__ = [
    "ArrayDataset",
    "Dataset",
    "TensorDataset",
    "from_detection_folder",
    "from_arrays",
    "from_image_folder",
    "from_segmentation_folder",
    "from_tensors",
]
