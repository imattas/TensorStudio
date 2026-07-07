"""Vision datasets."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tensorstudio.data import Dataset
from tensorstudio.tensor import Tensor, tensor
from tensorstudio.typing import PathLikeStr

from .io import load_image
from .transforms import to_tensor

IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff", ".webp")


class ImageFolder(Dataset):
    """Dataset for directory trees laid out as ``root/class_name/image.ext``."""

    classes: list[str]
    class_to_idx: dict[str, int]
    samples: list[tuple[Path, int]]
    targets: list[int]

    def __init__(
        self,
        root: PathLikeStr,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        extensions: tuple[str, ...] = IMG_EXTENSIONS,
        mode: str | None = "RGB",
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"ImageFolder root does not exist: {self.root}")
        self.transform = transform
        self.target_transform = target_transform
        self.extensions = tuple(extension.lower() for extension in extensions)
        self.mode = mode
        self.classes, self.class_to_idx = self._find_classes()
        self.samples = self._make_dataset()
        if not self.samples:
            raise ValueError(f"ImageFolder found no supported images in {self.root}")
        self.targets = [target for _, target in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        path, target = self.samples[index]
        image: Any = load_image(path, mode=self.mode)
        image = to_tensor(image) if self.transform is None else self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target

    def _find_classes(self) -> tuple[list[str], dict[str, int]]:
        classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        if not classes:
            raise ValueError(f"ImageFolder found no class folders in {self.root}")
        class_to_idx = {name: index for index, name in enumerate(classes)}
        return classes, class_to_idx

    def _make_dataset(self) -> list[tuple[Path, int]]:
        samples: list[tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = self.root / class_name
            target = self.class_to_idx[class_name]
            for path in sorted(class_dir.rglob("*")):
                if path.is_file() and path.suffix.lower() in self.extensions:
                    samples.append((path, target))
        return samples


class ImageList(Dataset):
    """Dataset from explicit image paths and targets."""

    def __init__(
        self,
        samples: list[tuple[PathLikeStr, int]],
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        mode: str | None = "RGB",
    ) -> None:
        if not samples:
            raise ValueError("ImageList requires at least one sample")
        self.samples = [(Path(path), int(target)) for path, target in samples]
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor | Any, Any]:
        path, target = self.samples[index]
        image: Any = load_image(path, mode=self.mode)
        image = to_tensor(image) if self.transform is None else self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class DetectionFolder(Dataset):
    """Detection dataset laid out as ``root/images`` and ``root/annotations``.

    Annotation files are JSON documents named after the image stem and contain
    ``boxes`` in ``xyxy`` format plus optional integer ``labels``.
    """

    def __init__(
        self,
        root: PathLikeStr,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        images_dir: str = "images",
        annotations_dir: str = "annotations",
        mode: str | None = "RGB",
    ) -> None:
        self.root = Path(root)
        self.image_root = self.root / images_dir
        self.annotation_root = self.root / annotations_dir
        if not self.image_root.exists():
            raise FileNotFoundError(f"DetectionFolder image root does not exist: {self.image_root}")
        if not self.annotation_root.exists():
            raise FileNotFoundError(
                f"DetectionFolder annotation root does not exist: {self.annotation_root}"
            )
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.samples = [
            path
            for path in sorted(self.image_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        if not self.samples:
            raise ValueError(f"DetectionFolder found no supported images in {self.image_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, dict[str, Any]]:
        path = self.samples[index]
        image: Any = load_image(path, mode=self.mode)
        image = to_tensor(image) if self.transform is None else self.transform(image)
        annotation_path = self.annotation_root / f"{path.stem}.json"
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        boxes = data.get("boxes", [])
        labels = data.get("labels", [1 for _ in boxes])
        target: dict[str, Any] = {
            "boxes": tensor(boxes, dtype="float32"),
            "labels": tensor(labels, dtype="int64"),
            "image_id": data.get("image_id", path.stem),
        }
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


class SegmentationFolder(Dataset):
    """Segmentation dataset laid out as ``root/images`` and ``root/masks``."""

    def __init__(
        self,
        root: PathLikeStr,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[Any], Any] | None = None,
        images_dir: str = "images",
        masks_dir: str = "masks",
        mode: str | None = "RGB",
    ) -> None:
        self.root = Path(root)
        self.image_root = self.root / images_dir
        self.mask_root = self.root / masks_dir
        if not self.image_root.exists():
            raise FileNotFoundError(
                f"SegmentationFolder image root does not exist: {self.image_root}"
            )
        if not self.mask_root.exists():
            raise FileNotFoundError(
                f"SegmentationFolder mask root does not exist: {self.mask_root}"
            )
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.samples = [
            path
            for path in sorted(self.image_root.rglob("*"))
            if path.is_file() and path.suffix.lower() in IMG_EXTENSIONS
        ]
        if not self.samples:
            raise ValueError(f"SegmentationFolder found no supported images in {self.image_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        path = self.samples[index]
        image: Any = load_image(path, mode=self.mode)
        image = to_tensor(image) if self.transform is None else self.transform(image)
        mask_path = self._mask_path(path.stem)
        mask: Any = load_image(mask_path, mode="L")
        mask = tensor(mask.astype("int64").tolist(), dtype="int64")
        if self.target_transform is not None:
            mask = self.target_transform(mask)
        return image, mask

    def _mask_path(self, stem: str) -> Path:
        for extension in IMG_EXTENSIONS:
            candidate = self.mask_root / f"{stem}{extension}"
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"missing mask for image stem {stem!r}")


__all__ = [
    "DetectionFolder",
    "IMG_EXTENSIONS",
    "ImageFolder",
    "ImageList",
    "SegmentationFolder",
]
