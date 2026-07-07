"""Vision datasets."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from tensorstudio.data import Dataset
from tensorstudio.tensor import Tensor
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


__all__ = ["IMG_EXTENSIONS", "ImageFolder", "ImageList"]
