"""Vision datasets."""

from __future__ import annotations

import hashlib
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
_MANIFEST_VERSION = 1


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


class ImageManifestDataset(Dataset):
    """Dataset backed by a validated image manifest."""

    def __init__(
        self,
        manifest: dict[str, Any] | PathLikeStr,
        root: PathLikeStr | None = None,
        transform: Callable[[Any], Any] | None = None,
        target_transform: Callable[[int], Any] | None = None,
        mode: str | None = "RGB",
    ) -> None:
        loaded = manifest if isinstance(manifest, dict) else load_image_manifest(manifest)
        if loaded.get("format") != "tensorstudio.image_folder_manifest":
            raise ValueError("manifest is not a TensorStudio image-folder manifest")
        samples = loaded.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError("image manifest does not contain samples")
        self.manifest = loaded
        self.root = Path(root) if root is not None else Path(str(loaded.get("root", ".")))
        self.classes = list(loaded.get("classes", []))
        self.class_to_idx = dict(loaded.get("class_to_idx", {}))
        self.samples = samples
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.targets = [int(sample["target"]) for sample in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Tensor | Any, Any]:
        sample = self.samples[index]
        path = self.root / str(sample["path"])
        target = int(sample["target"])
        image: Any = load_image(path, mode=self.mode)
        image = to_tensor(image) if self.transform is None else self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return image, target


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _image_samples(
    root: Path,
    extensions: tuple[str, ...],
) -> tuple[list[str], dict[str, int], list[tuple[Path, int, str]]]:
    classes = sorted(path.name for path in root.iterdir() if path.is_dir())
    if not classes:
        raise ValueError(f"image manifest found no class folders in {root}")
    class_to_idx = {name: index for index, name in enumerate(classes)}
    samples: list[tuple[Path, int, str]] = []
    for class_name in classes:
        class_dir = root / class_name
        target = class_to_idx[class_name]
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                samples.append((path, target, class_name))
    if not samples:
        raise ValueError(f"image manifest found no supported images in {root}")
    return classes, class_to_idx, samples


def build_image_manifest(
    root: PathLikeStr,
    output_path: PathLikeStr | None = None,
    *,
    extensions: tuple[str, ...] = IMG_EXTENSIONS,
    checksum: bool = True,
) -> dict[str, Any]:
    """Build a deterministic manifest for an ImageFolder-style dataset."""

    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"image manifest root does not exist: {root_path}")
    normalized_extensions = tuple(extension.lower() for extension in extensions)
    classes, class_to_idx, samples = _image_samples(root_path, normalized_extensions)
    manifest_samples: list[dict[str, Any]] = []
    total_bytes = 0
    for path, target, class_name in samples:
        size_bytes = path.stat().st_size
        total_bytes += size_bytes
        manifest_samples.append(
            {
                "path": path.relative_to(root_path).as_posix(),
                "target": int(target),
                "class_name": class_name,
                "size_bytes": int(size_bytes),
                "sha256": _sha256_file(path) if checksum else None,
            }
        )
    manifest: dict[str, Any] = {
        "format": "tensorstudio.image_folder_manifest",
        "version": _MANIFEST_VERSION,
        "root": str(root_path),
        "classes": classes,
        "class_to_idx": class_to_idx,
        "extensions": list(normalized_extensions),
        "checksum": checksum,
        "sample_count": len(manifest_samples),
        "total_bytes": total_bytes,
        "samples": manifest_samples,
    }
    if output_path is not None:
        save_image_manifest(manifest, output_path)
    return manifest


def save_image_manifest(manifest: dict[str, Any], path: PathLikeStr) -> Path:
    """Write an image manifest as stable JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output_path


def load_image_manifest(path: PathLikeStr) -> dict[str, Any]:
    """Load an image manifest JSON file."""

    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("image manifest JSON must contain an object")
    return manifest


def validate_image_manifest(
    manifest: dict[str, Any] | PathLikeStr,
    root: PathLikeStr | None = None,
    *,
    checksum: bool | None = None,
) -> dict[str, Any]:
    """Validate that manifest files still exist and optionally match checksums."""

    loaded = manifest if isinstance(manifest, dict) else load_image_manifest(manifest)
    root_path = Path(root) if root is not None else Path(str(loaded.get("root", ".")))
    check_hashes = bool(loaded.get("checksum", False)) if checksum is None else checksum
    samples = loaded.get("samples", [])
    if not isinstance(samples, list):
        raise ValueError("image manifest samples must be a list")

    missing_files: list[str] = []
    size_mismatches: list[str] = []
    checksum_mismatches: list[str] = []
    for sample in samples:
        if not isinstance(sample, dict):
            raise ValueError("image manifest sample entries must be objects")
        relative = str(sample.get("path", ""))
        path = root_path / relative
        if not path.is_file():
            missing_files.append(relative)
            continue
        expected_size = sample.get("size_bytes")
        if expected_size is not None and int(expected_size) != path.stat().st_size:
            size_mismatches.append(relative)
        expected_hash = sample.get("sha256")
        if check_hashes and expected_hash and _sha256_file(path) != expected_hash:
            checksum_mismatches.append(relative)

    valid = not missing_files and not size_mismatches and not checksum_mismatches
    return {
        "valid": valid,
        "root": str(root_path),
        "sample_count": len(samples),
        "missing_files": missing_files,
        "size_mismatches": size_mismatches,
        "checksum_mismatches": checksum_mismatches,
        "reason": "" if valid else "manifest validation found missing or changed files",
    }


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
    "ImageManifestDataset",
    "SegmentationFolder",
    "build_image_manifest",
    "load_image_manifest",
    "save_image_manifest",
    "validate_image_manifest",
]
