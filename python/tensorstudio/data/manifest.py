"""Dataset manifests, checksums, and lightweight caching."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .dataset import Dataset

PathLike = str | Path


def file_sha256(path: PathLike, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 digest for a local file."""

    input_path = Path(path)
    digest = hashlib.sha256()
    with input_path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class ManifestEntry:
    """One file entry in a dataset manifest."""

    path: str
    size: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"path": self.path, "size": self.size, "sha256": self.sha256}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ManifestEntry:
        return cls(str(data["path"]), int(data["size"]), str(data["sha256"]))


@dataclass(frozen=True)
class DatasetManifest:
    """Stable manifest for local dataset files."""

    root: str
    entries: tuple[ManifestEntry, ...]
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "tensorstudio.dataset_manifest",
            "version": self.version,
            "root": self.root,
            "files": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DatasetManifest:
        if data.get("format") != "tensorstudio.dataset_manifest":
            raise ValueError("not a TensorStudio dataset manifest")
        version = int(data.get("version", 1))
        if version != 1:
            raise ValueError(f"unsupported dataset manifest version: {version}")
        entries = tuple(ManifestEntry.from_dict(item) for item in data.get("files", []))
        return cls(str(data.get("root", "")), entries, version=version)

    def save(self, path: PathLike) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")
        return output_path

    @classmethod
    def load(cls, path: PathLike) -> DatasetManifest:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("dataset manifest must contain a JSON object")
        return cls.from_dict(data)

    def validate(self, root: PathLike | None = None) -> dict[str, Any]:
        base = Path(root) if root is not None else Path(self.root)
        missing: list[str] = []
        changed: list[str] = []
        checked = 0
        for entry in self.entries:
            path = base / entry.path
            if not path.exists():
                missing.append(entry.path)
                continue
            stat = path.stat()
            digest = file_sha256(path)
            if stat.st_size != entry.size or digest != entry.sha256:
                changed.append(entry.path)
            checked += 1
        return {
            "checked": checked,
            "missing": missing,
            "changed": changed,
            "ok": not missing and not changed,
        }


class CachedDataset(Dataset):
    """Memory cache wrapper for deterministic map-style datasets."""

    def __init__(self, dataset: Dataset, *, max_items: int | None = None) -> None:
        if max_items is not None and max_items <= 0:
            raise ValueError("max_items must be positive when provided")
        self.dataset = dataset
        self.max_items = max_items
        self._cache: OrderedDict[int, object] = OrderedDict()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> object:
        key = int(index)
        if key in self._cache:
            value = self._cache.pop(key)
            self._cache[key] = value
            return value
        value = self.dataset[key]
        self._cache[key] = value
        if self.max_items is not None:
            while len(self._cache) > self.max_items:
                self._cache.popitem(last=False)
        return value

    def clear_cache(self) -> None:
        self._cache.clear()

    @property
    def cache_size(self) -> int:
        return len(self._cache)


def build_dataset_manifest(
    root: PathLike,
    *,
    include: Iterable[str] = ("**/*",),
    exclude: Iterable[str] = (),
) -> DatasetManifest:
    """Build a deterministic manifest for files under ``root``."""

    base = Path(root).resolve()
    if not base.exists() or not base.is_dir():
        raise ValueError(f"dataset root must be an existing directory: {base}")

    include_patterns = tuple(include)
    exclude_patterns = tuple(exclude)
    entries: list[ManifestEntry] = []
    for path in sorted(item for item in base.rglob("*") if item.is_file()):
        relative = path.relative_to(base).as_posix()
        if not _matches(relative, include_patterns):
            continue
        if _matches(relative, exclude_patterns):
            continue
        entries.append(
            ManifestEntry(
                relative,
                path.stat().st_size,
                file_sha256(path),
            )
        )
    return DatasetManifest(str(base), tuple(entries))


def save_dataset_manifest(
    root: PathLike,
    path: PathLike,
    *,
    include: Iterable[str] = ("**/*",),
    exclude: Iterable[str] = (),
) -> Path:
    """Build and save a dataset manifest."""

    return build_dataset_manifest(root, include=include, exclude=exclude).save(path)


def load_dataset_manifest(path: PathLike) -> DatasetManifest:
    """Load a TensorStudio dataset manifest from JSON."""

    return DatasetManifest.load(path)


def validate_dataset_manifest(
    manifest: DatasetManifest | PathLike,
    *,
    root: PathLike | None = None,
) -> dict[str, Any]:
    """Validate a manifest object or manifest file against local files."""

    loaded = DatasetManifest.load(manifest) if isinstance(manifest, (str, Path)) else manifest
    return loaded.validate(root=root)


def cache_dataset(dataset: Dataset, *, max_items: int | None = None) -> CachedDataset:
    """Wrap a map-style dataset in a small LRU memory cache."""

    return CachedDataset(dataset, max_items=max_items)


def _matches(path: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in {"*", "**/*"} or fnmatch.fnmatch(path, pattern) for pattern in patterns)


__all__ = [
    "CachedDataset",
    "DatasetManifest",
    "ManifestEntry",
    "build_dataset_manifest",
    "cache_dataset",
    "file_sha256",
    "load_dataset_manifest",
    "save_dataset_manifest",
    "validate_dataset_manifest",
]
