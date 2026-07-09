"""Custom kernel extension registry."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
    import tomli as tomllib

from . import hardware
from .dtypes import normalize_dtype
from .typing import PathLikeStr

KernelCallable = Callable[..., Any]
_MANIFEST_FILENAMES = (
    "tensorstudio-kernel.json",
    "tensorstudio-kernel.toml",
    "tensorstudio_plugin.json",
    "tensorstudio_plugin.toml",
)


@dataclass(frozen=True)
class KernelInfo:
    name: str
    description: str = ""
    backend: str = "python"

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "description": self.description, "backend": self.backend}


class KernelRegistry:
    """Registry for experimental custom tensor kernels."""

    def __init__(self) -> None:
        self._kernels: dict[str, tuple[KernelInfo, KernelCallable]] = {}

    def register(
        self,
        name: str,
        function: KernelCallable,
        *,
        description: str = "",
        backend: str = "python",
        overwrite: bool = False,
    ) -> None:
        if not name:
            raise ValueError("kernel name must be non-empty")
        if not overwrite and name in self._kernels:
            raise ValueError(f"kernel {name!r} is already registered")
        self._kernels[name] = (KernelInfo(name, description, backend), function)

    def unregister(self, name: str) -> None:
        self._kernels.pop(name)

    def get(self, name: str) -> KernelCallable:
        if name not in self._kernels:
            raise KeyError(f"unknown TensorStudio kernel: {name}")
        return self._kernels[name][1]

    def info(self, name: str) -> dict[str, str]:
        if name not in self._kernels:
            raise KeyError(f"unknown TensorStudio kernel: {name}")
        return self._kernels[name][0].to_dict()

    def list(self) -> list[dict[str, str]]:
        return [self._kernels[name][0].to_dict() for name in sorted(self._kernels)]

    def call(self, name: str, *args: Any, **kwargs: Any) -> Any:
        return self.get(name)(*args, **kwargs)


DEFAULT_KERNEL_REGISTRY = KernelRegistry()


def register_kernel(
    name: str,
    function: KernelCallable,
    *,
    description: str = "",
    backend: str = "python",
    overwrite: bool = False,
) -> None:
    DEFAULT_KERNEL_REGISTRY.register(
        name,
        function,
        description=description,
        backend=backend,
        overwrite=overwrite,
    )


def unregister_kernel(name: str) -> None:
    DEFAULT_KERNEL_REGISTRY.unregister(name)


def get_kernel(name: str) -> KernelCallable:
    return DEFAULT_KERNEL_REGISTRY.get(name)


def kernel_info(name: str) -> dict[str, str]:
    return DEFAULT_KERNEL_REGISTRY.info(name)


def list_kernels() -> list[dict[str, str]]:
    return DEFAULT_KERNEL_REGISTRY.list()


def call_kernel(name: str, *args: Any, **kwargs: Any) -> Any:
    return DEFAULT_KERNEL_REGISTRY.call(name, *args, **kwargs)


@dataclass(frozen=True)
class KernelManifestEntry:
    """Metadata for one custom backend kernel capability."""

    op: str
    dtypes: tuple[str, ...]
    memory_space: str = "device"
    execution_mode: str = "sync"
    forward: bool = True
    backward: bool = False
    available: bool = True
    deterministic: bool = True
    priority: int = 50
    reason: str = ""


@dataclass(frozen=True)
class KernelManifest:
    """Validated custom backend manifest metadata."""

    path: Path
    schema_version: int
    backend: str
    compiled: bool
    available: bool
    device_count: int
    reason: str
    kernels: tuple[KernelManifestEntry, ...]
    library: str | None = None


def _load_mapping(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    with path.open("rb") as file:
        if suffix == ".json":
            value = json.load(file)
        elif suffix == ".toml":
            value = tomllib.load(file)
        else:
            raise ValueError(f"unsupported TensorStudio kernel manifest extension: {path.suffix}")
    if not isinstance(value, dict):
        raise ValueError("TensorStudio kernel manifest root must be a table/object")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"kernel manifest field {name!r} must be a non-empty string")
    return value.strip()


def _bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"kernel manifest field {name!r} must be a boolean")
    return value


def _int(value: Any, name: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"kernel manifest field {name!r} must be an integer")
    return value


def _validate_backend_name(name: str) -> str:
    normalized = name.lower()
    if normalized in {"cpu", "cuda", "gpu", "metal", "mps"}:
        raise ValueError(f"kernel manifest backend name {name!r} is reserved")
    for char in normalized:
        valid = char.isascii() and (char.islower() or char.isdigit() or char in {"_", "-", "."})
        if not valid:
            raise ValueError(
                "kernel manifest backend names must contain only lowercase ASCII letters, "
                "digits, underscores, dashes, or dots"
            )
    return normalized


def _enum_string(value: Any, name: str, allowed: set[str]) -> str:
    normalized = _string(value, name).lower()
    if normalized not in allowed:
        choices = ", ".join(sorted(allowed))
        raise ValueError(f"kernel manifest field {name!r} must be one of: {choices}")
    return normalized


def _backend_table(raw: dict[str, Any]) -> dict[str, Any]:
    backend = raw.get("backend", raw)
    if not isinstance(backend, dict):
        raise ValueError("kernel manifest field 'backend' must be a table/object")
    return backend


def load_kernel_manifest(path: PathLikeStr) -> KernelManifest:
    """Load and validate one TensorStudio custom-kernel manifest."""

    manifest_path = Path(path)
    raw = _load_mapping(manifest_path)
    schema_version = _int(raw.get("schema_version", raw.get("version", 1)), "schema_version")
    if schema_version != 1:
        raise ValueError(
            f"unsupported TensorStudio kernel manifest schema_version: {schema_version}"
        )

    backend = _backend_table(raw)
    backend_name = _validate_backend_name(_string(backend.get("name"), "backend.name"))
    compiled = _bool(backend.get("compiled", True), "backend.compiled")
    available = _bool(backend.get("available", False), "backend.available")
    device_count = _int(backend.get("device_count", 0), "backend.device_count")
    if available or device_count != 0:
        raise ValueError(
            "kernel manifests cannot mark external backends available until TensorStudio "
            "has executable plugin storage hooks"
        )
    reason = str(backend.get("reason", "descriptor only"))

    library_value = raw.get("library")
    library = _string(library_value, "library") if library_value is not None else None

    raw_kernels = raw.get("kernels", [])
    if not isinstance(raw_kernels, list):
        raise ValueError("kernel manifest field 'kernels' must be a list")
    kernels: list[KernelManifestEntry] = []
    for index, item in enumerate(raw_kernels):
        if not isinstance(item, dict):
            raise ValueError(f"kernel manifest kernels[{index}] must be a table/object")
        op = _string(item.get("op"), f"kernels[{index}].op")
        raw_dtypes = item.get("dtypes")
        if not isinstance(raw_dtypes, list) or not raw_dtypes:
            raise ValueError(f"kernel manifest kernels[{index}].dtypes must be a non-empty list")
        dtypes = tuple(
            normalize_dtype(_string(value, f"kernels[{index}].dtypes[]"))
            for value in raw_dtypes
        )
        kernels.append(
            KernelManifestEntry(
                op=op,
                dtypes=dtypes,
                memory_space=_enum_string(
                    item.get("memory_space", "device"),
                    f"kernels[{index}].memory_space",
                    {"host", "device", "unified"},
                ),
                execution_mode=_enum_string(
                    item.get("execution_mode", "sync"),
                    f"kernels[{index}].execution_mode",
                    {"sync", "async"},
                ),
                forward=_bool(item.get("forward", True), f"kernels[{index}].forward"),
                backward=_bool(item.get("backward", False), f"kernels[{index}].backward"),
                available=_bool(item.get("available", True), f"kernels[{index}].available"),
                deterministic=_bool(
                    item.get("deterministic", True),
                    f"kernels[{index}].deterministic",
                ),
                priority=_int(item.get("priority", 50), f"kernels[{index}].priority"),
                reason=str(item.get("reason", "")),
            )
        )

    return KernelManifest(
        path=manifest_path,
        schema_version=schema_version,
        backend=backend_name,
        compiled=compiled,
        available=available,
        device_count=device_count,
        reason=reason,
        kernels=tuple(kernels),
        library=library,
    )


def validate_kernel_manifest(path: PathLikeStr) -> KernelManifest:
    """Validate a manifest and return parsed metadata."""

    return load_kernel_manifest(path)


def discover_kernel_manifests(root: PathLikeStr) -> list[KernelManifest]:
    """Find TensorStudio kernel manifests below a directory without executing them."""

    base = Path(root)
    if base.is_file():
        return [load_kernel_manifest(base)]
    manifests: list[KernelManifest] = []
    for name in _MANIFEST_FILENAMES:
        for path in sorted(base.rglob(name)):
            manifests.append(load_kernel_manifest(path))
    return manifests


def register_kernel_manifest(manifest: KernelManifest) -> None:
    """Register backend and kernel metadata from a validated manifest."""

    hardware.register_backend(
        manifest.backend,
        compiled=manifest.compiled,
        available=manifest.available,
        device_count=manifest.device_count,
        reason=manifest.reason,
    )
    for kernel in manifest.kernels:
        hardware.register_backend_kernel(
            manifest.backend,
            kernel.op,
            kernel.dtypes,
            forward=kernel.forward,
            backward=kernel.backward,
            available=kernel.available,
            reason=kernel.reason,
            memory_space=kernel.memory_space,
            execution_mode=kernel.execution_mode,
            deterministic=kernel.deterministic,
            priority=kernel.priority,
        )


__all__ = [
    "DEFAULT_KERNEL_REGISTRY",
    "KernelCallable",
    "KernelInfo",
    "KernelManifest",
    "KernelManifestEntry",
    "KernelRegistry",
    "call_kernel",
    "discover_kernel_manifests",
    "get_kernel",
    "kernel_info",
    "list_kernels",
    "load_kernel_manifest",
    "register_kernel",
    "register_kernel_manifest",
    "unregister_kernel",
    "validate_kernel_manifest",
]
