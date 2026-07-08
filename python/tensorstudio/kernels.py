"""Custom kernel extension registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

KernelCallable = Callable[..., Any]


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


__all__ = [
    "DEFAULT_KERNEL_REGISTRY",
    "KernelCallable",
    "KernelInfo",
    "KernelRegistry",
    "call_kernel",
    "get_kernel",
    "kernel_info",
    "list_kernels",
    "register_kernel",
    "unregister_kernel",
]
