"""CPU performance configuration helpers."""

from __future__ import annotations

from typing import TypedDict, cast

from . import _C


class PerformanceInfo(TypedDict):
    """Runtime and compile-time CPU backend details."""

    num_threads: int
    threads_enabled: bool
    storage_pool_enabled: bool
    blas_enabled: bool
    simd_enabled: bool
    simd_level: str


def get_num_threads() -> int:
    """Return the configured native CPU thread count."""

    return int(_C.get_num_threads())


def set_num_threads(count: int) -> None:
    """Set the native CPU thread count used by large kernels."""

    _C.set_num_threads(int(count))


def performance_info() -> PerformanceInfo:
    """Return CPU backend feature information for diagnostics and benchmarks."""

    return cast(PerformanceInfo, _C.performance_info())


__all__ = ["PerformanceInfo", "get_num_threads", "performance_info", "set_num_threads"]
