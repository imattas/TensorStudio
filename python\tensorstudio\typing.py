"""Typing helpers for TensorStudio."""

from __future__ import annotations

from os import PathLike
from typing import TypeAlias

ShapeLike: TypeAlias = int | tuple[int, ...] | list[int]
PathLikeStr: TypeAlias = str | PathLike[str]
DTypeLike: TypeAlias = str | None

__all__ = ["DTypeLike", "PathLikeStr", "ShapeLike"]
