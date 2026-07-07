"""Autograd grad-mode helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from . import _C


def is_grad_enabled() -> bool:
    """Return whether autograd graph recording is enabled."""

    return bool(_C._grad_enabled())


def set_grad_enabled(enabled: bool) -> None:
    """Enable or disable autograd graph recording for the current thread."""

    _C._set_grad_enabled(enabled)


@contextmanager
def no_grad() -> Iterator[None]:
    """Temporarily disable autograd graph recording."""

    previous = is_grad_enabled()
    set_grad_enabled(False)
    try:
        yield
    finally:
        set_grad_enabled(previous)


__all__ = ["is_grad_enabled", "no_grad", "set_grad_enabled"]
