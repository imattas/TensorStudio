"""Checkpoint helpers for TensorStudio modules and optimizers."""

from __future__ import annotations

from typing import Any, Protocol, cast

from tensorstudio.nn import Module
from tensorstudio.serialization import load, load_npz, save, save_npz
from tensorstudio.tensor import Tensor
from tensorstudio.typing import PathLikeStr


class StatefulOptimizer(Protocol):
    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state: dict[str, object]) -> None: ...


def save_state_dict(model: Module, path: PathLikeStr) -> None:
    """Save model weights to TensorStudio's safe non-pickle NPZ format."""

    save_npz(model.state_dict(), path)


def load_state_dict(model: Module, path: PathLikeStr, *, strict: bool = True) -> dict[str, Tensor]:
    """Load model weights from a TensorStudio NPZ state dict."""

    state = load_npz(path)
    if not isinstance(state, dict):
        raise ValueError(
            "state dict checkpoint must contain a mapping of parameter names to tensors"
        )
    model.load_state_dict(state, strict=strict)
    return state


def save_checkpoint(
    model: Module,
    path: PathLikeStr,
    *,
    optimizer: StatefulOptimizer | None = None,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a trusted full checkpoint with pickle.

    Prefer :func:`save_state_dict` for portable weights. Full checkpoints may
    include optimizer internals and arbitrary metadata, so they use pickle and
    must only be loaded from trusted sources.
    """

    checkpoint: dict[str, Any] = {
        "model": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    save(checkpoint, path)


def load_checkpoint(
    model: Module,
    path: PathLikeStr,
    *,
    optimizer: StatefulOptimizer | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a trusted full checkpoint and restore model and optimizer state."""

    checkpoint = load(path)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError("checkpoint must contain a 'model' state_dict")
    model_state = checkpoint["model"]
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint 'model' must be a state_dict mapping")
    model.load_state_dict(cast(dict[str, Tensor], model_state), strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer_state = checkpoint["optimizer"]
        if not isinstance(optimizer_state, dict):
            raise ValueError("checkpoint 'optimizer' must be a state_dict mapping")
        optimizer.load_state_dict(cast(dict[str, object], optimizer_state))
    return cast(dict[str, Any], checkpoint)


__all__ = [
    "StatefulOptimizer",
    "load_checkpoint",
    "load_state_dict",
    "save_checkpoint",
    "save_state_dict",
]
