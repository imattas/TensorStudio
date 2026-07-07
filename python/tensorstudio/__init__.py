"""TensorStudio public Python API."""

from __future__ import annotations

from . import data, nn, optim
from ._version import __version__
from .grad_mode import is_grad_enabled, no_grad, set_grad_enabled
from .ops import astype, avg_pool2d, concat, conv2d, max_pool2d, stack
from .serialization import load, save
from .tensor import (
    Tensor,
    arange,
    empty,
    empty_like,
    eye,
    from_numpy,
    full,
    full_like,
    linspace,
    manual_seed,
    ones,
    ones_like,
    rand,
    rand_like,
    randn,
    randn_like,
    tensor,
    zeros,
    zeros_like,
)

__all__ = [
    "Tensor",
    "__version__",
    "arange",
    "astype",
    "avg_pool2d",
    "concat",
    "conv2d",
    "data",
    "empty",
    "empty_like",
    "eye",
    "from_numpy",
    "full",
    "full_like",
    "is_grad_enabled",
    "linspace",
    "load",
    "manual_seed",
    "max_pool2d",
    "nn",
    "no_grad",
    "ones",
    "ones_like",
    "optim",
    "rand",
    "rand_like",
    "randn",
    "randn_like",
    "save",
    "set_grad_enabled",
    "stack",
    "tensor",
    "zeros",
    "zeros_like",
]
