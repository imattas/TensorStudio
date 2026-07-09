"""Deterministic seeding helpers."""

from __future__ import annotations

import random

import numpy as np

from tensorstudio.tensor import manual_seed


def seed_everything(seed: int) -> int:
    """Seed Python random, NumPy, and TensorStudio's native RNG."""

    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    manual_seed(seed)
    return seed


__all__ = ["seed_everything"]
