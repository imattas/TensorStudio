"""Dataset splitting helpers."""

from __future__ import annotations

import random

from .dataset import Dataset, Subset


def train_val_split(
    dataset: Dataset,
    val_fraction: float = 0.2,
    *,
    seed: int | None = None,
    shuffle: bool = True,
) -> tuple[Subset, Subset]:
    """Split a dataset into deterministic train and validation subsets."""

    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be between 0 and 1")
    indices = list(range(len(dataset)))
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(indices)
    val_size = max(1, int(round(len(indices) * val_fraction)))
    if val_size >= len(indices):
        val_size = len(indices) - 1
    if val_size <= 0:
        raise ValueError("dataset is too small to split")
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return Subset(dataset, train_indices), Subset(dataset, val_indices)


__all__ = ["train_val_split"]
