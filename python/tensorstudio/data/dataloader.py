"""Minimal deterministic dataloader."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from .dataset import Dataset


class DataLoader:
    """Iterate over a dataset in Python batches."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[list[Any]]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            yield [self.dataset[index] for index in indices[start : start + self.batch_size]]


__all__ = ["DataLoader"]
