"""Minimal deterministic dataloader."""

from __future__ import annotations

import random
from collections.abc import Iterator
from typing import Any

from tensorstudio.tensor import Tensor, tensor

from .dataset import Dataset


class DataLoader:
    """Iterate over a dataset in Python batches."""

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        seed: int | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed

    def __iter__(self) -> Iterator[Any]:
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue
            yield self._collate([self.dataset[index] for index in batch_indices])

    @classmethod
    def _collate(cls, batch: list[Any]) -> Any:
        if not batch:
            return batch
        first = batch[0]
        if isinstance(first, tuple):
            columns = list(zip(*batch, strict=True))
            return tuple(cls._collate(list(column)) for column in columns)
        if isinstance(first, Tensor):
            return tensor([item.tolist() for item in batch], dtype=first.dtype)
        if isinstance(first, (int, float, bool, list)):
            return tensor(batch)
        return batch


__all__ = ["DataLoader"]
