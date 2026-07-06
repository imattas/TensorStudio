"""Dataset abstractions."""

from __future__ import annotations

from typing import Protocol

from tensorstudio.tensor import Tensor


class Dataset(Protocol):
    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> object: ...


class TensorDataset:
    """Dataset wrapping tensors with matching leading dimension."""

    def __init__(self, *tensors: Tensor) -> None:
        if not tensors:
            raise ValueError("TensorDataset requires at least one tensor")
        leading = tensors[0].shape[0]
        if any(tensor.shape[0] != leading for tensor in tensors):
            raise ValueError("all tensors must have the same leading dimension")
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    def __getitem__(self, index: int) -> tuple[object, ...]:
        return tuple(tensor.tolist()[index] for tensor in self.tensors)


__all__ = ["Dataset", "TensorDataset"]
