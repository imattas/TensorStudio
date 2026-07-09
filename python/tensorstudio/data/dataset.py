"""Dataset abstractions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor, tensor


class Dataset:
    """Base class for map-style datasets."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, index: int) -> object:
        raise NotImplementedError


class TensorDataset(Dataset):
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


class ArrayDataset(Dataset):
    """Dataset built from NumPy-like feature arrays and optional labels."""

    def __init__(self, features: Any, labels: Any | None = None, dtype: str = "float32") -> None:
        features_array = np.asarray(features)
        if features_array.ndim == 0:
            raise ValueError("features must have at least one dimension")
        self.features = tensor(features_array.tolist(), dtype=dtype)
        self.labels = None if labels is None else tensor(np.asarray(labels).tolist())
        if self.labels is not None and self.labels.shape[0] != self.features.shape[0]:
            raise ValueError("labels must have the same leading dimension as features")

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> object:
        feature = self.features[index]
        if self.labels is None:
            return feature
        return feature, self.labels[index]


class Subset(Dataset):
    """Dataset view over selected indices."""

    def __init__(self, dataset: Dataset, indices: Sequence[int]) -> None:
        self.dataset = dataset
        self.indices = [int(index) for index in indices]
        if any(index < 0 or index >= len(dataset) for index in self.indices):
            raise ValueError("subset indices must be within dataset bounds")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> object:
        return self.dataset[self.indices[index]]


def from_arrays(features: Any, labels: Any | None = None, dtype: str = "float32") -> ArrayDataset:
    return ArrayDataset(features, labels=labels, dtype=dtype)


def from_tensors(*tensors: Tensor) -> TensorDataset:
    return TensorDataset(*tensors)


def dataset_summary(dataset: Dataset) -> dict[str, Any]:
    length = len(dataset)
    sample = dataset[0] if length else None
    summary: dict[str, Any] = {
        "type": dataset.__class__.__name__,
        "length": length,
    }
    if isinstance(sample, tuple):
        summary["sample_parts"] = len(sample)
        summary["sample_shapes"] = [getattr(part, "shape", None) for part in sample]
    elif sample is not None:
        summary["sample_shape"] = getattr(sample, "shape", None)
    if hasattr(dataset, "classes"):
        summary["classes"] = list(dataset.classes)
    if hasattr(dataset, "targets"):
        targets = list(dataset.targets)
        summary["target_count"] = len(targets)
        summary["target_classes"] = sorted(set(targets))
    return summary


__all__ = [
    "ArrayDataset",
    "Dataset",
    "Subset",
    "TensorDataset",
    "dataset_summary",
    "from_arrays",
    "from_tensors",
]
