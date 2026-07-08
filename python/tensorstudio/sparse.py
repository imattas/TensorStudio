"""Experimental sparse tensor utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tensor import Tensor, from_numpy, tensor


@dataclass(frozen=True)
class SparseCOOTensor:
    """Coordinate-format sparse tensor.

    The v1.16 sparse surface is intentionally small: COO indices, 1D values,
    dense conversion, duplicate coalescing, and 2D sparse-dense matmul. Sparse
    tensors do not yet participate in TensorStudio autograd graphs.
    """

    indices: Tensor
    values: Tensor
    shape: tuple[int, ...]
    is_coalesced: bool = False

    def __post_init__(self) -> None:
        if self.indices.dtype not in {"int32", "int64"}:
            raise ValueError("sparse COO indices must have int32 or int64 dtype")
        if self.indices.ndim != 2:
            raise ValueError("sparse COO indices must have shape (nnz, ndim)")
        if self.values.ndim != 1:
            raise ValueError("sparse COO values must be a 1D tensor")
        if self.indices.shape[0] != self.values.shape[0]:
            raise ValueError("sparse COO indices and values must have the same nnz")
        if self.indices.shape[1] != len(self.shape):
            raise ValueError("sparse COO index rank must match sparse tensor shape")
        if any(dim < 0 for dim in self.shape):
            raise ValueError("sparse COO shape dimensions must be non-negative")
        self._validate_bounds()

    @property
    def nnz(self) -> int:
        return int(self.values.shape[0])

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def dtype(self) -> str:
        return self.values.dtype

    @property
    def density(self) -> float:
        total = int(np.prod(self.shape, dtype=np.int64)) if self.shape else 1
        return 0.0 if total == 0 else float(self.nnz) / float(total)

    def to_dense(self) -> Tensor:
        dense = np.zeros(self.shape, dtype=_numpy_dtype(self.values.dtype))
        index_array = np.asarray(self.indices.tolist(), dtype=np.int64)
        value_array = self.values.numpy()
        for row, value in zip(index_array, value_array, strict=True):
            dense[tuple(int(dim) for dim in row)] += value
        return tensor(dense.tolist(), dtype=self.values.dtype)

    def coalesce(self) -> SparseCOOTensor:
        if self.is_coalesced:
            return self
        accumulator: dict[tuple[int, ...], float] = {}
        for raw_index, raw_value in zip(self.indices.tolist(), self.values.tolist(), strict=True):
            key = tuple(int(dim) for dim in raw_index)
            accumulator[key] = accumulator.get(key, 0.0) + float(raw_value)
        ordered = sorted(accumulator.items())
        indices = [list(index) for index, _ in ordered]
        values = [value for _, value in ordered]
        return sparse_coo_tensor(
            indices,
            values,
            self.shape,
            dtype=self.values.dtype,
            coalesced=True,
        )

    def matmul(self, dense: Tensor) -> Tensor:
        if self.ndim != 2:
            raise ValueError("sparse matmul currently supports rank-2 sparse tensors")
        if dense.ndim not in {1, 2}:
            raise ValueError("sparse matmul expects a dense vector or matrix")
        if self.shape[1] != dense.shape[0]:
            raise ValueError(
                f"sparse matmul shape mismatch: sparse {self.shape}, dense {dense.shape}"
            )

        values = self.coalesce()
        rows = values.indices.numpy()[:, 0].astype(np.int64)
        cols = values.indices.numpy()[:, 1].astype(np.int64)
        sparse_values = values.values.numpy()
        dense_array = dense.numpy()
        if dense.ndim == 1:
            output = np.zeros((self.shape[0],), dtype=_numpy_dtype(self.values.dtype))
            for row, col, value in zip(rows, cols, sparse_values, strict=True):
                output[row] += value * dense_array[col]
        else:
            output = np.zeros(
                (self.shape[0], dense.shape[1]),
                dtype=_numpy_dtype(self.values.dtype),
            )
            for row, col, value in zip(rows, cols, sparse_values, strict=True):
                output[row, :] += value * dense_array[col, :]
        return tensor(output.tolist(), dtype=self.values.dtype)

    def __matmul__(self, dense: Tensor) -> Tensor:
        return self.matmul(dense)

    def transpose(self) -> SparseCOOTensor:
        if self.ndim != 2:
            raise ValueError("sparse transpose currently supports rank-2 sparse tensors")
        index_array = self.indices.numpy().copy()
        index_array[:, [0, 1]] = index_array[:, [1, 0]]
        return SparseCOOTensor(
            from_numpy(index_array.astype(np.int64)),
            self.values,
            (self.shape[1], self.shape[0]),
            is_coalesced=False,
        )

    @property
    def T(self) -> SparseCOOTensor:
        return self.transpose()

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "tensorstudio.sparse_coo",
            "shape": list(self.shape),
            "dtype": self.dtype,
            "indices": self.indices.tolist(),
            "values": self.values.tolist(),
            "is_coalesced": self.is_coalesced,
        }

    def _validate_bounds(self) -> None:
        for raw_index in self.indices.tolist():
            for axis, value in enumerate(raw_index):
                index = int(value)
                if index < 0 or index >= self.shape[axis]:
                    raise ValueError(
                        "sparse COO index "
                        f"{tuple(raw_index)} is out of bounds for shape {self.shape}"
                    )


def sparse_coo_tensor(
    indices: Tensor | Any,
    values: Tensor | Any,
    shape: tuple[int, ...] | list[int],
    *,
    dtype: str | None = None,
    coalesced: bool = False,
) -> SparseCOOTensor:
    index_tensor = indices if isinstance(indices, Tensor) else tensor(indices, dtype="int64")
    value_tensor = (
        values if isinstance(values, Tensor) else tensor(values, dtype=dtype or "float32")
    )
    if dtype is not None and value_tensor.dtype != dtype:
        value_tensor = tensor(value_tensor.tolist(), dtype=dtype)
    return SparseCOOTensor(index_tensor, value_tensor, tuple(int(dim) for dim in shape), coalesced)


def sparse_from_dense(input: Tensor, *, threshold: float = 0.0) -> SparseCOOTensor:
    values = input.numpy()
    mask = np.abs(values) > threshold
    indices = np.argwhere(mask).astype(np.int64)
    sparse_values = values[mask].astype(_numpy_dtype(input.dtype), copy=False)
    return SparseCOOTensor(
        from_numpy(indices),
        from_numpy(np.asarray(sparse_values)),
        tuple(int(dim) for dim in input.shape),
        is_coalesced=True,
    )


def sparse_mm(sparse: SparseCOOTensor, dense: Tensor) -> Tensor:
    return sparse.matmul(dense)


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    mapping = {
        "float32": np.dtype("float32"),
        "float64": np.dtype("float64"),
        "int32": np.dtype("int32"),
        "int64": np.dtype("int64"),
        "bool": np.dtype("bool"),
    }
    return mapping.get(dtype, np.dtype("float32"))


__all__ = [
    "SparseCOOTensor",
    "sparse_coo_tensor",
    "sparse_from_dense",
    "sparse_mm",
]
