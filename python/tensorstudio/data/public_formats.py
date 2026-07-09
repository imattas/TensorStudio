"""Datasets for small public data formats."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor, tensor
from tensorstudio.typing import PathLikeStr

from .dataset import Dataset


class CSVDataset(Dataset):
    """CSV dataset with optional header, selected feature columns, and target column."""

    def __init__(
        self,
        path: PathLikeStr,
        *,
        feature_columns: list[str | int] | None = None,
        target_column: str | int | None = None,
        has_header: bool = True,
        dtype: str = "float32",
        target_dtype: str | None = None,
    ) -> None:
        rows, header = _read_csv(path, has_header=has_header)
        if not rows:
            raise ValueError("CSV dataset is empty")
        feature_indices = _resolve_columns(
            feature_columns,
            header,
            len(rows[0]),
            skip=target_column,
        )
        target_index = (
            _resolve_column(target_column, header, len(rows[0]))
            if target_column is not None
            else None
        )
        features = [[float(row[index]) for index in feature_indices] for row in rows]
        self.features = tensor(features, dtype=dtype)
        self.labels = (
            None
            if target_index is None
            else tensor(
                [_parse_scalar(row[target_index]) for row in rows],
                dtype=target_dtype or dtype,
            )
        )
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.header = header

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, Tensor]:
        feature = self.features[index]
        if self.labels is None:
            return feature
        return feature, self.labels[index]


class JSONLDataset(Dataset):
    """Line-delimited JSON dataset with feature and optional label keys."""

    def __init__(
        self,
        path: PathLikeStr,
        *,
        features_key: str = "features",
        label_key: str | None = "label",
        dtype: str = "float32",
        target_dtype: str | None = None,
    ) -> None:
        records = [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not records:
            raise ValueError("JSONL dataset is empty")
        self.features = tensor([record[features_key] for record in records], dtype=dtype)
        self.labels = (
            None
            if label_key is None
            else tensor([record[label_key] for record in records], dtype=target_dtype or dtype)
        )

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> Tensor | tuple[Tensor, Tensor]:
        feature = self.features[index]
        if self.labels is None:
            return feature
        return feature, self.labels[index]


class TextLineDataset(Dataset):
    """Plain text dataset returning one stripped line per item."""

    def __init__(self, path: PathLikeStr, *, skip_empty: bool = True) -> None:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
        self.lines = [line.strip() for line in lines if line.strip() or not skip_empty]

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, index: int) -> str:
        return self.lines[index]


class LibSVMDataset(Dataset):
    """Dense TensorStudio view of small LIBSVM-format classification data."""

    def __init__(
        self,
        path: PathLikeStr,
        *,
        num_features: int | None = None,
        dtype: str = "float32",
    ) -> None:
        labels: list[float] = []
        rows: list[dict[int, float]] = []
        max_index = 0
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label, features = _parse_libsvm_line(line)
            labels.append(label)
            rows.append(features)
            max_index = max(max_index, *(features.keys() or [0]))
        if not rows:
            raise ValueError("LIBSVM dataset is empty")
        feature_count = num_features or max_index
        dense = np.zeros((len(rows), feature_count), dtype=np.float32)
        for row_index, features in enumerate(rows):
            for one_based_index, value in features.items():
                if one_based_index <= 0 or one_based_index > feature_count:
                    raise ValueError("LIBSVM feature index is outside num_features")
                dense[row_index, one_based_index - 1] = value
        self.features = tensor(dense.tolist(), dtype=dtype)
        self.labels = tensor(labels, dtype=dtype)

    def __len__(self) -> int:
        return self.features.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        return self.features[index], self.labels[index]


def from_csv(path: PathLikeStr, **kwargs: Any) -> CSVDataset:
    return CSVDataset(path, **kwargs)


def from_jsonl(path: PathLikeStr, **kwargs: Any) -> JSONLDataset:
    return JSONLDataset(path, **kwargs)


def from_text_lines(path: PathLikeStr, **kwargs: Any) -> TextLineDataset:
    return TextLineDataset(path, **kwargs)


def from_libsvm(path: PathLikeStr, **kwargs: Any) -> LibSVMDataset:
    return LibSVMDataset(path, **kwargs)


def _read_csv(path: PathLikeStr, *, has_header: bool) -> tuple[list[list[str]], list[str] | None]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        rows = list(reader)
    if has_header and rows:
        return rows[1:], rows[0]
    return rows, None


def _resolve_columns(
    columns: list[str | int] | None,
    header: list[str] | None,
    width: int,
    *,
    skip: str | int | None,
) -> list[int]:
    if columns is not None:
        return [_resolve_column(column, header, width) for column in columns]
    skipped = _resolve_column(skip, header, width) if skip is not None else None
    return [index for index in range(width) if index != skipped]


def _resolve_column(column: str | int | None, header: list[str] | None, width: int) -> int:
    if column is None:
        raise ValueError("column cannot be None")
    if isinstance(column, int):
        index = column
    else:
        if header is None:
            raise ValueError("string column names require a CSV header")
        if column not in header:
            raise ValueError(f"unknown column name: {column}")
        index = header.index(column)
    if index < 0:
        index += width
    if index < 0 or index >= width:
        raise ValueError("column index is out of range")
    return index


def _parse_scalar(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        return float(value)


def _parse_libsvm_line(line: str) -> tuple[float, dict[int, float]]:
    parts = line.split()
    label = float(parts[0])
    features: dict[int, float] = {}
    for part in parts[1:]:
        index, value = part.split(":", 1)
        features[int(index)] = float(value)
    return label, features


__all__ = [
    "CSVDataset",
    "JSONLDataset",
    "LibSVMDataset",
    "TextLineDataset",
    "from_csv",
    "from_jsonl",
    "from_libsvm",
    "from_text_lines",
]
