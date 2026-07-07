"""Regression metrics."""

from __future__ import annotations

import numpy as np

from tensorstudio.tensor import Tensor


def _array(value: Tensor) -> np.ndarray:
    return value.numpy().astype(np.float64)


def mean_absolute_error(prediction: Tensor, target: Tensor) -> float:
    return float(np.mean(np.abs(_array(prediction) - _array(target))))


def mean_squared_error(prediction: Tensor, target: Tensor) -> float:
    diff = _array(prediction) - _array(target)
    return float(np.mean(diff * diff))


def root_mean_squared_error(prediction: Tensor, target: Tensor) -> float:
    return float(np.sqrt(mean_squared_error(prediction, target)))


def r2_score(prediction: Tensor, target: Tensor) -> float:
    y_true = _array(target)
    residual = y_true - _array(prediction)
    ss_res = float(np.sum(residual * residual))
    centered = y_true - float(np.mean(y_true))
    ss_tot = float(np.sum(centered * centered))
    return 1.0 if ss_tot == 0.0 and ss_res == 0.0 else 1.0 - ss_res / ss_tot


__all__ = [
    "mean_absolute_error",
    "mean_squared_error",
    "r2_score",
    "root_mean_squared_error",
]
