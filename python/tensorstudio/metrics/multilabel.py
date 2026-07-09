"""Multilabel metrics."""

from __future__ import annotations

import numpy as np

from tensorstudio.tensor import Tensor


def _binary(prediction: Tensor, threshold: float) -> np.ndarray:
    return (prediction.numpy() >= threshold).astype(np.int64)


def _target(target: Tensor) -> np.ndarray:
    return target.numpy().astype(np.int64)


def multilabel_accuracy(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    true = _target(target)
    if pred.shape != true.shape:
        raise ValueError("prediction and target shapes must match")
    return float(np.mean(np.all(pred == true, axis=1)))


def hamming_loss(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    true = _target(target)
    if pred.shape != true.shape:
        raise ValueError("prediction and target shapes must match")
    return float(np.mean(pred != true))


def multilabel_f1(prediction: Tensor, target: Tensor, threshold: float = 0.5) -> float:
    pred = _binary(prediction, threshold)
    true = _target(target)
    if pred.shape != true.shape:
        raise ValueError("prediction and target shapes must match")
    tp = float(np.sum((pred == 1) & (true == 1)))
    fp = float(np.sum((pred == 1) & (true == 0)))
    fn = float(np.sum((pred == 0) & (true == 1)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0.0 else 2 * tp / denom


__all__ = ["hamming_loss", "multilabel_accuracy", "multilabel_f1"]
