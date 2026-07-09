"""Classification metrics."""

from __future__ import annotations

import numpy as np

from tensorstudio.tensor import Tensor


def _labels(prediction: Tensor) -> np.ndarray:
    values = prediction.numpy()
    if values.ndim == 1:
        return (values >= 0.5).astype(np.int64)
    return np.argmax(values, axis=1).astype(np.int64)


def _target(target: Tensor) -> np.ndarray:
    return target.numpy().astype(np.int64).reshape(-1)


def accuracy(prediction: Tensor, target: Tensor) -> float:
    pred = _labels(prediction)
    true = _target(target)
    if pred.shape[0] != true.shape[0]:
        raise ValueError("prediction and target batch sizes must match")
    return float(np.mean(pred == true))


def confusion_matrix(
    prediction: Tensor,
    target: Tensor,
    num_classes: int | None = None,
) -> np.ndarray:
    pred = _labels(prediction)
    true = _target(target)
    if pred.shape[0] != true.shape[0]:
        raise ValueError("prediction and target batch sizes must match")
    classes = (
        int(max(pred.max(initial=0), true.max(initial=0)) + 1)
        if num_classes is None
        else num_classes
    )
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for expected, actual in zip(true, pred, strict=True):
        if expected < 0 or expected >= classes or actual < 0 or actual >= classes:
            raise ValueError("class index out of range")
        matrix[expected, actual] += 1
    return matrix


def precision_recall_f1(
    prediction: Tensor,
    target: Tensor,
    num_classes: int | None = None,
) -> dict[str, float]:
    matrix = confusion_matrix(prediction, target, num_classes=num_classes).astype(np.float64)
    tp = np.diag(matrix)
    predicted_totals = matrix.sum(axis=0)
    expected_totals = matrix.sum(axis=1)
    precision = np.divide(
        tp,
        predicted_totals,
        out=np.zeros_like(tp),
        where=predicted_totals != 0,
    )
    recall = np.divide(
        tp,
        expected_totals,
        out=np.zeros_like(tp),
        where=expected_totals != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )
    return {
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
    }


__all__ = ["accuracy", "confusion_matrix", "precision_recall_f1"]
