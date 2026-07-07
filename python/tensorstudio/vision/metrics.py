"""Computer-vision metrics."""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor


def _array(value: Tensor | np.ndarray | list[float] | list[int]) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.numpy()
    return np.asarray(value)


def accuracy(logits: Tensor | np.ndarray, target: Tensor | np.ndarray | list[int]) -> float:
    """Top-1 classification accuracy for shape ``(batch, classes)`` logits."""

    predictions = _array(logits).argmax(axis=1)
    labels: np.ndarray[Any, Any] = _array(target).astype(np.int64).reshape(-1)
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError("logits and target must have matching batch size")
    return float((predictions == labels).mean())


def top_k_accuracy(
    logits: Tensor | np.ndarray,
    target: Tensor | np.ndarray | list[int],
    k: int | tuple[int, ...] = 5,
) -> float | tuple[float, ...]:
    """Top-k classification accuracy."""

    scores = _array(logits)
    if scores.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")
    labels: np.ndarray[Any, Any] = _array(target).astype(np.int64).reshape(-1)
    if scores.shape[0] != labels.shape[0]:
        raise ValueError("logits and target must have matching batch size")
    values = (k,) if isinstance(k, int) else tuple(k)
    if any(value <= 0 for value in values):
        raise ValueError("k values must be positive")
    order = np.argsort(scores, axis=1)[:, ::-1]
    results: list[float] = []
    for value in values:
        clipped = min(value, scores.shape[1])
        hits = [label in row[:clipped] for label, row in zip(labels, order, strict=True)]
        results.append(float(np.mean(hits)))
    if isinstance(k, int):
        return results[0]
    return tuple(results)


def confusion_matrix(
    prediction: Tensor | np.ndarray | list[int],
    target: Tensor | np.ndarray | list[int],
    num_classes: int | None = None,
    normalize: bool = False,
) -> np.ndarray:
    """Build a confusion matrix with rows as targets and columns as predictions."""

    predicted = _array(prediction)
    if predicted.ndim == 2:
        predicted = predicted.argmax(axis=1)
    predicted = predicted.astype(np.int64).reshape(-1)
    labels: np.ndarray[Any, Any] = _array(target).astype(np.int64).reshape(-1)
    if predicted.shape[0] != labels.shape[0]:
        raise ValueError("prediction and target must have matching length")
    if num_classes is None:
        num_classes = int(max(predicted.max(initial=0), labels.max(initial=0))) + 1
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    matrix: np.ndarray[Any, Any] = np.zeros(
        (num_classes, num_classes),
        dtype=np.float64 if normalize else np.int64,
    )
    for label, pred in zip(labels, predicted, strict=True):
        if label < 0 or label >= num_classes or pred < 0 or pred >= num_classes:
            raise ValueError("prediction and target values must be within num_classes")
        matrix[label, pred] += 1
    if normalize:
        row_sums = matrix.sum(axis=1, keepdims=True)
        np.divide(matrix, row_sums, out=matrix, where=row_sums != 0)
    return matrix


def box_iou(boxes1: Tensor | np.ndarray, boxes2: Tensor | np.ndarray) -> np.ndarray:
    """Pairwise IoU for ``xyxy`` boxes with shapes ``(N, 4)`` and ``(M, 4)``."""

    first: np.ndarray[Any, Any] = _array(boxes1).astype(np.float64)
    second: np.ndarray[Any, Any] = _array(boxes2).astype(np.float64)
    if first.ndim != 2 or second.ndim != 2 or first.shape[1] != 4 or second.shape[1] != 4:
        raise ValueError("boxes must have shapes (N, 4) and (M, 4)")
    first_area = np.clip(first[:, 2] - first[:, 0], 0, None) * np.clip(
        first[:, 3] - first[:, 1],
        0,
        None,
    )
    second_area = np.clip(second[:, 2] - second[:, 0], 0, None) * np.clip(
        second[:, 3] - second[:, 1],
        0,
        None,
    )
    top_left = np.maximum(first[:, None, :2], second[None, :, :2])
    bottom_right = np.minimum(first[:, None, 2:], second[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    union = first_area[:, None] + second_area[None, :] - intersection
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)


__all__ = ["accuracy", "box_iou", "confusion_matrix", "top_k_accuracy"]
