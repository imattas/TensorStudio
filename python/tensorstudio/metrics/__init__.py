"""Metrics for regression, classification, and multilabel tasks."""

from __future__ import annotations

from .classification import accuracy, confusion_matrix, precision_recall_f1
from .multilabel import hamming_loss, multilabel_accuracy, multilabel_f1
from .regression import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)

__all__ = [
    "accuracy",
    "confusion_matrix",
    "hamming_loss",
    "mean_absolute_error",
    "mean_squared_error",
    "multilabel_accuracy",
    "multilabel_f1",
    "precision_recall_f1",
    "r2_score",
    "root_mean_squared_error",
]
