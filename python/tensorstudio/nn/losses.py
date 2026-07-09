"""Loss modules."""

from __future__ import annotations

from tensorstudio.tensor import Tensor

from . import functional as F
from .modules import Module


class MSELoss(Module):
    """Mean squared error loss."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target)


class L1Loss(Module):
    """Mean absolute error loss."""

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.l1_loss(input, target)


class BCELoss(Module):
    """Binary cross entropy over probabilities."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy(input, target, eps=self.eps)


class BCEWithLogitsLoss(Module):
    """Binary cross entropy over logits."""

    def __init__(self, eps: float = 1e-7) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.binary_cross_entropy_with_logits(input, target, eps=self.eps)


class CrossEntropyLoss(Module):
    """Multiclass cross entropy over unnormalized logits."""

    def __init__(self, reduction: str = "mean", label_smoothing: float = 0.0) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must satisfy 0 <= label_smoothing < 1")
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.cross_entropy(
            input,
            target,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )


class NLLLoss(Module):
    """Negative log likelihood over log-probabilities."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.nll_loss(input, target, reduction=self.reduction)


class KLDivLoss(Module):
    """Kullback-Leibler divergence loss for log-probability inputs."""

    def __init__(self, reduction: str = "mean", log_target: bool = False) -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "batchmean", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', 'batchmean', or 'none'")
        self.reduction = reduction
        self.log_target = log_target

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)


class FocalLoss(Module):
    """Multiclass focal loss over unnormalized logits."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float | None = None,
        reduction: str = "mean",
    ) -> None:
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if alpha is not None and not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.focal_loss(
            input,
            target,
            gamma=self.gamma,
            alpha=self.alpha,
            reduction=self.reduction,
        )


class CosineEmbeddingLoss(Module):
    """Cosine embedding loss for similar or dissimilar pairs."""

    def __init__(self, margin: float = 0.0, reduction: str = "mean") -> None:
        super().__init__()
        if reduction not in {"mean", "sum", "none"}:
            raise ValueError("reduction must be 'mean', 'sum', or 'none'")
        self.margin = margin
        self.reduction = reduction

    def forward(self, input1: Tensor, input2: Tensor, target: Tensor) -> Tensor:
        return F.cosine_embedding_loss(
            input1,
            input2,
            target,
            margin=self.margin,
            reduction=self.reduction,
        )


class HuberLoss(Module):
    """Huber loss with configurable transition point."""

    def __init__(self, delta: float = 1.0) -> None:
        super().__init__()
        if delta <= 0:
            raise ValueError("delta must be positive")
        self.delta = delta

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.huber_loss(input, target, delta=self.delta)


__all__ = [
    "BCELoss",
    "BCEWithLogitsLoss",
    "CosineEmbeddingLoss",
    "CrossEntropyLoss",
    "FocalLoss",
    "HuberLoss",
    "KLDivLoss",
    "L1Loss",
    "MSELoss",
    "NLLLoss",
]
