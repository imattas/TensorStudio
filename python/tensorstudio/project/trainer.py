"""Small reusable training loop utilities."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from tensorstudio.grad_mode import no_grad
from tensorstudio.nn import Module
from tensorstudio.tensor import Tensor


class OptimizerLike(Protocol):
    def zero_grad(self) -> None: ...

    def step(self) -> None: ...


MetricFn = Callable[[Tensor, Tensor], float]
LossFn = Callable[..., Tensor]
Callback = Any


@dataclass(slots=True)
class History:
    """Epoch-level training metrics."""

    records: list[dict[str, float]] = field(default_factory=list)

    @property
    def last(self) -> dict[str, float]:
        if not self.records:
            return {}
        return dict(self.records[-1])

    def append(self, record: dict[str, float]) -> None:
        self.records.append(dict(record))

    def to_dict(self) -> dict[str, list[dict[str, float]]]:
        return {"records": [dict(record) for record in self.records]}


class Trainer:
    """A compact eager trainer for TensorStudio modules.

    The trainer intentionally stays transparent: each batch performs
    ``zero_grad -> forward -> loss -> backward -> step`` and stores epoch-level
    averages in :class:`History`.
    """

    def __init__(
        self,
        model: Module,
        optimizer: OptimizerLike,
        loss_fn: LossFn,
        *,
        metric_fn: MetricFn | None = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.metric_fn = metric_fn

    def fit(
        self,
        loader: Iterable[Any],
        *,
        epochs: int = 1,
        validation_loader: Iterable[Any] | None = None,
        scheduler: Any | None = None,
        callbacks: Sequence[Callback] | None = None,
        start_epoch: int = 1,
    ) -> History:
        if epochs <= 0:
            raise ValueError("epochs must be positive")
        callbacks = callbacks or ()
        history = History()
        self.stop_training = False
        for epoch in range(start_epoch, start_epoch + epochs):
            self.model.train()
            record = self._run_epoch(loader, train=True)
            record["epoch"] = float(epoch)
            if validation_loader is not None:
                validation = self.evaluate(validation_loader)
                record.update({f"val_{key}": value for key, value in validation.items()})
            if scheduler is not None:
                scheduler.step()
            context = {
                "model": self.model,
                "optimizer": self.optimizer,
                "scheduler": scheduler,
                "history": history,
                "record": record,
                "stop_training": False,
            }
            self._run_callbacks(callbacks, context)
            history.append(record)
            self.stop_training = bool(context.get("stop_training")) or any(
                bool(getattr(callback, "stop_training", False)) for callback in callbacks
            )
            if self.stop_training:
                break
        return history

    def evaluate(self, loader: Iterable[Any]) -> dict[str, float]:
        self.model.eval()
        with no_grad():
            return self._run_epoch(loader, train=False)

    def _run_epoch(self, loader: Iterable[Any], *, train: bool) -> dict[str, float]:
        total_loss = 0.0
        total_metric = 0.0
        batches = 0
        for batch in loader:
            inputs, target = self._split_batch(batch)
            if train:
                self.optimizer.zero_grad()
            prediction = self._forward(inputs)
            loss = self._loss(prediction, target)
            if train:
                loss.backward()
                self.optimizer.step()
            total_loss += float(loss.item())
            if target is not None and self.metric_fn is not None:
                total_metric += float(self.metric_fn(prediction, target))
            batches += 1

        if batches == 0:
            raise ValueError("loader produced no batches")

        record = {"loss": total_loss / batches}
        if self.metric_fn is not None:
            record["metric"] = total_metric / batches
        return record

    @staticmethod
    def _run_callbacks(callbacks: Sequence[Callback], context: dict[str, Any]) -> None:
        for callback in callbacks:
            if hasattr(callback, "on_epoch_end"):
                callback.on_epoch_end(context)
            else:
                callback(dict(context["record"]))

    def _forward(self, inputs: Any) -> Tensor:
        output = self.model(*inputs) if isinstance(inputs, tuple) else self.model(inputs)
        if not isinstance(output, Tensor):
            raise TypeError("model forward must return a Tensor")
        return output

    def _loss(self, prediction: Tensor, target: Tensor | None) -> Tensor:
        if target is None:
            return self.loss_fn(prediction)
        return self.loss_fn(prediction, target)

    @staticmethod
    def _split_batch(batch: Any) -> tuple[Any, Tensor | None]:
        if isinstance(batch, tuple) and len(batch) == 2 and isinstance(batch[1], Tensor):
            return batch[0], batch[1]
        return batch, None


__all__ = ["Callback", "History", "LossFn", "MetricFn", "OptimizerLike", "Trainer"]
