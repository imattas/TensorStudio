"""Training callbacks."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from tensorstudio.typing import PathLikeStr

from .checkpoint import save_checkpoint


class CallbackBase:
    """Base class for trainer callbacks."""

    def __init__(self) -> None:
        self.stop_training = False

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        pass


class LrLogger(CallbackBase):
    """Append the optimizer learning rate to each epoch record."""

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        optimizer = context.get("optimizer")
        record = context["record"]
        if optimizer is not None and hasattr(optimizer, "lr"):
            record["lr"] = float(optimizer.lr)


class CSVLogger(CallbackBase):
    """Write epoch records to a CSV file."""

    def __init__(self, path: PathLikeStr) -> None:
        super().__init__()
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fieldnames: list[str] | None = None

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        record = dict(context["record"])
        if self._fieldnames is None:
            self._fieldnames = list(record)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                writer.writerow(record)
            return
        missing = [key for key in record if key not in self._fieldnames]
        if missing:
            rows: list[dict[str, str]] = []
            if self.path.exists():
                with self.path.open("r", newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
            self._fieldnames.extend(missing)
            with self.path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({key: row.get(key, "") for key in self._fieldnames})
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=self._fieldnames)
            writer.writerow({key: record.get(key, "") for key in self._fieldnames})


class EarlyStopping(CallbackBase):
    """Stop training when a monitored value stops improving."""

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 5,
        min_delta: float = 0.0,
        mode: str = "min",
    ) -> None:
        super().__init__()
        if patience < 0:
            raise ValueError("patience must be non-negative")
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best: float | None = None
        self.bad_epochs = 0

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        record = context["record"]
        if self.monitor not in record:
            return
        value = float(record[self.monitor])
        improved = (
            self.best is None
            or (self.mode == "min" and value < self.best - self.min_delta)
            or (self.mode == "max" and value > self.best + self.min_delta)
        )
        if improved:
            self.best = value
            self.bad_epochs = 0
            return
        self.bad_epochs += 1
        if self.bad_epochs > self.patience:
            self.stop_training = True
            context["stop_training"] = True


class CheckpointCallback(CallbackBase):
    """Save trusted full checkpoints at epoch end."""

    def __init__(
        self,
        path: PathLikeStr,
        *,
        monitor: str = "val_loss",
        mode: str = "min",
        save_best_only: bool = False,
    ) -> None:
        super().__init__()
        if mode not in {"min", "max"}:
            raise ValueError("mode must be 'min' or 'max'")
        self.path = Path(path)
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best: float | None = None

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        record = context["record"]
        value = record.get(self.monitor)
        should_save = not self.save_best_only
        if value is not None:
            current = float(value)
            improved = (
                self.best is None
                or (self.mode == "min" and current < self.best)
                or (self.mode == "max" and current > self.best)
            )
            if improved:
                self.best = current
                should_save = True
        if not should_save:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            context["model"],
            self.path,
            optimizer=context.get("optimizer"),
            scheduler=context.get("scheduler"),
            epoch=int(record.get("epoch", 0)),
            metadata={"record": dict(record)},
        )


__all__ = ["CSVLogger", "CallbackBase", "CheckpointCallback", "EarlyStopping", "LrLogger"]
