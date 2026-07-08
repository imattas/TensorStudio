"""Distributed-training research helpers.

TensorStudio does not ship a production distributed runtime. This module
provides explicit single-process collectives, environment parsing, and planning
metadata so experiments can be structured without fake multi-node execution.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .tensor import Tensor


@dataclass(frozen=True)
class DistributedConfig:
    """Minimal distributed execution descriptor."""

    backend: str = "single_process"
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0

    def __post_init__(self) -> None:
        if self.world_size <= 0:
            raise ValueError("world_size must be positive")
        if self.rank < 0 or self.rank >= self.world_size:
            raise ValueError("rank must satisfy 0 <= rank < world_size")
        if self.local_rank < 0:
            raise ValueError("local_rank must be non-negative")
        if self.backend not in {"single_process", "research"}:
            raise ValueError("backend must be 'single_process' or 'research'")

    @property
    def is_distributed(self) -> bool:
        return self.world_size > 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "world_size": self.world_size,
            "rank": self.rank,
            "local_rank": self.local_rank,
            "is_distributed": self.is_distributed,
        }


def config_from_env() -> DistributedConfig:
    """Read common launcher environment variables into a config object."""

    return DistributedConfig(
        backend=os.environ.get("TENSORSTUDIO_DISTRIBUTED_BACKEND", "single_process"),
        world_size=int(os.environ.get("WORLD_SIZE", "1")),
        rank=int(os.environ.get("RANK", "0")),
        local_rank=int(os.environ.get("LOCAL_RANK", "0")),
    )


def distributed_info(config: DistributedConfig | None = None) -> dict[str, Any]:
    return (config or config_from_env()).to_dict()


def all_reduce_sum(tensor: Tensor, config: DistributedConfig | None = None) -> Tensor:
    cfg = config or config_from_env()
    if cfg.world_size == 1:
        return tensor.clone()
    raise NotImplementedError(
        "multi-process all_reduce_sum is research-only in TensorStudio 1.16.0"
    )


def average_gradients(parameters: list[Tensor], config: DistributedConfig | None = None) -> None:
    cfg = config or config_from_env()
    if cfg.world_size == 1:
        return
    for parameter in parameters:
        if parameter.grad is not None:
            reduced = all_reduce_sum(parameter.grad, cfg) / float(cfg.world_size)
            parameter.grad._assign(reduced)


def data_parallel_plan(
    dataset_size: int,
    batch_size: int,
    config: DistributedConfig | None = None,
) -> dict[str, Any]:
    """Return deterministic per-rank batch planning metadata."""

    cfg = config or config_from_env()
    if dataset_size < 0:
        raise ValueError("dataset_size must be non-negative")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    samples_per_rank = [dataset_size // cfg.world_size] * cfg.world_size
    for index in range(dataset_size % cfg.world_size):
        samples_per_rank[index] += 1
    return {
        "backend": cfg.backend,
        "world_size": cfg.world_size,
        "batch_size": batch_size,
        "global_batch_size": batch_size * cfg.world_size,
        "rank": cfg.rank,
        "rank_samples": samples_per_rank[cfg.rank],
        "rank_batches": (samples_per_rank[cfg.rank] + batch_size - 1) // batch_size,
        "samples_per_rank": samples_per_rank,
    }


__all__ = [
    "DistributedConfig",
    "all_reduce_sum",
    "average_gradients",
    "config_from_env",
    "data_parallel_plan",
    "distributed_info",
]
