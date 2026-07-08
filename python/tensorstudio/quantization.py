"""Quantization research utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .tensor import Tensor, tensor


@dataclass(frozen=True)
class QuantizationConfig:
    """Affine quantization configuration."""

    num_bits: int = 8
    symmetric: bool = False
    dtype: str = "int32"

    def __post_init__(self) -> None:
        if self.num_bits < 2 or self.num_bits > 16:
            raise ValueError("num_bits must be in [2, 16]")
        if self.dtype not in {"int32", "int64"}:
            raise ValueError("quantized storage dtype must be int32 or int64")

    @property
    def qmin(self) -> int:
        return -(2 ** (self.num_bits - 1)) if self.symmetric else 0

    @property
    def qmax(self) -> int:
        return (2 ** (self.num_bits - 1)) - 1 if self.symmetric else (2**self.num_bits) - 1


@dataclass(frozen=True)
class QuantizedTensor:
    """Tensor plus affine quantization metadata."""

    qvalues: Tensor
    scale: float
    zero_point: int
    original_shape: tuple[int, ...]
    original_dtype: str
    config: QuantizationConfig

    def dequantize(self) -> Tensor:
        return dequantize_tensor(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qvalues": self.qvalues.tolist(),
            "scale": self.scale,
            "zero_point": self.zero_point,
            "original_shape": list(self.original_shape),
            "original_dtype": self.original_dtype,
            "num_bits": self.config.num_bits,
            "symmetric": self.config.symmetric,
            "dtype": self.config.dtype,
        }


def quantize_tensor(
    input: Tensor,
    config: QuantizationConfig | None = None,
) -> QuantizedTensor:
    cfg = config or QuantizationConfig()
    data = input.numpy().astype(np.float64)
    if data.size == 0:
        scale = 1.0
        zero_point = 0
        qvalues = np.asarray([], dtype=np.int64).reshape(input.shape)
    elif cfg.symmetric:
        max_abs = float(np.max(np.abs(data)))
        scale = max(max_abs / max(float(cfg.qmax), 1.0), 1e-12)
        zero_point = 0
        qvalues = np.clip(np.round(data / scale), cfg.qmin, cfg.qmax)
    else:
        min_value = float(np.min(data))
        max_value = float(np.max(data))
        scale = max((max_value - min_value) / float(cfg.qmax - cfg.qmin), 1e-12)
        zero_point = int(round(cfg.qmin - min_value / scale))
        zero_point = int(np.clip(zero_point, cfg.qmin, cfg.qmax))
        qvalues = np.clip(np.round(data / scale + zero_point), cfg.qmin, cfg.qmax)
    return QuantizedTensor(
        tensor(qvalues.astype(np.int64).tolist(), dtype=cfg.dtype),
        scale,
        zero_point,
        tuple(int(dim) for dim in input.shape),
        input.dtype,
        cfg,
    )


def dequantize_tensor(input: QuantizedTensor) -> Tensor:
    values = (input.qvalues.numpy().astype(np.float64) - input.zero_point) * input.scale
    return tensor(values.reshape(input.original_shape).tolist(), dtype=input.original_dtype)


def fake_quantize(input: Tensor, config: QuantizationConfig | None = None) -> Tensor:
    return quantize_tensor(input, config).dequantize()


def quantize_state_dict(
    state: dict[str, Tensor],
    config: QuantizationConfig | None = None,
) -> dict[str, QuantizedTensor]:
    return {name: quantize_tensor(value, config) for name, value in state.items()}


def dequantize_state_dict(state: dict[str, QuantizedTensor]) -> dict[str, Tensor]:
    return {name: value.dequantize() for name, value in state.items()}


def quantization_report(state: dict[str, Tensor]) -> dict[str, Any]:
    entries: dict[str, Any] = {}
    total_parameters = 0
    for name, value in state.items():
        total_parameters += value.size
        entries[name] = {
            "shape": value.shape,
            "dtype": value.dtype,
            "parameters": value.size,
            "float32_bytes": value.size * 4,
            "int8_bytes": value.size,
        }
    return {"total_parameters": total_parameters, "entries": entries}


__all__ = [
    "QuantizationConfig",
    "QuantizedTensor",
    "dequantize_state_dict",
    "dequantize_tensor",
    "fake_quantize",
    "quantization_report",
    "quantize_state_dict",
    "quantize_tensor",
]
