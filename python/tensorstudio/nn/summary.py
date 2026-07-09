"""Model summary helpers."""

from __future__ import annotations

from typing import Any

from tensorstudio.grad_mode import no_grad
from tensorstudio.tensor import Tensor, zeros

_DTYPE_SIZES = {
    "bool": 1,
    "int32": 4,
    "int64": 8,
    "float32": 4,
    "float64": 8,
}


def _tensor_bytes(tensor: Tensor) -> int:
    return tensor.size * _DTYPE_SIZES.get(tensor.dtype, 4)


def _module_state(module: Any) -> list[tuple[Any, bool]]:
    modules = module.modules()
    return [(item, item.training) for item in modules]


def _restore_module_state(state: list[tuple[Any, bool]]) -> None:
    for module, was_training in state:
        module.training = was_training


def model_summary(
    module: Any,
    input_shape: tuple[int, ...] | list[int] | None = None,
    dtype: str = "float32",
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    shape_by_name: dict[str, tuple[int, ...]] = {}

    if input_shape is not None:
        from .modules import Sequential

        state = _module_state(module)
        module.eval()
        try:
            with no_grad():
                current = zeros(tuple(input_shape), dtype=dtype)
                if isinstance(module, Sequential):
                    for name, child in module._named_children():
                        input_before = current.shape
                        current = child(current)
                        shape_by_name[name] = current.shape
                        rows.append(
                            {
                                "name": name,
                                "type": child.__class__.__name__,
                                "input_shape": input_before,
                                "output_shape": current.shape,
                                "parameters": child.parameter_count(),
                                "trainable_parameters": child.parameter_count(trainable_only=True),
                            }
                        )
                else:
                    output = module(current)
                    shape_by_name[""] = output.shape
        finally:
            _restore_module_state(state)

    existing = {row["name"] for row in rows}
    for name, child in module.named_modules():
        if name == "" or name in existing:
            continue
        rows.append(
            {
                "name": name,
                "type": child.__class__.__name__,
                "input_shape": None,
                "output_shape": shape_by_name.get(name),
                "parameters": child.parameter_count(),
                "trainable_parameters": child.parameter_count(trainable_only=True),
            }
        )

    parameters = module.parameters()
    buffers = module.buffers() if hasattr(module, "buffers") else []
    total_parameters = sum(parameter.size for parameter in parameters)
    trainable_parameters = sum(
        parameter.size for parameter in parameters if parameter.requires_grad
    )
    parameter_bytes = sum(_tensor_bytes(parameter) for parameter in parameters)
    buffer_bytes = sum(_tensor_bytes(buffer) for buffer in buffers)
    return {
        "module": module.__class__.__name__,
        "rows": rows,
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "buffers": sum(buffer.size for buffer in buffers),
        "estimated_parameter_bytes": parameter_bytes,
        "estimated_buffer_bytes": buffer_bytes,
        "estimated_total_bytes": parameter_bytes + buffer_bytes,
    }


def summary(module: Any, input_shape: tuple[int, ...] | list[int] | None = None) -> dict[str, Any]:
    return model_summary(module, input_shape=input_shape)


__all__ = ["model_summary", "summary"]
