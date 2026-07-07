"""Limited ONNX export for TensorStudio module graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from tensorstudio.nn import (
    AvgPool2d,
    Conv2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from tensorstudio.typing import PathLikeStr

_DTYPE_TO_ONNX_NAME = {
    "float32": "FLOAT",
    "float64": "DOUBLE",
    "int32": "INT32",
    "int64": "INT64",
    "bool": "BOOL",
}


def _require_onnx() -> tuple[Any, Any, Any]:
    try:
        from onnx import TensorProto, helper, numpy_helper
    except ImportError as exc:  # pragma: no cover - exercised when optional dep is absent
        raise ImportError(
            "ONNX export requires the optional dependency: "
            "python -m pip install 'tensorstudio[onnx]'"
        ) from exc
    return TensorProto, helper, numpy_helper


def _onnx_dtype(dtype: str) -> int:
    TensorProto, _, _ = _require_onnx()
    name = _DTYPE_TO_ONNX_NAME.get(dtype)
    if name is None:
        raise ValueError(f"unsupported ONNX export dtype: {dtype!r}")
    return int(getattr(TensorProto, name))


def _module_sequence(model: Module) -> list[Module]:
    if isinstance(model, Sequential):
        return list(model)
    nested = getattr(model, "model", None)
    if isinstance(nested, Sequential):
        return list(nested)
    layers = getattr(model, "layers", None)
    if isinstance(layers, Iterable) and not isinstance(layers, (str, bytes)):
        return list(layers)
    return [model]


def _tensor_initializer(name: str, value: Any, numpy_helper: Any) -> Any:
    return numpy_helper.from_array(np.asarray(value.numpy()), name=name)


def _conv_output_shape(
    shape: tuple[int, ...],
    out_channels: int,
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int],
) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("Conv2d ONNX export expects input shape (N, C, H, W)")
    batch, _, height, width = shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    effective_h = dilation_h * (kernel_h - 1) + 1
    effective_w = dilation_w * (kernel_w - 1) + 1
    out_h = (height + 2 * padding_h - effective_h) // stride_h + 1
    out_w = (width + 2 * padding_w - effective_w) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Conv2d ONNX export produced a non-positive spatial shape")
    return (batch, out_channels, out_h, out_w)


def _pool_output_shape(
    shape: tuple[int, ...],
    kernel_size: tuple[int, int],
    stride: tuple[int, int],
    padding: tuple[int, int],
    dilation: tuple[int, int] = (1, 1),
) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise ValueError("Pool2d ONNX export expects input shape (N, C, H, W)")
    batch, channels, height, width = shape
    kernel_h, kernel_w = kernel_size
    stride_h, stride_w = stride
    padding_h, padding_w = padding
    dilation_h, dilation_w = dilation
    effective_h = dilation_h * (kernel_h - 1) + 1
    effective_w = dilation_w * (kernel_w - 1) + 1
    out_h = (height + 2 * padding_h - effective_h) // stride_h + 1
    out_w = (width + 2 * padding_w - effective_w) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("Pool2d ONNX export produced a non-positive spatial shape")
    return (batch, channels, out_h, out_w)


def _flatten_shape(shape: tuple[int, ...], axis: int) -> tuple[int, ...]:
    ndim = len(shape)
    normalized = axis if axis >= 0 else ndim + axis
    if normalized < 0 or normalized > ndim:
        raise ValueError("Flatten ONNX export axis is out of range")
    prefix = shape[:normalized]
    flattened = math.prod(shape[normalized:]) if normalized < ndim else 1
    return (*prefix, flattened)


def export_onnx(
    model: Module,
    path: PathLikeStr,
    input_shape: tuple[int, ...] | list[int],
    input_name: str = "input",
    output_name: str = "output",
    input_dtype: str = "float32",
    opset: int = 17,
) -> Path:
    """Export a supported TensorStudio module stack to ONNX.

    Supported modules are ``Linear``, ``Conv2d``, ``Flatten``, ``ReLU``,
    ``Sigmoid``, ``Tanh``, ``MaxPool2d``, and ``AvgPool2d``. Custom Python
    control flow is intentionally not traced in v1.2.
    """

    _, helper, numpy_helper = _require_onnx()
    shape = tuple(int(dim) for dim in input_shape)
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("input_shape must contain positive dimensions")

    nodes: list[Any] = []
    initializers: list[Any] = []
    current_name = input_name
    current_shape = shape

    for index, layer in enumerate(_module_sequence(model)):
        next_name = f"node_{index}_out"
        if isinstance(layer, Linear):
            if len(current_shape) != 2:
                raise ValueError("Linear ONNX export expects a rank-2 input")
            weight_name = f"linear_{index}_weight"
            inputs = [current_name, weight_name]
            initializers.append(_tensor_initializer(weight_name, layer.weight, numpy_helper))
            if layer.bias is not None:
                bias_name = f"linear_{index}_bias"
                inputs.append(bias_name)
                initializers.append(_tensor_initializer(bias_name, layer.bias, numpy_helper))
            nodes.append(
                helper.make_node(
                    "Gemm",
                    inputs,
                    [next_name],
                    name=f"Linear_{index}",
                    transB=1,
                )
            )
            current_shape = (current_shape[0], layer.out_features)
        elif isinstance(layer, Conv2d):
            weight_name = f"conv_{index}_weight"
            inputs = [current_name, weight_name]
            initializers.append(_tensor_initializer(weight_name, layer.weight, numpy_helper))
            if layer.bias is not None:
                bias_name = f"conv_{index}_bias"
                inputs.append(bias_name)
                initializers.append(_tensor_initializer(bias_name, layer.bias, numpy_helper))
            pads = [layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]]
            nodes.append(
                helper.make_node(
                    "Conv",
                    inputs,
                    [next_name],
                    name=f"Conv2d_{index}",
                    kernel_shape=list(layer.kernel_size),
                    strides=list(layer.stride),
                    pads=pads,
                    dilations=list(layer.dilation),
                )
            )
            current_shape = _conv_output_shape(
                current_shape,
                layer.out_channels,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
            )
        elif isinstance(layer, Flatten):
            nodes.append(
                helper.make_node(
                    "Flatten",
                    [current_name],
                    [next_name],
                    name=f"Flatten_{index}",
                    axis=layer.start_dim,
                )
            )
            current_shape = _flatten_shape(current_shape, layer.start_dim)
        elif isinstance(layer, ReLU):
            nodes.append(
                helper.make_node("Relu", [current_name], [next_name], name=f"ReLU_{index}")
            )
        elif isinstance(layer, Sigmoid):
            nodes.append(
                helper.make_node("Sigmoid", [current_name], [next_name], name=f"Sigmoid_{index}")
            )
        elif isinstance(layer, Tanh):
            nodes.append(
                helper.make_node("Tanh", [current_name], [next_name], name=f"Tanh_{index}")
            )
        elif isinstance(layer, MaxPool2d):
            pads = [layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]]
            nodes.append(
                helper.make_node(
                    "MaxPool",
                    [current_name],
                    [next_name],
                    name=f"MaxPool2d_{index}",
                    kernel_shape=list(layer.kernel_size),
                    strides=list(layer.stride),
                    pads=pads,
                    dilations=list(layer.dilation),
                )
            )
            current_shape = _pool_output_shape(
                current_shape,
                layer.kernel_size,
                layer.stride,
                layer.padding,
                layer.dilation,
            )
        elif isinstance(layer, AvgPool2d):
            pads = [layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]]
            nodes.append(
                helper.make_node(
                    "AveragePool",
                    [current_name],
                    [next_name],
                    name=f"AvgPool2d_{index}",
                    kernel_shape=list(layer.kernel_size),
                    strides=list(layer.stride),
                    pads=pads,
                    count_include_pad=int(layer.count_include_pad),
                )
            )
            current_shape = _pool_output_shape(
                current_shape,
                layer.kernel_size,
                layer.stride,
                layer.padding,
            )
        else:
            raise ValueError(f"ONNX export does not support module {layer.__class__.__name__}")
        current_name = next_name

    if nodes:
        nodes[-1].output[0] = output_name
    else:
        raise ValueError("ONNX export requires at least one module")

    graph = helper.make_graph(
        nodes,
        "TensorStudioModel",
        [helper.make_tensor_value_info(input_name, _onnx_dtype(input_dtype), list(shape))],
        [helper.make_tensor_value_info(output_name, _onnx_dtype(input_dtype), list(current_shape))],
        initializer=initializers,
    )
    onnx_model = helper.make_model(
        graph,
        producer_name="tensorstudio",
        opset_imports=[helper.make_operatorsetid("", opset)],
    )

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import onnx

    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, output_path)
    return output_path


__all__ = ["export_onnx"]
