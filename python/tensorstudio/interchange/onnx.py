"""Limited ONNX export for TensorStudio module graphs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping
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
from tensorstudio.tensor import Tensor, from_numpy
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


def _load_onnx(path: PathLikeStr) -> Any:
    _require_onnx()
    import onnx

    return onnx.load(Path(path))


def _value_info_metadata(value_info: Any, tensor_proto: Any) -> dict[str, Any]:
    tensor_type = value_info.type.tensor_type
    elem_type = int(tensor_type.elem_type)
    dtype = str(tensor_proto.DataType.Name(elem_type)).lower()
    shape: list[int | str | None] = []
    for dim in tensor_type.shape.dim:
        if dim.dim_param:
            shape.append(str(dim.dim_param))
        elif dim.HasField("dim_value"):
            shape.append(int(dim.dim_value))
        else:
            shape.append(None)
    return {"name": str(value_info.name), "dtype": dtype, "shape": shape}


def _onnx_dtype(dtype: str) -> int:
    TensorProto, _, _ = _require_onnx()
    name = _DTYPE_TO_ONNX_NAME.get(dtype)
    if name is None:
        raise ValueError(f"unsupported ONNX export dtype: {dtype!r}")
    return int(getattr(TensorProto, name))


def _module_sequence(model: Module) -> list[Module]:
    return _flatten_modules([model])


def _flatten_modules(modules: Iterable[Module]) -> list[Module]:
    result: list[Module] = []
    for module in modules:
        if isinstance(module, Sequential):
            result.extend(_flatten_modules(list(module)))
            continue
        nested = getattr(module, "model", None)
        if isinstance(nested, Module) and nested is not module:
            result.extend(_flatten_modules([nested]))
            continue
        layers = getattr(module, "layers", None)
        if isinstance(layers, Sequential):
            result.extend(_flatten_modules(list(layers)))
            continue
        if isinstance(layers, Iterable) and not isinstance(layers, (str, bytes)):
            layer_list = list(layers)
            if layer_list and all(isinstance(layer, Module) for layer in layer_list):
                result.extend(_flatten_modules(layer_list))
                continue
        result.append(module)
    return result


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
    control flow is intentionally not traced.
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
    if onnx_model.ir_version > 10:
        onnx_model.ir_version = 10

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    import onnx

    onnx.checker.check_model(onnx_model)
    onnx.save_model(onnx_model, output_path)
    return output_path


def inspect_onnx(path: PathLikeStr, *, check: bool = True) -> dict[str, Any]:
    """Inspect ONNX model metadata without executing the model.

    The returned dictionary contains graph IO metadata, node/operator counts,
    opset versions, initializer names, producer metadata, and checker status.
    """

    TensorProto, _, _ = _require_onnx()
    import onnx

    model = _load_onnx(path)
    checker_error: str | None = None
    if check:
        try:
            onnx.checker.check_model(model)
        except Exception as exc:  # pragma: no cover - depends on malformed external files
            checker_error = str(exc)

    initializer_names = {initializer.name for initializer in model.graph.initializer}
    op_counts = Counter(str(node.op_type) for node in model.graph.node)
    metadata_props = {prop.key: prop.value for prop in model.metadata_props}

    return {
        "path": str(Path(path)),
        "ir_version": int(model.ir_version),
        "producer_name": str(model.producer_name),
        "producer_version": str(model.producer_version),
        "domain": str(model.domain),
        "model_version": int(model.model_version),
        "graph_name": str(model.graph.name),
        "opsets": {
            str(opset.domain): int(opset.version)
            for opset in model.opset_import
        },
        "inputs": [
            _value_info_metadata(value_info, TensorProto)
            for value_info in model.graph.input
            if value_info.name not in initializer_names
        ],
        "outputs": [
            _value_info_metadata(value_info, TensorProto)
            for value_info in model.graph.output
        ],
        "initializers": sorted(initializer_names),
        "node_count": len(model.graph.node),
        "op_types": sorted(op_counts),
        "op_counts": dict(sorted(op_counts.items())),
        "metadata": metadata_props,
        "check_passed": checker_error is None,
        "check_error": checker_error,
    }


def onnx_runtime_info(providers: Iterable[str] | None = None) -> dict[str, Any]:
    """Return optional ONNX Runtime provider availability metadata."""

    requested = list(providers) if providers is not None else []
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "available": False,
            "version": None,
            "available_providers": [],
            "requested_providers": requested,
            "selected_providers": [],
            "unsupported_providers": requested,
            "reason": (
                "install the optional dependency with: "
                "python -m pip install 'tensorstudio[onnxruntime]'"
            ),
        }

    available = list(ort.get_available_providers())
    unsupported = [provider for provider in requested if provider not in available]
    selected = requested if requested and not unsupported else ([] if requested else available)
    return {
        "available": True,
        "version": str(ort.__version__),
        "available_providers": available,
        "requested_providers": requested,
        "selected_providers": selected,
        "unsupported_providers": unsupported,
        "reason": (
            ""
            if not unsupported
            else "one or more requested ONNX Runtime providers are unavailable"
        ),
    }


def _selected_runtime_providers(providers: Iterable[str] | None = None) -> list[str]:
    runtime = onnx_runtime_info(providers)
    if not runtime["available"]:
        raise ImportError(str(runtime["reason"]))
    if runtime["unsupported_providers"]:
        missing = ", ".join(str(provider) for provider in runtime["unsupported_providers"])
        raise ValueError(f"unsupported ONNX Runtime provider(s): {missing}")
    selected = list(runtime["selected_providers"])
    if providers is None and "CPUExecutionProvider" in runtime["available_providers"]:
        return ["CPUExecutionProvider"]
    return selected


def _onnx_input_array(value: Any) -> np.ndarray:
    array = value.numpy() if isinstance(value, Tensor) else np.asarray(value)
    return np.ascontiguousarray(array)


def _normalize_output_names(output_names: str | Iterable[str] | None) -> list[str] | None:
    if output_names is None:
        return None
    if isinstance(output_names, str):
        return [output_names]
    return [str(name) for name in output_names]


def run_onnx_inference(
    path: PathLikeStr,
    inputs: Mapping[str, Any] | Tensor | np.ndarray,
    *,
    providers: Iterable[str] | None = None,
    output_names: str | Iterable[str] | None = None,
    as_tensor: bool = True,
) -> dict[str, Tensor | np.ndarray]:
    """Run a compatible ONNX model with ONNX Runtime.

    The helper never executes Python code from the model file; execution is
    delegated to ONNX Runtime. By default, CPUExecutionProvider is selected
    when it is available so local runs do not accidentally depend on optional
    accelerator providers.
    """

    selected_providers = _selected_runtime_providers(providers)
    import onnxruntime as ort

    session = ort.InferenceSession(str(Path(path)), providers=selected_providers or None)
    model_inputs = [item.name for item in session.get_inputs()]
    if isinstance(inputs, Mapping):
        feed = {str(name): _onnx_input_array(value) for name, value in inputs.items()}
    else:
        if len(model_inputs) != 1:
            raise ValueError(
                "non-mapping ONNX inputs are only supported for single-input models; "
                f"model expects {model_inputs}"
            )
        feed = {model_inputs[0]: _onnx_input_array(inputs)}

    missing = [name for name in model_inputs if name not in feed]
    extra = sorted(name for name in feed if name not in model_inputs)
    if missing:
        raise ValueError(f"missing ONNX input(s): {missing}")
    if extra:
        raise ValueError(f"unknown ONNX input(s): {extra}; expected {model_inputs}")

    requested_outputs = _normalize_output_names(output_names)
    output_values = session.run(requested_outputs, feed)
    names = requested_outputs or [item.name for item in session.get_outputs()]
    result: dict[str, Tensor | np.ndarray] = {}
    for name, value in zip(names, output_values, strict=True):
        array = np.asarray(value)
        result[name] = from_numpy(array) if as_tensor else array
    return result


def check_onnx_runtime_compatibility(
    path: PathLikeStr,
    *,
    providers: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Check ONNX checker status and optional ONNX Runtime session creation.

    This loads the ONNX model into ONNX Runtime when the optional dependency is
    installed, but it does not run inference.
    """

    model_info = inspect_onnx(path)
    runtime = onnx_runtime_info(providers)
    result: dict[str, Any] = {
        "path": str(Path(path)),
        "onnx_check_passed": model_info["check_passed"],
        "runtime_available": runtime["available"],
        "runtime_version": runtime["version"],
        "available_providers": runtime["available_providers"],
        "requested_providers": runtime["requested_providers"],
        "selected_providers": runtime["selected_providers"],
        "unsupported_providers": runtime["unsupported_providers"],
        "compatible": None,
        "session_providers": [],
        "reason": runtime["reason"],
    }
    if not model_info["check_passed"]:
        result["compatible"] = False
        result["reason"] = model_info["check_error"]
        return result
    if not runtime["available"]:
        return result
    if runtime["unsupported_providers"]:
        result["compatible"] = False
        return result

    import onnxruntime as ort

    try:
        session = ort.InferenceSession(
            str(Path(path)),
            providers=runtime["selected_providers"] or None,
        )
    except Exception as exc:  # pragma: no cover - provider/runtime dependent
        result["compatible"] = False
        result["reason"] = str(exc)
        return result

    result["compatible"] = True
    result["session_providers"] = list(session.get_providers())
    result["reason"] = ""
    return result


__all__ = [
    "check_onnx_runtime_compatibility",
    "export_onnx",
    "inspect_onnx",
    "onnx_runtime_info",
    "run_onnx_inference",
]
