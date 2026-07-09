"""Limited ONNX export for TensorStudio module graphs."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from tensorstudio.nn import (
    AvgPool2d,
    Conv2d,
    ConvTranspose2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
    Sigmoid,
    Tanh,
)
from tensorstudio.ops import avg_pool2d, conv2d, conv_transpose2d, max_pool2d
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
                    group=layer.groups,
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
        elif isinstance(layer, ConvTranspose2d):
            weight_name = f"conv_transpose_{index}_weight"
            inputs = [current_name, weight_name]
            initializers.append(_tensor_initializer(weight_name, layer.weight, numpy_helper))
            if layer.bias is not None:
                bias_name = f"conv_transpose_{index}_bias"
                inputs.append(bias_name)
                initializers.append(_tensor_initializer(bias_name, layer.bias, numpy_helper))
            pads = [layer.padding[0], layer.padding[1], layer.padding[0], layer.padding[1]]
            nodes.append(
                helper.make_node(
                    "ConvTranspose",
                    inputs,
                    [next_name],
                    name=f"ConvTranspose2d_{index}",
                    kernel_shape=list(layer.kernel_size),
                    strides=list(layer.stride),
                    pads=pads,
                    dilations=list(layer.dilation),
                    output_padding=list(layer.output_padding),
                    group=layer.groups,
                )
            )
            if len(current_shape) != 4:
                raise ValueError("ConvTranspose2d ONNX export expects input shape (N, C, H, W)")
            batch, _, height, width = current_shape
            out_h = (
                (height - 1) * layer.stride[0]
                - 2 * layer.padding[0]
                + layer.dilation[0] * (layer.kernel_size[0] - 1)
                + layer.output_padding[0]
                + 1
            )
            out_w = (
                (width - 1) * layer.stride[1]
                - 2 * layer.padding[1]
                + layer.dilation[1] * (layer.kernel_size[1] - 1)
                + layer.output_padding[1]
                + 1
            )
            current_shape = (batch, layer.out_channels, out_h, out_w)
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


class ImportedOnnxModel:
    """Executable TensorStudio representation of a supported static ONNX graph."""

    def __init__(self, path: PathLikeStr) -> None:
        _, _, numpy_helper = _require_onnx()
        import onnx

        self.path = Path(path)
        self.model = onnx.load(self.path)
        self.initializers: dict[str, Tensor] = {
            initializer.name: from_numpy(np.array(numpy_helper.to_array(initializer), copy=True))
            for initializer in self.model.graph.initializer
        }
        self.input_names = [
            value.name
            for value in self.model.graph.input
            if value.name not in self.initializers
        ]
        self.output_names = [value.name for value in self.model.graph.output]
        self.nodes = list(self.model.graph.node)
        unsupported = sorted({node.op_type for node in self.nodes} - _SUPPORTED_IMPORT_OPS)
        if unsupported:
            raise ValueError(f"unsupported ONNX operators for TensorStudio import: {unsupported}")
        if len(self.input_names) != 1:
            raise ValueError("TensorStudio ONNX import currently supports exactly one graph input")

    def __call__(self, input: Tensor) -> Tensor:
        values: dict[str, Tensor] = dict(self.initializers)
        values[self.input_names[0]] = input
        for node in self.nodes:
            self._execute_node(node, values)
        if len(self.output_names) != 1:
            raise ValueError("TensorStudio ONNX import currently supports exactly one graph output")
        return values[self.output_names[0]]

    def _execute_node(self, node: Any, values: dict[str, Tensor]) -> None:
        attrs = _node_attributes(node)
        inputs = [values[name] for name in node.input if name]
        op_type = node.op_type
        if op_type == "Gemm":
            left = inputs[0].T if int(attrs.get("transA", 0)) else inputs[0]
            right = inputs[1].T if int(attrs.get("transB", 0)) else inputs[1]
            output = left @ right
            if len(inputs) > 2:
                output = output + inputs[2]
        elif op_type == "Relu":
            output = inputs[0].relu()
        elif op_type == "Sigmoid":
            output = inputs[0].sigmoid()
        elif op_type == "Tanh":
            output = inputs[0].tanh()
        elif op_type == "Flatten":
            axis = int(attrs.get("axis", 1))
            shape = inputs[0].shape
            normalized = axis if axis >= 0 else len(shape) + axis
            prefix = shape[:normalized]
            flattened = math.prod(shape[normalized:]) if normalized < len(shape) else 1
            output = inputs[0].reshape((*prefix, flattened))
        elif op_type == "Conv":
            output = conv2d(
                inputs[0],
                inputs[1],
                inputs[2] if len(inputs) > 2 else None,
                stride=_attr_pair(attrs, "strides", (1, 1)),
                padding=_pads_to_pair(attrs.get("pads", [0, 0, 0, 0])),
                dilation=_attr_pair(attrs, "dilations", (1, 1)),
                groups=int(attrs.get("group", 1)),
            )
        elif op_type == "ConvTranspose":
            output = conv_transpose2d(
                inputs[0],
                inputs[1],
                inputs[2] if len(inputs) > 2 else None,
                stride=_attr_pair(attrs, "strides", (1, 1)),
                padding=_pads_to_pair(attrs.get("pads", [0, 0, 0, 0])),
                output_padding=_attr_pair(attrs, "output_padding", (0, 0)),
                dilation=_attr_pair(attrs, "dilations", (1, 1)),
                groups=int(attrs.get("group", 1)),
            )
        elif op_type == "MaxPool":
            output = max_pool2d(
                inputs[0],
                kernel_size=_attr_pair(attrs, "kernel_shape", (1, 1)),
                stride=_attr_pair(attrs, "strides", (1, 1)),
                padding=_pads_to_pair(attrs.get("pads", [0, 0, 0, 0])),
                dilation=_attr_pair(attrs, "dilations", (1, 1)),
            )
        elif op_type == "AveragePool":
            output = avg_pool2d(
                inputs[0],
                kernel_size=_attr_pair(attrs, "kernel_shape", (1, 1)),
                stride=_attr_pair(attrs, "strides", (1, 1)),
                padding=_pads_to_pair(attrs.get("pads", [0, 0, 0, 0])),
                count_include_pad=bool(attrs.get("count_include_pad", 0)),
            )
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(f"unsupported ONNX operator: {op_type}")
        values[node.output[0]] = output


_SUPPORTED_IMPORT_OPS = {
    "AveragePool",
    "Conv",
    "ConvTranspose",
    "Flatten",
    "Gemm",
    "MaxPool",
    "Relu",
    "Sigmoid",
    "Tanh",
}


def inspect_onnx(path: PathLikeStr) -> dict[str, Any]:
    """Inspect supported ONNX metadata without executing the graph."""

    _, _, numpy_helper = _require_onnx()
    import onnx

    model = onnx.load(Path(path))
    try:
        onnx.checker.check_model(model)
        check_passed = True
        check_error = None
    except Exception as exc:
        check_passed = False
        check_error = str(exc)
    op_counts: dict[str, int] = {}
    for node in model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    initializers = [
        {
            "name": initializer.name,
            "shape": list(numpy_helper.to_array(initializer).shape),
            "dtype": str(numpy_helper.to_array(initializer).dtype),
        }
        for initializer in model.graph.initializer
    ]
    return {
        "format": "onnx",
        "check_passed": check_passed,
        "check_error": check_error,
        "producer_name": model.producer_name,
        "producer_version": model.producer_version,
        "opsets": {item.domain or "": item.version for item in model.opset_import},
        "graph_name": model.graph.name,
        "inputs": [value.name for value in model.graph.input],
        "outputs": [value.name for value in model.graph.output],
        "node_count": len(model.graph.node),
        "operators": sorted({node.op_type for node in model.graph.node}),
        "op_counts": op_counts,
        "initializer_count": len(initializers),
        "initializers": initializers,
    }


def import_onnx(path: PathLikeStr) -> ImportedOnnxModel:
    """Import a supported static ONNX graph as a callable TensorStudio model."""

    return ImportedOnnxModel(path)


class OnnxRuntimeModel:
    """Callable ONNX Runtime session returning TensorStudio tensors.

    This adapter is optional and requires ``onnxruntime`` to be installed. It is
    intentionally separate from TensorStudio's native supported-subset importer:
    ONNX Runtime can execute a broader ONNX surface, while TensorStudio import
    executes only graphs whose operators map to TensorStudio tensor ops.
    """

    def __init__(
        self,
        path: PathLikeStr,
        *,
        providers: list[str] | None = None,
    ) -> None:
        ort = _require_onnxruntime()
        _validate_onnxruntime_providers(providers)
        self.path = Path(path)
        self.session = ort.InferenceSession(str(self.path), providers=providers)
        self.input_names = [item.name for item in self.session.get_inputs()]
        self.output_names = [item.name for item in self.session.get_outputs()]
        if len(self.input_names) != 1:
            raise ValueError("TensorStudio ONNX Runtime adapter currently supports one input")

    def __call__(self, input: Tensor) -> Tensor:
        outputs = self.session.run(None, {self.input_names[0]: input.numpy()})
        if len(outputs) != 1:
            raise ValueError("TensorStudio ONNX Runtime adapter currently supports one output")
        return from_numpy(np.asarray(outputs[0]))


def onnxruntime_is_available() -> bool:
    """Return whether optional ONNX Runtime execution is importable."""

    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return True


def onnxruntime_available_providers() -> list[str]:
    """Return ONNX Runtime execution providers available in this environment."""

    try:
        ort = _require_onnxruntime()
    except ImportError:
        return []
    return list(ort.get_available_providers())


def check_onnxruntime_compatibility(
    path: PathLikeStr,
    *,
    providers: list[str] | None = None,
) -> dict[str, Any]:
    """Inspect whether ONNX Runtime can load a model with requested providers."""

    result: dict[str, Any] = {
        "available": onnxruntime_is_available(),
        "available_providers": onnxruntime_available_providers(),
        "requested_providers": list(providers or []),
        "loadable": False,
        "error": None,
    }
    if not result["available"]:
        result["error"] = "onnxruntime is not installed"
        return result
    try:
        model = OnnxRuntimeModel(path, providers=providers)
        result["loadable"] = True
        result["inputs"] = list(model.input_names)
        result["outputs"] = list(model.output_names)
    except Exception as exc:
        result["error"] = str(exc)
    return result


def _selected_onnxruntime_providers(providers: Iterable[str] | None = None) -> list[str] | None:
    requested = list(providers) if providers is not None else None
    _validate_onnxruntime_providers(requested)
    if requested:
        return requested
    available = onnxruntime_available_providers()
    if "CPUExecutionProvider" in available:
        return ["CPUExecutionProvider"]
    return None


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

    The helper executes the model through ONNX Runtime rather than importing
    TensorFlow/PyTorch/Keras code. CPUExecutionProvider is selected by default
    when it is available so local runs are deterministic across machines.
    """

    ort = _require_onnxruntime()
    selected_providers = _selected_onnxruntime_providers(providers)
    session = ort.InferenceSession(str(Path(path)), providers=selected_providers)
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


def run_onnx(
    path: PathLikeStr,
    input: Tensor,
    *,
    prefer_onnxruntime: bool = True,
    providers: list[str] | None = None,
) -> Tensor:
    """Run an ONNX file through ONNX Runtime when available, otherwise TensorStudio import."""

    runtime_error: Exception | None = None
    if prefer_onnxruntime and onnxruntime_is_available():
        try:
            return OnnxRuntimeModel(path, providers=providers)(input)
        except Exception as exc:
            runtime_error = exc

    try:
        return import_onnx(path)(input)
    except Exception as fallback_error:
        if runtime_error is not None:
            raise RuntimeError(
                "ONNX Runtime execution failed and TensorStudio's supported-subset "
                "fallback could not execute the graph. "
                f"ONNX Runtime error: {runtime_error}. "
                f"TensorStudio fallback error: {fallback_error}."
            ) from fallback_error
        raise


def export_model_card_metadata(
    metadata: dict[str, Any],
    path: PathLikeStr,
) -> Path:
    """Write a small JSON model-card metadata file."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json_dumps_stable(metadata),
        encoding="utf-8",
    )
    return output_path


def json_dumps_stable(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def _require_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ImportError as exc:  # pragma: no cover - optional dependency path
        raise ImportError(
            "ONNX Runtime execution requires the optional dependency: "
            "python -m pip install 'tensorstudio[onnxruntime]'"
        ) from exc
    return ort


def _validate_onnxruntime_providers(providers: list[str] | None) -> None:
    if not providers:
        return
    available = set(onnxruntime_available_providers())
    missing = [provider for provider in providers if provider not in available]
    if missing:
        raise ValueError(
            "requested ONNX Runtime providers are unavailable: "
            f"{missing}; available providers: {sorted(available)}"
        )


def _node_attributes(node: Any) -> dict[str, Any]:
    _, helper, _ = _require_onnx()
    return {attribute.name: helper.get_attribute_value(attribute) for attribute in node.attribute}


def _attr_pair(
    attrs: dict[str, Any],
    name: str,
    default: tuple[int, int],
) -> tuple[int, int]:
    value = attrs.get(name, default)
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, (int, float)):
        return (int(value), int(value))
    if len(value) != 2:
        raise ValueError(f"ONNX attribute {name!r} must contain two values")
    return (int(value[0]), int(value[1]))


def _pads_to_pair(value: Any) -> tuple[int, int]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if len(value) == 2:
        return (int(value[0]), int(value[1]))
    if len(value) == 4 and int(value[0]) == int(value[2]) and int(value[1]) == int(value[3]):
        return (int(value[0]), int(value[1]))
    raise ValueError("TensorStudio ONNX import supports symmetric 2D padding only")


__all__ = [
    "ImportedOnnxModel",
    "OnnxRuntimeModel",
    "check_onnxruntime_compatibility",
    "export_model_card_metadata",
    "export_onnx",
    "import_onnx",
    "inspect_onnx",
    "onnxruntime_available_providers",
    "onnxruntime_is_available",
    "run_onnx",
    "run_onnx_inference",
]
