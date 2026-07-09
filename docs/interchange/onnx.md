# ONNX Interchange

TensorStudio `2.1.0` includes a limited ONNX exporter for supported
TensorStudio module stacks.

Install the optional export/metadata dependency:

```bash
python -m pip install "tensorstudio[onnx]"
```

Install optional ONNX Runtime provider diagnostics:

```bash
python -m pip install "tensorstudio[onnxruntime]"
```

## Export A Sequential Model

```python
import tensorstudio as ts
from tensorstudio import nn

model = nn.Sequential(
    nn.Conv2d(1, 2, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2),
    nn.Flatten(),
    nn.Linear(2 * 2 * 2, 3),
)

ts.export_onnx(model, "classifier.onnx", input_shape=(1, 1, 4, 4))
```

The exporter writes an ONNX graph and validates it with `onnx.checker` before
saving. TensorStudio caps the exported ONNX IR version to a runtime-compatible
value for the operators it emits so current ONNX Runtime builds can load the
resulting static graphs.

## Supported Modules

- `nn.Linear`
- `nn.Conv2d`
- `nn.Flatten`
- `nn.ReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.MaxPool2d`
- `nn.AvgPool2d`

`ts.vision.TinyConvClassifier` and `ts.vision.ImageClassifier` are exportable
because their nested vision blocks flatten to supported TensorStudio layers.

## Export Contract

`export_onnx` requires a concrete `input_shape`. It performs lightweight shape
propagation while writing nodes, initializers, and output metadata.

```python
path = ts.export_onnx(
    model,
    "model.onnx",
    input_shape=(1, 1, 28, 28),
    input_name="image",
    output_name="logits",
)
```

## Metadata Inspection

`inspect_onnx` reads model metadata without executing the model:

```python
info = ts.inspect_onnx("model.onnx")
same = ts.inspect_model_format("model.onnx")

print(info["inputs"])
print(info["outputs"])
print(info["op_counts"])
print(info["opsets"])
```

The returned dictionary includes:

- IR version, producer, graph name, and model metadata properties.
- Opset versions by domain.
- Graph inputs and outputs with dtype and shape metadata.
- Initializer names.
- Node count, operator types, and operator counts.
- ONNX checker pass/fail status.

## ONNX Runtime Diagnostics

TensorStudio does not bundle an ONNX runtime. When the optional
`tensorstudio[onnxruntime]` extra is installed, it can ask ONNX Runtime which
providers are available and whether a model can be loaded with requested
providers. It can also run compatible static graphs through ONNX Runtime:

```python
runtime = ts.onnx_runtime_info(providers=["CPUExecutionProvider"])
compat = ts.check_onnx_runtime_compatibility(
    "model.onnx",
    providers=["CPUExecutionProvider"],
)
outputs = ts.run_onnx_inference(
    "model.onnx",
    {"image": ts.zeros((1, 1, 28, 28))},
    providers=["CPUExecutionProvider"],
)

print(runtime["available_providers"])
print(compat["compatible"])
print(compat["session_providers"])
print(outputs["logits"].shape)
```

`check_onnx_runtime_compatibility` creates an ONNX Runtime session when the
optional dependency is installed, but it does not run inference.
`run_onnx_inference` accepts TensorStudio tensors or NumPy-compatible arrays,
validates input names, defaults to `CPUExecutionProvider` when it is available,
and returns TensorStudio tensors by default. Use `as_tensor=False` to receive
NumPy arrays from ONNX Runtime directly.

## Current Limits

- TensorStudio does not import ONNX models into TensorStudio modules yet.
- No ONNX runtime is bundled; runtime diagnostics require the optional
  `tensorstudio[onnxruntime]` extra.
- ONNX Runtime execution uses ONNX Runtime providers, not TensorStudio kernels.
- Arbitrary Python `forward` control flow is not traced.
- Unsupported modules raise `ValueError` instead of silently producing an
  incomplete graph.
- Dynamic axes are not modeled yet; provide a concrete input shape.
