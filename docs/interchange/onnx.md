# ONNX Interchange

TensorStudio `1.12.0` includes a limited ONNX exporter, ONNX metadata
inspection, and supported-subset ONNX import/execution for TensorStudio module
stacks.

Install the optional dependency:

```bash
python -m pip install "tensorstudio[onnx]"
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
saving.

## Inspect And Import

```python
metadata = ts.inspect_onnx("classifier.onnx")
imported = ts.import_onnx("classifier.onnx")
output = imported(ts.ones((1, 1, 4, 4)))
```

Import supports a constrained static subset: `Gemm`, `Relu`, `Sigmoid`, `Tanh`,
`Flatten`, `Conv`, `ConvTranspose`, `MaxPool`, and `AveragePool`. Unsupported
operators, dynamic graph features, multiple graph inputs, and asymmetric
padding are rejected clearly.

## Supported Modules

- `nn.Linear`
- `nn.Conv2d`, including grouped/depthwise convolution metadata
- `nn.ConvTranspose2d`
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

## Current Limits

- Import is limited to the static subset listed above.
- No full ONNX runtime is bundled.
- Arbitrary Python `forward` control flow is not traced.
- Unsupported modules raise `ValueError` instead of silently producing an
  incomplete graph.
- Dynamic axes are not modeled yet; provide a concrete input shape.
