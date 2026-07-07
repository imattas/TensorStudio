# ONNX Import

TensorStudio `1.14.0` can inspect ONNX files and import a constrained static
subset for eager TensorStudio execution.

## Inspect

```python
metadata = ts.inspect_onnx("model.onnx")
print(metadata["operators"])
print(metadata["initializers"])
```

Inspection does not execute graph nodes.

## Import And Run

```python
model = ts.import_onnx("model.onnx")
output = model(ts.ones((1, 3)))
```

Supported operators:

- `Gemm`
- `Relu`
- `Sigmoid`
- `Tanh`
- `Flatten`
- `Conv`
- `ConvTranspose`
- `MaxPool`
- `AveragePool`

Imported graphs must have exactly one non-initializer input and one graph
output. Dynamic control flow, dynamic shapes, unsupported operators, and
asymmetric padding are rejected clearly.

## Export Coverage

`export_onnx()` now writes grouped convolution metadata and can export
`ConvTranspose2d` in addition to the earlier linear, convolution, activation,
flatten, and pooling modules.

## Model Card Metadata

Use `export_model_card_metadata()` for a small JSON sidecar:

```python
ts.export_model_card_metadata(
    {"name": "demo", "framework": "tensorstudio"},
    "model-card.json",
)
```
