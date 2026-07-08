# ONNX Import

TensorStudio can inspect ONNX files and import a constrained static
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

For broader runtime compatibility, install `tensorstudio[onnxruntime]` and use
the optional ONNX Runtime adapter:

```python
output = ts.run_onnx("model.onnx", ts.ones((1, 3)))
```

When `onnxruntime` is unavailable, `run_onnx(..., prefer_onnxruntime=False)`
uses the same constrained TensorStudio importer described here.

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
