# Serialization

TensorStudio exposes trusted pickle roundtrips and safer NPZ tensor files.

## Trusted Object Roundtrips

```python
import tensorstudio as ts

ts.save(obj, "object.tsmodel")
loaded = ts.load("object.tsmodel")
```

## Supported Objects

The v1 release supports saving and loading:

- `Tensor` objects.
- Plain dictionaries and lists containing tensors.
- `nn.Module.state_dict()` dictionaries.
- Optimizer `state_dict()` dictionaries.
- Python objects that are otherwise pickle-compatible.

For models, prefer saving state dictionaries:

```python
checkpoint = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
}
ts.save(checkpoint, "checkpoint.tsmodel")
```

Load into fresh objects:

```python
checkpoint = ts.load("checkpoint.tsmodel")
model.load_state_dict(checkpoint["model"])
optimizer.load_state_dict(checkpoint["optimizer"])
```

## Security Warning

Pickle is not a safe interchange format. Loading pickle files from untrusted
sources is unsafe because pickle can execute arbitrary code during
deserialization.

Use TensorStudio serialization only for files you created or otherwise trust.

## NPZ Tensor And State Files

For portable tensor weights, prefer the non-pickle NPZ helpers:

```python
ts.save_npz(model.state_dict(), "weights.tsnpz")
model.load_state_dict(ts.load_npz("weights.tsnpz"))
```

`save_npz` supports a single `Tensor` or a flat `dict[str, Tensor]`, such as an
`nn.Module.state_dict()`. It stores raw NumPy arrays plus TensorStudio JSON
metadata and loads with NumPy pickle support disabled.

## ONNX Model Files

TensorStudio can export supported module stacks to ONNX:

```python
ts.export_onnx(model, "model.onnx", input_shape=(1, 1, 28, 28))
```

ONNX support is export-only in v1.2 and covers common TensorStudio layers such
as `Linear`, `Conv2d`, `Flatten`, activations, and 2D pooling.
