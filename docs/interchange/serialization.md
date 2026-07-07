# Serialization

TensorStudio `1.10.0` provides two serialization paths: trusted pickle
roundtrips for internal objects and non-pickle NPZ files for tensor/state_dict
interchange.

## Trusted Pickle Roundtrips

```python
import tensorstudio as ts

ts.save(obj, "object.tsmodel")
obj = ts.load("object.tsmodel")
```

## Security Warning

`save` and `load` use Python pickle. Pickle can execute arbitrary code during
deserialization. Only load `.tsmodel` files from trusted sources.

## What Can Be Saved

The pickle path is intended for TensorStudio objects such as:

- tensors
- Python modules made from `tensorstudio.nn`
- simple containers of tensors and modules

Example:

```python
from tensorstudio import nn

model = nn.Sequential(nn.Linear(2, 4), nn.Tanh(), nn.Linear(4, 1))
ts.save(model, "model.tsmodel")
loaded = ts.load("model.tsmodel")
```

## Safer NPZ Tensor Files

Use `save_npz` and `load_npz` for numeric tensor data and flat module
`state_dict` mappings:

```python
state = model.state_dict()
ts.save_npz(state, "weights.tsnpz")
loaded_state = ts.load_npz("weights.tsnpz")
model.load_state_dict(loaded_state)
```

Single tensors are supported too:

```python
x = ts.tensor([[1.0, 2.0]], requires_grad=True)
ts.save_npz(x, "tensor.tsnpz")
roundtrip = ts.load_npz("tensor.tsnpz")
```

NPZ archives store NumPy arrays and TensorStudio JSON metadata with pickle
disabled. They do not save arbitrary Python modules, optimizer objects, or
custom classes.

## ONNX Export

For cross-tool model exchange, TensorStudio can export a supported module graph
to ONNX when the `onnx` extra is installed:

```python
ts.export_onnx(model, "model.onnx", input_shape=(1, 1, 28, 28))
```

See [ONNX Interchange](onnx.md) for supported module types and limits.

## Recommended File Extensions

- Use `.tsmodel` for trusted TensorStudio pickle files.
- Use `.tsnpz` for tensor and state_dict NPZ files.
- Use `.onnx` for exported ONNX model files.
