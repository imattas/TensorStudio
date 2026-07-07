# Serialization

TensorStudio exposes simple pickle-based helpers:

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

## Future Work

Future releases may add a structured non-executable tensor checkpoint format.
For `1.0.1`, pickle keeps the implementation small and clear.
