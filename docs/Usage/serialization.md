# Serialization

TensorStudio `1.1.0` provides simple Python-level serialization:

```python
import tensorstudio as ts

ts.save(obj, "object.tsmodel")
obj = ts.load("object.tsmodel")
```

## Security Warning

Serialization uses Python pickle. Pickle can execute arbitrary code during
deserialization. Only load files from trusted sources.

## What Can Be Saved

The current implementation is intended for TensorStudio objects such as:

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

## Version Compatibility

The release-candidate pickle format is intended for TensorStudio object
roundtrips. For long-term model exchange, ONNX import/export is planned as
future work.

## Recommended File Extension

Use `.tsmodel` for TensorStudio pickle files.
