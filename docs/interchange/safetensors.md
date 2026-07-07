# SafeTensors

TensorStudio `1.12.0` supports optional SafeTensors weight storage for flat
tensor mappings such as `state_dict()` outputs.

Install the optional dependency:

```bash
python -m pip install "tensorstudio[safetensors]"
```

## Save And Load

```python
import tensorstudio as ts

state = model.state_dict()
ts.save_safetensors(state, "weights.safetensors", metadata={"task": "demo"})
loaded = ts.load_safetensors("weights.safetensors")
model.load_state_dict(loaded)
```

SafeTensors stores tensor arrays without pickle. It is appropriate for model
weights and other numeric tensor maps, not arbitrary Python objects.

## Inspect Metadata

```python
metadata = ts.inspect_model_metadata("weights.safetensors")
print(metadata["tensor_count"])
print(metadata["metadata"])
```

TensorStudio records a small format marker and package version when writing
SafeTensors files.
