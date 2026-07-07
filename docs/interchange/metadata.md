# Model Metadata

`tensorstudio.inspect_model_metadata()` reads metadata for supported model and
weight files without pretending unsupported formats are compatible.

## NPZ

```python
ts.save_npz(model.state_dict(), "weights.tsnpz", metadata={"task": "classification"})
metadata = ts.inspect_model_metadata("weights.tsnpz")
```

TensorStudio NPZ metadata includes:

- archive format and version
- TensorStudio version that wrote the file
- tensor count
- tensor names, dtypes, shapes, and `requires_grad` flags
- user metadata passed to `save_npz`

## SafeTensors

```python
metadata = ts.inspect_model_metadata("weights.safetensors")
```

The SafeTensors path reports tensor count, tensor names, dtypes, shapes, and
embedded SafeTensors metadata.

## ONNX

```python
metadata = ts.inspect_model_metadata("model.onnx")
```

ONNX metadata includes graph input/output names, opset imports, producer fields,
operators, node counts, and initializer summaries.

## Trusted Checkpoints

Full TensorStudio checkpoints use pickle for optimizer and scheduler state.
Metadata inspection therefore requires an explicit trust opt-in:

```python
metadata = ts.inspect_model_metadata("checkpoint.tsmodel", trusted_pickle=True)
```

Do not enable `trusted_pickle=True` for files from untrusted sources.
