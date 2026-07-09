# Model Formats

TensorStudio supports a small set of model interchange formats where behavior
can be implemented honestly and tested.

## Supported Today

TensorStudio state dictionaries:

- `nn.Module.state_dict()` returns flat tensor dictionaries.
- `save_npz` and `load_npz` store those dictionaries without pickle.
- `inspect_npz` and `check_npz_compatibility` report versioned archive
  compatibility before loading tensors.

Trusted pickle checkpoints:

- `save` and `load` can round-trip Python objects.
- Only load pickle files from trusted sources.

ONNX export:

- Supported TensorStudio module stacks can be exported to ONNX.
- Export is limited to known layers and static shapes.
- `inspect_onnx` reports safe graph metadata and checker status.
- `check_onnx_runtime_compatibility` reports optional ONNX Runtime provider
  compatibility without running inference.
- `run_onnx_inference` can execute compatible static ONNX graphs through the
  optional ONNX Runtime dependency and return TensorStudio tensors or NumPy
  arrays.

Metadata-only inspection:

- `inspect_model_format(path)` routes supported model files to a safe inspector.
- `inspect_keras(path)` reads `.keras` archive JSON metadata without extracting
  the archive or loading custom code.
- `inspect_saved_model(path)` reports TensorFlow SavedModel directory structure,
  protobuf file presence, variables, and assets without importing TensorFlow.
- `inspect_hdf5(path)` checks HDF5/Keras weight-file signatures and, when
  optional `h5py` is available, reports groups, datasets, dtypes, shapes, and
  root attributes without loading a model.
- `inspect_tflite(path)` identifies TensorFlow Lite flatbuffer files without
  loading or executing the graph.

```python
info = ts.inspect_model_format("model.keras")
print(info["format"], info["safe_metadata_only"])
print(ts.inspect_tflite("model.tflite")["has_tflite_identifier"])
```

## Planned In Roadmap Order

- Richer NPZ metadata for model and optimizer state.
- Safe tensor storage formats such as `safetensors`.
- ONNX import for a tested subset of static graphs.
- TensorStudio-native execution of imported ONNX graphs only when TensorStudio
  has matching native ops.

## Unsupported Formats

TensorStudio does not claim compatibility with every neural-network file format.
Formats that require executing arbitrary Python code or unsupported dynamic
graphs should be rejected clearly rather than partially loaded.
