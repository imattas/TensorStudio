# Model Formats

TensorStudio supports model interchange formats only where behavior can be
implemented honestly, inspected safely, and covered by tests. The goal is useful
small-framework interoperability, not a fake claim of universal compatibility.

## Supported Today

TensorStudio state dictionaries:

- `nn.Module.state_dict()` returns flat tensor dictionaries.
- `save_npz` and `load_npz` store those dictionaries without pickle.
- `save_npz(..., metadata={...})` records TensorStudio version, tensor count,
  tensor names, shapes, dtypes, `requires_grad` flags, and user metadata.
- `load_npz_metadata()` and `inspect_model_metadata()` can inspect the archive
  metadata before loading tensor values.

SafeTensors:

- `save_safetensors` and `load_safetensors` support flat tensor weight maps.
- SafeTensors files do not use pickle and are appropriate for tensor-only
  weights.
- TensorStudio reports tensor names, shapes, dtypes, count, and embedded
  SafeTensors metadata through `inspect_model_metadata()`.

Trusted pickle checkpoints:

- `save` and `load` can round-trip Python objects.
- `save_checkpoint` stores versioned checkpoint metadata for model state,
  optimizer state, and user metadata.
- Checkpoint metadata inspection requires `trusted_pickle=True` because pickle
  is executable code.
- Only load pickle files from trusted sources.

ONNX:

- Supported TensorStudio module stacks can be exported to ONNX with static input
  shapes.
- Supported ONNX files can be inspected for graph inputs, outputs, opset,
  initializers, node counts, operator names, and producer fields.
- TensorStudio can import and execute a constrained static ONNX subset:
  `Gemm`, `Relu`, `Sigmoid`, `Tanh`, `Flatten`, `Conv`, `ConvTranspose`,
  `MaxPool`, and `AveragePool`.
- `run_onnx()` can delegate execution to the external ONNX Runtime package
  when installed through `tensorstudio[onnxruntime]`.
- Unsupported operators, multiple graph inputs or outputs, dynamic graph
  features, and asymmetric padding are rejected clearly.

## Practical Format Research

These formats are intentionally supported:

- TensorStudio NPZ for safe local state dictionaries and experiment artifacts.
- SafeTensors for safe tensor-only weight exchange.
- ONNX for static graph exchange when TensorStudio has matching eager tensor
  operations.

Metadata-only inspection is implemented in `2.1.0` for additional ecosystem
formats:

- Keras `.keras` archives: `inspect_keras()` reads archive entries,
  metadata/config JSON, layer classes, and weight-entry names without importing
  Keras code.
- TensorFlow SavedModel directories: `inspect_saved_model()` reports
  `saved_model.pb`, variables, assets, and fingerprint files without loading
  TensorFlow.
- HDF5/Keras weight files: `inspect_hdf5()` verifies the HDF5 signature and,
  when optional `h5py` is installed, reports groups/datasets/attributes.
- TensorFlow Lite files: `inspect_tflite()` verifies the `TFL3` flatbuffer
  identifier and reports safe file metadata.
- `inspect_model_format()` routes ONNX, Keras, SavedModel, HDF5, and TFLite
  inputs to the matching safe inspector.

These formats are not executable/imported in `2.1.0`:

- PyTorch `.pt` and `.pth`: loading general PyTorch files often executes pickle
  and depends on PyTorch-specific module code. TensorStudio should interoperate
  through SafeTensors, ONNX, or explicit state conversion instead.
- TensorFlow SavedModel, Keras `.keras`, HDF5, and TensorFlow Lite execution:
  these formats encode TensorFlow/Keras runtime semantics that TensorStudio
  does not implement yet. TensorStudio inspects them safely and rejects
  execution/import claims.
- DLPack capsules: useful after TensorStudio has mature device semantics and
  non-CPU storage ownership rules.

## Unsupported Formats

TensorStudio does not claim compatibility with every neural-network file format.
Formats that require executing arbitrary Python code, depending on another
framework runtime, or supporting dynamic graph features should be rejected
clearly rather than partially loaded.
