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
- Unsupported operators, multiple graph inputs or outputs, dynamic graph
  features, and asymmetric padding are rejected clearly.

## Practical Format Research

These formats are intentionally supported:

- TensorStudio NPZ for safe local state dictionaries and experiment artifacts.
- SafeTensors for safe tensor-only weight exchange.
- ONNX for static graph exchange when TensorStudio has matching eager tensor
  operations.

These formats are not implemented in `1.14.0`:

- PyTorch `.pt` and `.pth`: loading general PyTorch files often executes pickle
  and depends on PyTorch-specific module code. TensorStudio should interoperate
  through SafeTensors, ONNX, or explicit state conversion instead.
- TensorFlow SavedModel and Keras `.keras`: these formats encode TensorFlow
  runtime semantics that TensorStudio does not implement yet. Future support
  should start with metadata inspection or ONNX-mediated paths, not partial
  execution claims.
- HDF5 weight files: possible for tensor-only arrays, but it adds another
  optional dependency and needs a tested schema before being advertised.
- DLPack capsules: useful after TensorStudio has mature device semantics and
  non-CPU storage ownership rules.

## Unsupported Formats

TensorStudio does not claim compatibility with every neural-network file format.
Formats that require executing arbitrary Python code, depending on another
framework runtime, or supporting dynamic graph features should be rejected
clearly rather than partially loaded.
