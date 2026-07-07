# Model Formats

TensorStudio supports a small set of model interchange formats where behavior
can be implemented honestly and tested.

## Supported Today

TensorStudio state dictionaries:

- `nn.Module.state_dict()` returns flat tensor dictionaries.
- `save_npz` and `load_npz` store those dictionaries without pickle.

Trusted pickle checkpoints:

- `save` and `load` can round-trip Python objects.
- Only load pickle files from trusted sources.

ONNX export:

- Supported TensorStudio module stacks can be exported to ONNX.
- Export is limited to known layers and static shapes.

## Planned In Roadmap Order

- Richer NPZ metadata for model and optimizer state.
- Safe tensor storage formats such as `safetensors`.
- ONNX metadata inspection.
- ONNX import for a tested subset of static graphs.
- Execution of imported ONNX graphs only when TensorStudio has matching native
  ops.

## Unsupported Formats

TensorStudio does not claim compatibility with every neural-network file format.
Formats that require executing arbitrary Python code or unsupported dynamic
graphs should be rejected clearly rather than partially loaded.

