# Milestones

This page breaks the broad roadmap into release-sized milestones. Exact version
numbers can change, but the order should stay foundation-first.

## Recent Progress

- Native storage allocation telemetry is available for CPU profiling.
- Large contiguous unary, binary, and full-reduction CPU kernels have a
  conservative native thread dispatcher.
- ONNX Runtime can execute compatible exported static graphs through the
  optional `tensorstudio[onnxruntime]` extra.
- Safe model-format inspection now covers ONNX, Keras archives, SavedModel
  directories, HDF5/Keras weight files, and TFLite flatbuffer identifiers.
- Vision datasets can be indexed with checksummed image-folder manifests.

## Milestone A: Tensor Core Depth

- Fill missing NumPy-style tensor primitives.
- Expand reduction and indexing coverage.
- Improve dtype conversion and promotion tests.
- Add more native C++ kernels for hot paths.

## Milestone B: Autograd Coverage

- Add more gradient formulas.
- Add gradient checking helpers.
- Reduce temporary tensors in backward passes.
- Document unsupported gradients per op.

## Milestone C: Neural Network Breadth

- Add normalization layers.
- Add embedding-style layers when indexing support is ready.
- Add more initialization helpers.
- Improve module serialization and state migration.

## Milestone D: Vision Workloads

- Expand image transforms.
- Add dataset metadata helpers.
- Add more CNN building blocks.
- Benchmark convolution and pooling improvements.

## Milestone E: Interchange

- Improve ONNX export coverage.
- Add ONNX metadata extraction where reliable.
- Add import for supported operator subsets.
- Reject unsupported graphs clearly.

## Milestone F: Hardware And Scale

- Extend the backend metadata interface into executable storage, streams, and
  dispatch once the safety boundary is proven.
- Add CPU threading.
- Prototype CUDA only after backend boundaries are clean.
- Explore graph/JIT execution after eager semantics are stable.
