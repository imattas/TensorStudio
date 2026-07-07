# Milestones

This page breaks the broad roadmap into release-sized milestones. Exact version
numbers can change, but the order should stay foundation-first.

## Milestone A: Tensor Core Depth

- Fill missing NumPy-style tensor primitives.
- Expand reduction and indexing coverage.
- Improve dtype conversion and promotion tests.
- Add more native C++ kernels for hot paths.
- Completed in `1.7.0`: core math expansion covering stable softmax,
  logsumexp, batched matmul, statistics, boolean reductions, random
  distributions, and a practical `einsum` subset.

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

- Introduce a backend interface.
- Add CPU threading.
- Prototype CUDA only after backend boundaries are clean.
- Explore graph/JIT execution after eager semantics are stable.
