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
- Completed in `1.8.0`: documented the gradient coverage matrix, added
  retained-graph and graph-release behavior, hardened leaf and in-place
  mutation semantics, and expanded finite-difference tests for the current
  differentiable math surface.

## Milestone C: Neural Network Breadth

- Add normalization layers.
- Add embedding-style layers when indexing support is ready.
- Add more initialization helpers.
- Improve module serialization and state migration.
- Completed in `1.9.0`: added initializers, normalization, embedding,
  grouped/depthwise/1D/transposed convolution, adaptive/global pooling,
  expanded activations, expanded losses, BatchNorm buffers, and model summary
  utilities.

## Milestone D: Training And Project Workflows

- Add dataset creation helpers for arrays, tensors, labels, and images.
- Add train/validation loop support.
- Add callback and resume workflows.
- Add metric helpers for common supervised tasks.
- Completed in `1.10.0`: added array/tensor/image dataset factories,
  deterministic train/validation splitting, metadata summaries, callbacks,
  regression/classification/multilabel metrics, JSON/TOML/YAML config loading,
  checkpoint resume helpers, trainer validation loops, deterministic seeding,
  and starter project templates.

## Milestone E: Vision Workloads

- Expand image transforms.
- Add dataset metadata helpers.
- Add more CNN building blocks.
- Benchmark convolution and pooling improvements.
- Completed in `1.11.0`: added batch-aware image transforms, color jitter,
  random resized crop, rotation, affine transforms, cutout, mixup, CutMix,
  detection utilities, segmentation mask helpers, detection/segmentation folder
  datasets, ResNet-style blocks, MobileNet-style blocks, CompactUNet, and
  prediction/mask/feature-map visualization helpers.

## Milestone F: Interchange

- Improve ONNX export coverage.
- Add ONNX metadata extraction where reliable.
- Add import for supported operator subsets.
- Reject unsupported graphs clearly.

## Milestone G: Hardware And Scale

- Introduce a backend interface.
- Add CPU threading.
- Prototype CUDA only after backend boundaries are clean.
- Explore graph/JIT execution after eager semantics are stable.
