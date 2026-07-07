# TensorStudio

TensorStudio is a compact C++ tensor and autograd engine with a Python API for
learning, experimentation, and lightweight ML workloads.

`1.11.0` is a CPU-only stable API foundation for the tensor, autograd,
neural-network, optimizer, data, project, serialization, ONNX export, vision,
docs, packaging, and wheel workflows.

## Status

TensorStudio is:

- CPU-only.
- Eager-only.
- Built around a native `tensorstudio._C` extension.
- Suitable for learning, experimentation, and lightweight ML workloads.
- Not intended to replace or broadly outperform mature production ML frameworks.

## What Is Included

- C++20 tensor storage with shared storage, shape, strides, offset, dtype,
  device, and autograd metadata.
- Dtypes: `float32`, `float64`, `int32`, `int64`, and `bool`.
- Tensor creation helpers: `tensor`, `from_numpy`, `zeros`, `ones`, `full`,
  `empty`, `rand`, `randn`, `uniform`, `normal`, `randint`, `bernoulli`,
  `arange`, `eye`, `linspace`, and matching `*_like` helpers where useful.
- NumPy-style broadcasting for binary elementwise operations.
- Arithmetic, comparison helpers, `where`, `maximum`, `minimum`, matrix
  multiplication, batched matrix multiplication, all-element, single-axis, and
  tuple-axis reductions, arg reductions, boolean reductions, statistical
  reductions, common activations, stable `softmax`, `log_softmax`,
  `logsumexp`, trigonometric functions, inverse trigonometric functions,
  `log1p`, `sqrt`, `rsqrt`, `abs`, `clamp`/`clip`, CPU NCHW `conv2d`,
  grouped convolution, `conv_transpose2d`, embedding lookup, `max_pool2d`,
  `avg_pool2d`, reshape, flatten, 2D
  transpose, and common integer/slice indexing views.
- Higher-level `tensorstudio.math` helpers for variance, standard deviation,
  norms, stable probability helpers, boolean reductions, a practical `einsum`
  subset, square, and reciprocal.
- Basic dtype casting plus native `concat` and `stack`.
- Reverse-mode autograd for the v1 operation set.
- Autograd graph lifecycle controls including `retain_graph`,
  `clear_history`, `detach_`, leaf introspection, and guarded public in-place
  mutation helpers.
- Python `nn.Module`, parameters, linear, convolution, and pooling layers,
  sequential models, activation modules, dropout, flatten, module introspection,
  and common losses including multiclass cross entropy.
- Python `optim.SGD`, `optim.Adam`, `optim.AdamW`, gradient clipping helpers,
  and small learning-rate schedulers.
- Minimal Windows-friendly `Dataset`, `TensorDataset`, array/image dataset
  factories, deterministic train/validation splitting, metadata summaries, and
  `DataLoader`.
- Project utilities for JSON/TOML/YAML configs, deterministic seeding, run
  folders, reusable train/validation trainers, callbacks, metrics, safe NPZ
  state-dict checkpoints, trusted full checkpoints, resume helpers, and starter
  templates.
- NumPy copy interop, pickle-based internal serialization, non-pickle NPZ
  tensor/state_dict files, limited ONNX export, vision IO, batch transforms,
  image augmentations, detection utilities, segmentation helpers, vision
  datasets, metrics, visualization, compact CNN/UNet helpers, examples, tests,
  benchmarks, and GitHub Actions release workflows.

## Design Goals

TensorStudio prioritizes:

- Clear C++ loops and shape utilities over complex template metaprogramming.
- Honest errors for shape, dtype, device, and autograd issues.
- A Python API that feels familiar without pretending to be PyTorch-compatible.
- Source builds that work with MSVC, GCC, Clang, and Apple Clang.
- Wheels so end users can install without a compiler.

## Boundaries

TensorStudio does not currently include CUDA, Metal, distributed execution,
graph compilation, advanced list/tensor/boolean indexing, sparse tensors, mixed
precision, a high performance kernel library, pretrained vision model zoo,
detection/segmentation training stack, video IO, an ONNX runtime, or ONNX
import. Benchmarks are rough local references only; TensorStudio wins some
small eager CPU cases but is not faster than PyTorch, NumPy, TensorFlow, or JAX
overall.

Pickle serialization is available for trusted internal objects. Prefer
`save_npz`/`load_npz` for portable tensor and state_dict files.

## Next Steps

- Start with [Quickstart](getting-started/quickstart.md).
- Learn tensor semantics in [Tensors](core/tensors.md).
- Understand gradients in [Autograd](autograd/overview.md).
- Build small models with [Neural Networks](nn/overview.md).
- Organize complete runs with [Projects](project/workflows.md).
- Structure project folders with [Project Layout](project/project-layout.md).
- Save and resume models with [Checkpoints](project/checkpoints.md).
- Build image classifiers with [Vision](vision/overview.md).
- Export supported module stacks with [ONNX](interchange/onnx.md).
- Use small datasets with [Data](data/datasets-and-dataloaders.md).
- Batch data predictably with [DataLoader Patterns](data/dataloader-patterns.md).
- Profile runtime behavior with [Profiling](performance/profiling.md).
- Read [Publishing](release/publishing.md) before release work.
- Follow release steps in the [Release Checklist](release/checklist.md).
- Review implementation order in [Roadmap Priorities](roadmap/priorities.md).

## Documentation Map

The docs are organized as multi-page sections:

- `getting-started`: install, quickstart, and source-build notes.
- `core`: tensor semantics, broadcasting, native C++ design, and API reference.
- `autograd`: engine overview, gradient notes, and coverage matrix.
- `nn`: modules, optimizers, and training workflows.
- `data`: dataset contracts, dataset creation, tensor datasets, and DataLoader
  usage.
- `vision`: image utilities, transforms, detection, segmentation, model blocks,
  visualization, and dataset creation.
- `project`: local project structure, workflow helpers, callbacks, metrics,
  configs, templates, and checkpoints.
- `interchange`: NumPy, serialization, trusted pickle notes, ONNX, and formats.
- `performance`: benchmark results, profiling guidance, and optimization notes.
- `hardware`: CPU backend plus platform build notes.
- `development`: contribution guidance and the C++-first policy.
- `release`: publishing, versioning, and release checklist.
- `roadmap`: ordered roadmap, priorities, and release-sized milestones.
