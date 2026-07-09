# Backend Architecture

TensorStudio keeps backend metadata separate from executable tensor storage.
That mirrors the useful part of TensorFlow's public pluggable-device model:
device discovery, kernel capability registration, placement decisions, and
transfer support are different concerns.

## Current Runtime Boundary

The CPU backend is the only executable backend in this release. CUDA, Metal,
and external plugin descriptors can be represented, inspected, and used for
placement diagnostics, but they do not allocate tensors or execute kernels.

The public hardware API exposes:

- `backend_info()` for CPU, CUDA, Metal, and registered plugin descriptors.
- `backend_allocator_info()` for allocator, memory-space, alignment, stream,
  pinned-memory, and unified-memory capability metadata.
- `backend_runtime_info()` for eager execution, graph/compiler boundary,
  stream/event, peer-access, and host-fallback capability metadata.
- `backend_device_properties()` for TensorFlow-style physical device metadata
  such as vendor, architecture, memory fields, precision flags, and unified
  addressing support.
- `logical_device_info()` for the logical devices exposed by available
  physical devices.
- `backend_op_info()` for op-interface metadata such as arity, category,
  differentiability, statefulness, and shape-inference availability.
- `backend_kernel_info()` for native CPU kernels and external kernel metadata.
- `kernel_placement_info()` for explicit op/dtype/device placement answers.
- `backend_execution_plan()` for selected backend, CPU fallback, transfer,
  stream, eager execution, and graph-compatibility diagnostics.
- `device_transfer_info()` for transfer-route diagnostics.
- `storage_telemetry()` for native CPU storage allocation counters.
- `Tensor.to_device()` and `ts.to_device()` for supported tensor movement.
- `current_device()`, `set_default_device()`, `reset_default_device()`, and
  `device_scope()` for thread-local placement policy.
- `register_backend()` and `register_backend_kernel()` for safe plugin
  metadata registration without loading arbitrary native code.
- `include/tensorstudio/kernel_abi.hpp` for the first C-compatible custom
  kernel ABI structs and function signatures.
- `tensorstudio.kernels` manifest helpers for validating JSON/TOML plugin
  metadata without importing modules or loading shared libraries.

## Why Metadata Is Not Execution

Registering an external backend descriptor does not make a tensor executable on
that backend. Real accelerator support still needs:

- device-owned storage
- host/device and device/device copy kernels
- backend dispatch inside tensor operations
- synchronization and stream semantics
- autograd kernels
- CI and benchmark coverage on real hardware

Until those pieces exist, TensorStudio rejects non-CPU allocation and reports
clear unavailable-backend reasons.

The current tensor movement API supports CPU-to-CPU aliasing or cloning:

```python
x = ts.arange(3)
same_storage = x.to_device("cpu")
copied = ts.to_device(x, "cpu", copy=True)

with ts.device_scope("cpu"):
    y = ts.zeros((2, 2))
```

Moving to CUDA, Metal, or a plugin backend raises `DeviceError` until that
backend provides storage ownership and copy kernels.
Setting a CUDA or Metal placement scope is allowed as a descriptor, but tensor
creation inside that scope fails clearly in the current CPU-only build.

The ABI header and manifest loader are intentionally declarative in this
release. They let external projects describe capabilities while TensorStudio
keeps dynamic loading and execution out of the runtime until plugin security,
ownership, and test coverage are stronger.

CMake descriptor flags can mark CUDA or Metal hooks as compiled for diagnostics
(`TENSORSTUDIO_ENABLE_CUDA_BACKEND_DESCRIPTOR` and
`TENSORSTUDIO_ENABLE_METAL_BACKEND_DESCRIPTOR`), but they still do not enable
non-CPU tensor storage or execution.

## Example

```python
import tensorstudio as ts

ts.register_backend("demo_accel", reason="descriptor only")
ts.register_backend_kernel("demo_accel", "add", ["float32"], backward=True)

print(ts.backend_info())
print(ts.backend_allocator_info())
print(ts.backend_runtime_info())
print(ts.backend_device_properties())
print(ts.logical_device_info())
print(ts.backend_op_info("add"))
print(ts.backend_kernel_info("demo_accel:0"))
print(ts.kernel_placement_info("add", "demo_accel:0", "float32"))
print(ts.backend_execution_plan("add", "demo_accel:0", "float32"))
print(ts.storage_telemetry())

ts.unregister_backend("demo_accel")
```

Manifest discovery is metadata-only:

```python
import tensorstudio as ts

for manifest in ts.discover_kernel_manifests("plugins"):
    ts.register_kernel_manifest(manifest)
```

Supported manifest names are `tensorstudio-kernel.json`,
`tensorstudio-kernel.toml`, `tensorstudio_plugin.json`, and
`tensorstudio_plugin.toml`.

Kernel manifest entries may include:

- `op`
- `dtypes`
- `memory_space`: `host`, `device`, or `unified`
- `execution_mode`: `sync` or `async`
- `forward`
- `backward`
- `available`
- `deterministic`
- `priority`
- `reason`

## TensorFlow-Inspired Split

TensorFlow's public custom-op and pluggable-device model separates op
definitions, device placement, allocators, and per-device kernels. TensorStudio
uses that split as design guidance while remaining much smaller:

- `backend_op_info()` describes the operation contract.
- `backend_kernel_info()` describes backend-specific kernels for that contract.
- `backend_allocator_info()` describes memory ownership and execution limits.
- `backend_runtime_info()` describes executable runtime boundaries, including
  eager execution, graph/compiler support, synchronization primitives, peer
  access, and CPU fallback capability.
- `backend_device_properties()` and `logical_device_info()` separate physical
  device discovery from logical execution devices.
- `kernel_placement_info()` reports direct support and CPU-fallback
  diagnostics.
- `backend_execution_plan()` combines placement, runtime, kernel, and transfer
  metadata into one eager execution answer.
- `kernel_abi.hpp` defines a future C ABI boundary for native plugins.

This is not TensorFlow compatibility. It is a compact backend boundary that can
grow toward executable CUDA, Metal, and audited plugin backends without
pretending those backends already work.

References used for the architecture comparison:

- TensorFlow custom op guide:
  <https://www.tensorflow.org/guide/create_op>
- TensorFlow system paper:
  <https://arxiv.org/abs/1605.08695>
- OpenXLA overview:
  <https://openxla.org/xla>
