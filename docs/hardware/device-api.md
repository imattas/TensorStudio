# Device API

TensorStudio `2.1.0` has a formal device API for CPU, CUDA, Metal, plugin, and future
backends. The published CPU wheels still execute tensors on CPU only; CUDA and
Metal descriptors exist so code can check availability and fail clearly instead
of silently falling back.

## Device Descriptors

```python
import tensorstudio as ts

cpu = ts.device("cpu")
print(cpu.type)        # cpu
print(cpu.index)       # 0
print(str(cpu))        # cpu
```

Supported descriptor strings:

- `cpu`
- `cuda` or `cuda:0`
- `gpu` as an alias for `cuda`
- `metal` or `metal:0`
- `mps` as an alias for `metal`

`cpu:1` is rejected because the CPU backend is represented as a single logical
device.

## Availability

```python
print(ts.available_devices())
print(ts.backend_info())
print(ts.backend_runtime_info())
print(ts.backend_execution_plan("add", "cpu", "float32", input_devices=["cpu", "cpu"]))
print(ts.cuda_is_available())
print(ts.metal_is_available())
```

The CPU backend reports available in every normal build. CUDA and Metal report
unavailable in the default wheels unless a future accelerator backend is
compiled and enabled.

## Tensor Transfers

```python
x = ts.ones((2, 2), device="cpu")
y = x.to_device("cpu")
z = x.cpu()
```

`Tensor.to("float64")` still performs dtype casting. `Tensor.to("cpu")` or
`Tensor.to(ts.device("cpu"))` performs a device transfer. Transfers to
unavailable devices raise `DeviceError`.

## Factory Device Keyword

Tensor factories accept `device=`:

```python
x = ts.randn((4, 4), device="cpu")
```

Passing `device="cuda"` on a CPU-only build raises a clear error. TensorStudio
does not silently copy back to CPU or pretend a CUDA tensor was created.

## Storage Boundary

Native storage now records its device. The current allocator only permits CPU
storage in published wheels. Accelerator allocation is rejected until the
matching runtime kernels and ownership rules exist.

## Storage Telemetry

```python
ts.reset_storage_telemetry()
x = ts.ones((1024,))
print(ts.storage_telemetry())
```

Telemetry reports cumulative checkouts, active checkouts, active bytes, and peak
active usage for native CPU storage. It is diagnostic data, not a replacement
for a full memory profiler.

## Plugin Backend Descriptors

`register_backend()` and `register_backend_kernel()` register descriptor-only
metadata for external backends. Kernel manifests can be discovered with
`discover_kernel_manifests()` and safely validated without importing Python
modules or loading shared libraries.
