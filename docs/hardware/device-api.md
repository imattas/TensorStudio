# Device API

TensorStudio `1.14.0` has a formal device API for CPU, CUDA, Metal, and future
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
