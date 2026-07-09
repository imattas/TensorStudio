# Metal Backend

TensorStudio `1.14.0` adds Metal/MPS device descriptors and CMake metadata hooks
for Apple platforms. The default wheels do not execute tensors on Metal yet.

## What Works Today

```python
import tensorstudio as ts

print(ts.device("metal:0"))
print(ts.device("mps"))
print(ts.metal_is_available())
```

On current wheels, Metal availability is `False` and transfers to Metal raise
`DeviceError`.

## Build Hook

CMake exposes:

```bash
cmake -DTENSORSTUDIO_ENABLE_METAL=ON ...
```

The option is meaningful only on Apple platforms. It marks Metal metadata hooks
as compiled, but TensorStudio still keeps runtime availability false until real
Metal storage and kernels are implemented.

## Research Track

The practical Metal path should cover:

- storage backed by Apple GPU buffers
- command queue and synchronization ownership
- elementwise and reduction kernels
- matmul and convolution strategy
- CPU fallback policy that never hides device mistakes
- benchmark reporting for Apple Silicon
