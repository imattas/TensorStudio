# CUDA Backend

TensorStudio does not ship CUDA execution in the default `1.14.0` wheels. The
release adds explicit CUDA device descriptors, build metadata hooks, CMake
configuration gates, device-aware storage boundaries, and clear runtime errors.

## What Works Today

```python
import tensorstudio as ts

print(ts.device("cuda:0"))
print(ts.cuda_is_available())
print(ts.backend_info())
```

On the published CPU wheels, `cuda_is_available()` returns `False` and tensor
allocation or transfer to CUDA raises `DeviceError`.

## Build Hooks

CMake exposes:

```bash
cmake -DTENSORSTUDIO_ENABLE_CUDA=ON ...
```

If the CUDA toolkit is not found, configuration warns and the wheel remains
CPU-only. If CUDA metadata hooks are compiled, TensorStudio still reports CUDA
runtime kernels unavailable until real CUDA storage, stream ownership, and
kernel implementations are enabled.

## Kernel Roadmap

CUDA kernels must land with tests for:

- elementwise ops and activations
- reductions
- matrix multiplication
- convolution
- pooling
- tensor transfers and synchronization
- autograd correctness against CPU references

Until those tests exist, TensorStudio rejects CUDA tensor execution rather than
claiming partial support.
