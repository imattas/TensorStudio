# Backend Benchmarks

Backend benchmark coverage is included in `benchmark_all.py`:

```bash
python benchmark_all.py --section backends
```

The backend section records:

- `backend_info`
- `available_devices`
- CPU transfer overhead through `to_device("cpu")`
- CUDA availability check overhead

These benchmarks are not accelerator performance claims. They are release
artifacts that prove the backend API is importable, callable, and represented in
the same benchmark report as tensor kernels.

When CUDA or Metal execution lands, this section should expand into separate
device-specific suites for transfers, elementwise kernels, reductions, matmul,
convolution, pooling, and autograd synchronization.
