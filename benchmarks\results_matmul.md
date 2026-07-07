# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.0`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: available (2.12.1+cpu)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `0`
- TensorStudio losses versus NumPy: `4`
- TensorStudio wins versus PyTorch CPU: `1`
- TensorStudio losses versus PyTorch CPU: `3`

TensorStudio did not beat NumPy on this machine for the available benchmark set. Performance remains a blocker for broad performance claims.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0000 | 0.6483 | n/a | 1.3297 | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0020 | 0.0019 | 0.0020 | 0.0000 | 0.6483 | n/a | 1.3297 | n/a | NumPy baseline |
| matmul | matmul | `(16, 16)` | PyTorch CPU | 0.0040 | 0.0040 | 0.0040 | 0.0041 | 0.0000 | 0.6483 | n/a | 1.3297 | n/a | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0339 | 0.0341 | 0.0336 | 0.0351 | 0.0005 | 0.3074 | n/a | 0.5671 | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0104 | 0.0105 | 0.0104 | 0.0109 | 0.0002 | 0.3074 | n/a | 0.5671 | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | PyTorch CPU | 0.0192 | 0.0215 | 0.0136 | 0.0349 | 0.0081 | 0.3074 | n/a | 0.5671 | n/a | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4751 | 0.4846 | 0.4662 | 0.5125 | 0.0170 | 0.4157 | n/a | 0.0629 | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1975 | 0.1931 | 0.1666 | 0.2256 | 0.0210 | 0.4157 | n/a | 0.0629 | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | PyTorch CPU | 0.0299 | 0.0306 | 0.0293 | 0.0338 | 0.0017 | 0.4157 | n/a | 0.0629 | n/a | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 3.0180 | 3.4004 | 2.4239 | 4.5115 | 0.8689 | 0.1363 | n/a | 0.0461 | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4115 | 0.4334 | 0.3622 | 0.5482 | 0.0628 | 0.1363 | n/a | 0.0461 | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | PyTorch CPU | 0.1392 | 0.1879 | 0.1264 | 0.2735 | 0.0700 | 0.1363 | n/a | 0.0461 | n/a | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
