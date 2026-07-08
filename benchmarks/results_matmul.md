# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.15.0`
- TensorStudio threads: `12`
- TensorStudio BLAS enabled: `False`
- TensorStudio SIMD level: `sse2`
- TensorStudio storage pool enabled: `True`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `0`
- TensorStudio losses versus NumPy: `4`
- TensorStudio wins versus JAX CPU dispatch: `0`
- TensorStudio losses versus JAX CPU dispatch: `4`

TensorStudio did not beat NumPy on this machine for the available benchmark set. Performance remains a blocker for broad performance claims.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0336 | 0.0296 | 0.0215 | 0.0351 | 0.0059 | 0.0652 | n/a | n/a | 0.3234 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.0652 | n/a | n/a | 0.3234 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0109 | 0.0111 | 0.0101 | 0.0122 | 0.0008 | 0.0652 | n/a | n/a | 0.3234 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1289 | 0.1329 | 0.1164 | 0.1621 | 0.0170 | 0.0912 | n/a | n/a | 0.2110 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0117 | 0.0118 | 0.0114 | 0.0123 | 0.0003 | 0.0912 | n/a | n/a | 0.2110 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0272 | 0.0288 | 0.0198 | 0.0450 | 0.0089 | 0.0912 | n/a | n/a | 0.2110 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4180 | 0.4218 | 0.4107 | 0.4340 | 0.0094 | 0.5466 | n/a | n/a | 0.2470 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2285 | 0.2453 | 0.2140 | 0.3093 | 0.0336 | 0.5466 | n/a | n/a | 0.2470 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1032 | 0.1068 | 0.1007 | 0.1171 | 0.0060 | 0.5466 | n/a | n/a | 0.2470 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.7965 | 2.8803 | 2.5834 | 3.3033 | 0.2419 | 0.1457 | n/a | n/a | 0.0720 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4075 | 0.4231 | 0.3811 | 0.4798 | 0.0340 | 0.1457 | n/a | n/a | 0.0720 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2013 | 0.2366 | 0.1797 | 0.3635 | 0.0694 | 0.1457 | n/a | n/a | 0.0720 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
