# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `2.1.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0457 | 0.0498 | 0.0385 | 0.0725 | 0.0120 | 0.1106 | n/a | n/a | 0.6247 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0051 | 0.0051 | 0.0030 | 0.0073 | 0.0014 | 0.1106 | n/a | n/a | 0.6247 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0285 | 0.0322 | 0.0208 | 0.0535 | 0.0113 | 0.1106 | n/a | n/a | 0.6247 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.2150 | 0.2381 | 0.1639 | 0.3862 | 0.0789 | 0.0590 | n/a | n/a | 0.2873 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0127 | 0.0129 | 0.0126 | 0.0136 | 0.0004 | 0.0590 | n/a | n/a | 0.2873 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0618 | 0.0676 | 0.0547 | 0.0948 | 0.0143 | 0.0590 | n/a | n/a | 0.2873 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.6150 | 0.6589 | 0.5804 | 0.7876 | 0.0756 | 0.9646 | n/a | n/a | 0.3018 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.5932 | 0.5697 | 0.2900 | 0.9141 | 0.2499 | 0.9646 | n/a | n/a | 0.3018 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1856 | 0.2551 | 0.1215 | 0.5507 | 0.1553 | 0.9646 | n/a | n/a | 0.3018 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 3.9020 | 3.9034 | 3.5587 | 4.2645 | 0.2263 | 0.9616 | n/a | n/a | 0.0961 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 3.7523 | 12.6448 | 1.4607 | 42.4295 | 15.4030 | 0.9616 | n/a | n/a | 0.0961 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.3750 | 0.4004 | 0.2984 | 0.5340 | 0.0832 | 0.9616 | n/a | n/a | 0.0961 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
