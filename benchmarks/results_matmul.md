# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.10.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0228 | 0.0237 | 0.0211 | 0.0293 | 0.0028 | 0.1167 | n/a | n/a | 0.6896 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0027 | 0.0027 | 0.0022 | 0.0035 | 0.0004 | 0.1167 | n/a | n/a | 0.6896 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0157 | 0.0159 | 0.0132 | 0.0199 | 0.0022 | 0.1167 | n/a | n/a | 0.6896 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1215 | 0.1227 | 0.1115 | 0.1361 | 0.0096 | 0.0935 | n/a | n/a | 0.3487 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0114 | 0.0114 | 0.0113 | 0.0114 | 0.0000 | 0.0935 | n/a | n/a | 0.3487 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0424 | 0.0410 | 0.0161 | 0.0803 | 0.0230 | 0.0935 | n/a | n/a | 0.3487 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4112 | 0.4159 | 0.3893 | 0.4637 | 0.0263 | 0.7613 | n/a | n/a | 0.3065 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3131 | 0.3158 | 0.2995 | 0.3406 | 0.0141 | 0.7613 | n/a | n/a | 0.3065 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1260 | 0.1252 | 0.1178 | 0.1324 | 0.0055 | 0.7613 | n/a | n/a | 0.3065 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.6211 | 2.5425 | 2.2118 | 2.8589 | 0.2399 | 0.1580 | n/a | n/a | 0.1001 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4141 | 0.4476 | 0.3678 | 0.6351 | 0.0954 | 0.1580 | n/a | n/a | 0.1001 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2623 | 0.2523 | 0.2045 | 0.2942 | 0.0321 | 0.1580 | n/a | n/a | 0.1001 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
