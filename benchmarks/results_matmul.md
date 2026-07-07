# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.9.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0184 | 0.0187 | 0.0183 | 0.0200 | 0.0006 | 0.1227 | n/a | n/a | 0.5309 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0023 | 0.0022 | 0.0019 | 0.0024 | 0.0002 | 0.1227 | n/a | n/a | 0.5309 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0098 | 0.0107 | 0.0096 | 0.0124 | 0.0012 | 0.1227 | n/a | n/a | 0.5309 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1052 | 0.1100 | 0.0966 | 0.1381 | 0.0146 | 0.1018 | n/a | n/a | 0.1538 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0107 | 0.0107 | 0.0107 | 0.0108 | 0.0000 | 0.1018 | n/a | n/a | 0.1538 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0162 | 0.0162 | 0.0159 | 0.0167 | 0.0003 | 0.1018 | n/a | n/a | 0.1538 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3969 | 0.4143 | 0.3927 | 0.4797 | 0.0330 | 0.7789 | n/a | n/a | 0.2822 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3092 | 0.3164 | 0.2978 | 0.3346 | 0.0143 | 0.7789 | n/a | n/a | 0.2822 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1120 | 0.1173 | 0.0894 | 0.1467 | 0.0230 | 0.7789 | n/a | n/a | 0.2822 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.1846 | 2.2059 | 2.0108 | 2.3371 | 0.1180 | 0.1943 | n/a | n/a | 0.1099 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4245 | 0.4278 | 0.4108 | 0.4471 | 0.0155 | 0.1943 | n/a | n/a | 0.1099 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2402 | 0.2364 | 0.2055 | 0.2639 | 0.0192 | 0.1943 | n/a | n/a | 0.1099 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
