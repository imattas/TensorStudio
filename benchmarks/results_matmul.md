# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.11.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0339 | 0.0375 | 0.0298 | 0.0566 | 0.0097 | 0.1334 | n/a | n/a | 0.3237 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0045 | 0.0057 | 0.0044 | 0.0094 | 0.0019 | 0.1334 | n/a | n/a | 0.3237 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0110 | 0.0122 | 0.0108 | 0.0173 | 0.0026 | 0.1334 | n/a | n/a | 0.3237 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1567 | 0.1528 | 0.1083 | 0.2228 | 0.0425 | 0.0782 | n/a | n/a | 0.1175 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0123 | 0.0125 | 0.0122 | 0.0133 | 0.0004 | 0.0782 | n/a | n/a | 0.1175 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0184 | 0.0186 | 0.0170 | 0.0215 | 0.0016 | 0.0782 | n/a | n/a | 0.1175 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4095 | 0.4078 | 0.4010 | 0.4161 | 0.0056 | 0.6347 | n/a | n/a | 0.3749 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2599 | 0.2575 | 0.2294 | 0.2876 | 0.0205 | 0.6347 | n/a | n/a | 0.3749 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1535 | 0.1502 | 0.1077 | 0.1879 | 0.0261 | 0.6347 | n/a | n/a | 0.3749 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 3.0105 | 3.3559 | 2.6853 | 4.3199 | 0.6213 | 0.1256 | n/a | n/a | 0.0794 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.3780 | 0.3812 | 0.3404 | 0.4215 | 0.0266 | 0.1256 | n/a | n/a | 0.0794 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2392 | 0.2352 | 0.1947 | 0.2652 | 0.0240 | 0.1256 | n/a | n/a | 0.0794 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
