# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.16.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0192 | 0.0199 | 0.0181 | 0.0224 | 0.0017 | 0.0988 | n/a | n/a | 0.8150 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0020 | 0.0001 | 0.0988 | n/a | n/a | 0.8150 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0156 | 0.0172 | 0.0098 | 0.0263 | 0.0057 | 0.0988 | n/a | n/a | 0.8150 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0997 | 0.1075 | 0.0960 | 0.1368 | 0.0154 | 0.1045 | n/a | n/a | 0.2474 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0104 | 0.0105 | 0.0104 | 0.0110 | 0.0002 | 0.1045 | n/a | n/a | 0.2474 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0247 | 0.0277 | 0.0160 | 0.0475 | 0.0114 | 0.1045 | n/a | n/a | 0.2474 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3886 | 0.3917 | 0.3826 | 0.4101 | 0.0099 | 0.9711 | n/a | n/a | 0.2505 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3774 | 0.3775 | 0.3488 | 0.3979 | 0.0162 | 0.9711 | n/a | n/a | 0.2505 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.0974 | 0.0983 | 0.0749 | 0.1178 | 0.0168 | 0.9711 | n/a | n/a | 0.2505 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.1109 | 2.1445 | 1.9735 | 2.3164 | 0.1296 | 0.2285 | n/a | n/a | 0.1209 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4823 | 0.4609 | 0.4224 | 0.4847 | 0.0277 | 0.2285 | n/a | n/a | 0.1209 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2552 | 0.2780 | 0.2473 | 0.3417 | 0.0354 | 0.2285 | n/a | n/a | 0.1209 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
