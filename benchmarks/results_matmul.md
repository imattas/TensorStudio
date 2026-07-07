# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.12.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0206 | 0.0212 | 0.0200 | 0.0244 | 0.0016 | 0.1018 | n/a | n/a | 0.5146 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0000 | 0.1018 | n/a | n/a | 0.5146 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0106 | 0.0119 | 0.0098 | 0.0157 | 0.0023 | 0.1018 | n/a | n/a | 0.5146 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1104 | 0.1129 | 0.1013 | 0.1376 | 0.0130 | 0.0955 | n/a | n/a | 0.1539 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0105 | 0.0107 | 0.0105 | 0.0110 | 0.0003 | 0.0955 | n/a | n/a | 0.1539 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0170 | 0.0173 | 0.0166 | 0.0187 | 0.0008 | 0.0955 | n/a | n/a | 0.1539 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4075 | 0.4236 | 0.3913 | 0.4878 | 0.0358 | 0.5939 | n/a | n/a | 0.2170 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2420 | 0.2630 | 0.2330 | 0.3154 | 0.0332 | 0.5939 | n/a | n/a | 0.2170 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.0884 | 0.0943 | 0.0810 | 0.1129 | 0.0131 | 0.5939 | n/a | n/a | 0.2170 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.7452 | 2.7106 | 2.5279 | 2.8506 | 0.1145 | 0.1745 | n/a | n/a | 0.1107 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4791 | 0.4947 | 0.4527 | 0.5591 | 0.0420 | 0.1745 | n/a | n/a | 0.1107 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.3038 | 0.3065 | 0.2338 | 0.4227 | 0.0670 | 0.1745 | n/a | n/a | 0.1107 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
