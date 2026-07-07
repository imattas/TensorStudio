# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.14.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0203 | 0.0203 | 0.0198 | 0.0209 | 0.0004 | 0.1194 | n/a | n/a | 0.4985 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0024 | 0.0026 | 0.0022 | 0.0034 | 0.0004 | 0.1194 | n/a | n/a | 0.4985 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0101 | 0.0106 | 0.0096 | 0.0121 | 0.0010 | 0.1194 | n/a | n/a | 0.4985 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1149 | 0.1126 | 0.1042 | 0.1206 | 0.0057 | 0.0984 | n/a | n/a | 0.1892 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0113 | 0.0113 | 0.0112 | 0.0113 | 0.0000 | 0.0984 | n/a | n/a | 0.1892 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0217 | 0.0286 | 0.0181 | 0.0597 | 0.0156 | 0.0984 | n/a | n/a | 0.1892 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4135 | 0.4248 | 0.3889 | 0.4903 | 0.0346 | 0.6350 | n/a | n/a | 0.1844 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2625 | 0.2662 | 0.2587 | 0.2813 | 0.0081 | 0.6350 | n/a | n/a | 0.1844 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.0762 | 0.0784 | 0.0757 | 0.0849 | 0.0035 | 0.6350 | n/a | n/a | 0.1844 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 3.1515 | 3.1359 | 2.4330 | 3.9023 | 0.4820 | 0.2028 | n/a | n/a | 0.0708 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.6392 | 0.6218 | 0.5176 | 0.7188 | 0.0718 | 0.2028 | n/a | n/a | 0.0708 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2232 | 0.2388 | 0.2124 | 0.2871 | 0.0275 | 0.2028 | n/a | n/a | 0.0708 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
