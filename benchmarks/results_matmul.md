# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.13.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0205 | 0.0209 | 0.0200 | 0.0221 | 0.0009 | 0.1015 | n/a | n/a | 0.5857 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0021 | 0.0021 | 0.0020 | 0.0022 | 0.0001 | 0.1015 | n/a | n/a | 0.5857 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0120 | 0.0155 | 0.0104 | 0.0260 | 0.0060 | 0.1015 | n/a | n/a | 0.5857 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1084 | 0.1171 | 0.1018 | 0.1480 | 0.0165 | 0.1771 | n/a | n/a | 0.1498 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0192 | 0.0185 | 0.0167 | 0.0193 | 0.0010 | 0.1771 | n/a | n/a | 0.1498 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0162 | 0.0207 | 0.0160 | 0.0386 | 0.0090 | 0.1771 | n/a | n/a | 0.1498 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4190 | 0.4156 | 0.3720 | 0.4464 | 0.0246 | 0.7100 | n/a | n/a | 0.2348 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2975 | 0.2908 | 0.2764 | 0.3016 | 0.0107 | 0.7100 | n/a | n/a | 0.2348 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.0984 | 0.1026 | 0.0939 | 0.1188 | 0.0091 | 0.7100 | n/a | n/a | 0.2348 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.4819 | 2.5410 | 2.4032 | 2.7732 | 0.1317 | 0.1819 | n/a | n/a | 0.0875 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4515 | 0.4547 | 0.4373 | 0.4796 | 0.0141 | 0.1819 | n/a | n/a | 0.0875 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2170 | 0.2222 | 0.2065 | 0.2546 | 0.0167 | 0.1819 | n/a | n/a | 0.0875 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
