# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.8.0`
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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0300 | 0.0286 | 0.0207 | 0.0328 | 0.0041 | 0.0883 | n/a | n/a | 0.6347 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0026 | 0.0030 | 0.0023 | 0.0046 | 0.0008 | 0.0883 | n/a | n/a | 0.6347 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0191 | 0.0180 | 0.0116 | 0.0221 | 0.0039 | 0.0883 | n/a | n/a | 0.6347 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1053 | 0.1136 | 0.1021 | 0.1345 | 0.0124 | 0.1019 | n/a | n/a | 0.7163 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0107 | 0.0108 | 0.0107 | 0.0113 | 0.0002 | 0.1019 | n/a | n/a | 0.7163 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0754 | 0.0741 | 0.0197 | 0.1211 | 0.0332 | 0.1019 | n/a | n/a | 0.7163 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3877 | 0.3964 | 0.3855 | 0.4125 | 0.0123 | 0.9563 | n/a | n/a | 0.3096 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3708 | 0.3912 | 0.3620 | 0.4826 | 0.0458 | 0.9563 | n/a | n/a | 0.3096 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1200 | 0.1146 | 0.0958 | 0.1259 | 0.0114 | 0.9563 | n/a | n/a | 0.3096 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.1730 | 2.2938 | 1.9938 | 2.6659 | 0.2631 | 0.2103 | n/a | n/a | 0.1085 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4569 | 0.5410 | 0.4218 | 0.8891 | 0.1754 | 0.2103 | n/a | n/a | 0.1085 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2357 | 0.2590 | 0.2247 | 0.3436 | 0.0437 | 0.2103 | n/a | n/a | 0.1085 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
