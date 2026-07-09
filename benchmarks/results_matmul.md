# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.4.0`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `0`
- TensorStudio losses versus NumPy: `4`
- TensorStudio wins versus JAX CPU dispatch: `1`
- TensorStudio losses versus JAX CPU dispatch: `3`

TensorStudio did not beat NumPy on this machine for the available benchmark set. Performance remains a blocker for broad performance claims.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0045 | 0.0045 | 0.0044 | 0.0045 | 0.0000 | 0.4564 | n/a | n/a | 2.7091 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0000 | 0.4564 | n/a | n/a | 2.7091 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0121 | 0.0144 | 0.0097 | 0.0206 | 0.0049 | 0.4564 | n/a | n/a | 2.7091 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1148 | 0.1200 | 0.1119 | 0.1369 | 0.0092 | 0.1009 | n/a | n/a | 0.1525 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0116 | 0.0117 | 0.0112 | 0.0123 | 0.0004 | 0.1009 | n/a | n/a | 0.1525 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0175 | 0.0195 | 0.0166 | 0.0292 | 0.0048 | 0.1009 | n/a | n/a | 0.1525 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.9039 | 0.9293 | 0.8707 | 1.0019 | 0.0512 | 0.3623 | n/a | n/a | 0.1289 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3275 | 0.3290 | 0.3168 | 0.3483 | 0.0112 | 0.3623 | n/a | n/a | 0.1289 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1165 | 0.1278 | 0.1025 | 0.1817 | 0.0277 | 0.3623 | n/a | n/a | 0.1289 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 7.7507 | 8.0097 | 7.6382 | 8.8978 | 0.4630 | 0.0538 | n/a | n/a | 0.0317 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4174 | 0.4215 | 0.4033 | 0.4480 | 0.0160 | 0.0538 | n/a | n/a | 0.0317 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2454 | 0.2428 | 0.1926 | 0.2977 | 0.0344 | 0.0538 | n/a | n/a | 0.0317 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
