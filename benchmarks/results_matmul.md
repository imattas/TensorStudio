# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `2.0.0`
- TensorStudio threads: `12`
- TensorStudio BLAS enabled: `False`
- TensorStudio SIMD level: `sse2`
- TensorStudio storage pool enabled: `True`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `1`
- TensorStudio losses versus NumPy: `3`
- TensorStudio wins versus JAX CPU dispatch: `0`
- TensorStudio losses versus JAX CPU dispatch: `4`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0449 | 0.0437 | 0.0311 | 0.0493 | 0.0066 | 0.1129 | n/a | n/a | 0.6903 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0051 | 0.0051 | 0.0049 | 0.0053 | 0.0001 | 0.1129 | n/a | n/a | 0.6903 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0310 | 0.0313 | 0.0186 | 0.0441 | 0.0088 | 0.1129 | n/a | n/a | 0.6903 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1764 | 0.1893 | 0.1585 | 0.2437 | 0.0311 | 0.1156 | n/a | n/a | 0.3073 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0204 | 0.0194 | 0.0137 | 0.0247 | 0.0036 | 0.1156 | n/a | n/a | 0.3073 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0542 | 0.0553 | 0.0512 | 0.0628 | 0.0043 | 0.1156 | n/a | n/a | 0.3073 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.7503 | 0.8530 | 0.6467 | 1.4460 | 0.3003 | 1.1791 | n/a | n/a | 0.1618 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.8847 | 0.7533 | 0.2599 | 1.2825 | 0.4073 | 1.1791 | n/a | n/a | 0.1618 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1214 | 0.1212 | 0.1089 | 0.1329 | 0.0099 | 1.1791 | n/a | n/a | 0.1618 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 5.1440 | 4.9868 | 4.0472 | 5.9145 | 0.6258 | 0.1759 | n/a | n/a | 0.0781 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.9046 | 1.5236 | 0.7440 | 3.9092 | 1.2064 | 0.1759 | n/a | n/a | 0.0781 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.4019 | 0.4393 | 0.2605 | 0.6641 | 0.1393 | 0.1759 | n/a | n/a | 0.0781 | no | n/a | n/a | no | JAX CPU dispatch | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
