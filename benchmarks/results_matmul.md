# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.0rc2`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `0`
- TensorStudio losses versus NumPy: `4`

TensorStudio did not beat NumPy on this machine for the available benchmark set. Performance remains a blocker for a final `1.0.0` performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0869 | 0.0863 | 0.0848 | 0.0876 | 0.0012 | 0.0227 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0020 | 0.0000 | 0.0227 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | TensorStudio | 4.0006 | 3.9999 | 3.9602 | 4.0581 | 0.0328 | 0.0027 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0107 | 0.0116 | 0.0107 | 0.0149 | 0.0017 | 0.0027 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | TensorStudio | 31.3731 | 31.5163 | 30.7380 | 32.2682 | 0.5995 | 0.0068 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2132 | 0.2213 | 0.1444 | 0.3102 | 0.0627 | 0.0068 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | TensorStudio | 258.7756 | 259.8867 | 256.2724 | 264.3848 | 2.7738 | 0.0016 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4208 | 0.4424 | 0.3846 | 0.5473 | 0.0590 | 0.0016 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
