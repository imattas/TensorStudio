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
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0037 | 0.0037 | 0.0037 | 0.0038 | 0.0000 | 0.5152 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 0.5152 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0740 | 0.0743 | 0.0721 | 0.0769 | 0.0021 | 0.1394 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0103 | 0.0106 | 0.0103 | 0.0115 | 0.0005 | 0.1394 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.5220 | 0.5225 | 0.5156 | 0.5321 | 0.0063 | 0.3884 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2027 | 0.1835 | 0.1362 | 0.2216 | 0.0342 | 0.3884 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | TensorStudio | 4.1393 | 4.1426 | 3.9799 | 4.3337 | 0.1304 | 0.1020 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4222 | 0.4184 | 0.3353 | 0.5050 | 0.0544 | 0.1020 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
