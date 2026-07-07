# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.1`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: available (2.12.1+cpu)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `2`
- TensorStudio losses versus NumPy: `0`
- TensorStudio wins versus PyTorch CPU: `0`
- TensorStudio losses versus PyTorch CPU: `2`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1479 | 0.1485 | 0.1473 | 0.1499 | 0.0010 | 8.2897 | n/a | 0.1619 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2257 | 1.2203 | 1.1894 | 1.2432 | 0.0184 | 8.2897 | n/a | 0.1619 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | PyTorch CPU | 0.0239 | 0.0247 | 0.0206 | 0.0316 | 0.0037 | 8.2897 | n/a | 0.1619 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.5807 | 7.6669 | 7.0996 | 8.6100 | 0.5681 | 7.9867 | n/a | 0.0141 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 60.5450 | 62.6833 | 59.2884 | 70.3396 | 4.0343 | 7.9867 | n/a | 0.0141 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | PyTorch CPU | 0.1069 | 0.1113 | 0.1028 | 0.1338 | 0.0114 | 7.9867 | n/a | 0.0141 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
