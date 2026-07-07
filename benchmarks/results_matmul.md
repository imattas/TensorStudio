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

- TensorStudio wins versus NumPy: `0`
- TensorStudio losses versus NumPy: `4`
- TensorStudio wins versus PyTorch CPU: `1`
- TensorStudio losses versus PyTorch CPU: `3`

TensorStudio did not beat NumPy on this machine for the available benchmark set. Performance remains a blocker for broad performance claims.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0033 | 0.0033 | 0.0031 | 0.0037 | 0.0002 | 0.6632 | n/a | 1.5897 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0022 | 0.0021 | 0.0019 | 0.0024 | 0.0002 | 0.6632 | n/a | 1.5897 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | PyTorch CPU | 0.0052 | 0.0052 | 0.0045 | 0.0068 | 0.0008 | 0.6632 | n/a | 1.5897 | n/a | no | n/a | yes | n/a | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0360 | 0.0368 | 0.0339 | 0.0439 | 0.0037 | 0.4481 | n/a | 0.3681 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0161 | 0.0169 | 0.0156 | 0.0194 | 0.0014 | 0.4481 | n/a | 0.3681 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(64, 64)` | PyTorch CPU | 0.0132 | 0.0129 | 0.0108 | 0.0141 | 0.0011 | 0.4481 | n/a | 0.3681 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3792 | 0.4161 | 0.2906 | 0.5590 | 0.1139 | 0.4682 | n/a | 0.0938 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1775 | 0.1787 | 0.1519 | 0.2087 | 0.0188 | 0.4682 | n/a | 0.0938 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(128, 128)` | PyTorch CPU | 0.0356 | 0.0347 | 0.0289 | 0.0385 | 0.0032 | 0.4682 | n/a | 0.0938 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 4.4976 | 4.5300 | 4.2643 | 4.9701 | 0.2543 | 0.0987 | n/a | 0.0360 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4437 | 0.4068 | 0.3054 | 0.4530 | 0.0567 | 0.0987 | n/a | 0.0360 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(256, 256)` | PyTorch CPU | 0.1617 | 0.2005 | 0.1373 | 0.3023 | 0.0677 | 0.0987 | n/a | 0.0360 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
