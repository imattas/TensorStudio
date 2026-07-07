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

- TensorStudio wins versus NumPy: `4`
- TensorStudio losses versus NumPy: `0`
- TensorStudio wins versus PyTorch CPU: `0`
- TensorStudio losses versus PyTorch CPU: `4`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0138 | 0.0139 | 0.0133 | 0.0152 | 0.0007 | 12.5006 | n/a | 0.4152 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1725 | 0.1862 | 0.1666 | 0.2358 | 0.0256 | 12.5006 | n/a | 0.4152 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | PyTorch CPU | 0.0057 | 0.0065 | 0.0057 | 0.0085 | 0.0011 | 12.5006 | n/a | 0.4152 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0136 | 0.0137 | 0.0132 | 0.0143 | 0.0004 | 46.1156 | n/a | 0.4078 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.6287 | 0.6218 | 0.5475 | 0.7471 | 0.0718 | 46.1156 | n/a | 0.4078 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | PyTorch CPU | 0.0056 | 0.0060 | 0.0043 | 0.0094 | 0.0018 | 46.1156 | n/a | 0.4078 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7450 | 0.7938 | 0.7350 | 0.8843 | 0.0672 | 10.6899 | n/a | 0.0657 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.9638 | 8.0252 | 7.6285 | 8.3240 | 0.2513 | 10.6899 | n/a | 0.0657 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | PyTorch CPU | 0.0489 | 0.0478 | 0.0447 | 0.0509 | 0.0024 | 10.6899 | n/a | 0.0657 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7596 | 0.7704 | 0.6697 | 0.8598 | 0.0628 | 44.6836 | n/a | 0.0251 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 33.9406 | 33.1031 | 28.6286 | 37.9662 | 3.2201 | 44.6836 | n/a | 0.0251 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | PyTorch CPU | 0.0191 | 0.0193 | 0.0134 | 0.0238 | 0.0038 | 44.6836 | n/a | 0.0251 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
