# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.5.1`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `21`
- TensorStudio losses versus NumPy: `82`
- TensorStudio wins versus JAX CPU dispatch: `84`
- TensorStudio losses versus JAX CPU dispatch: `14`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0046 | 0.0046 | 0.0044 | 0.0049 | 0.0002 | 0.1463 | n/a | n/a | 2.5621 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.1463 | n/a | n/a | 2.5621 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0118 | 0.0121 | 0.0116 | 0.0133 | 0.0006 | 0.1463 | n/a | n/a | 2.5621 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0026 | 0.0027 | 0.0025 | 0.0031 | 0.0002 | 0.2657 | n/a | n/a | 4.7203 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2657 | n/a | n/a | 4.7203 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0124 | 0.0129 | 0.0116 | 0.0157 | 0.0015 | 0.2657 | n/a | n/a | 4.7203 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0025 | 0.0027 | 0.0024 | 0.0034 | 0.0004 | 0.2737 | n/a | n/a | 4.6423 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2737 | n/a | n/a | 4.6423 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0116 | 0.0120 | 0.0113 | 0.0129 | 0.0007 | 0.2737 | n/a | n/a | 4.6423 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0026 | 0.0026 | 0.0025 | 0.0028 | 0.0002 | 0.2761 | n/a | n/a | 3.4863 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2761 | n/a | n/a | 3.4863 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0089 | 0.0089 | 0.0087 | 0.0093 | 0.0002 | 0.2761 | n/a | n/a | 3.4863 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0097 | 0.0098 | 0.0096 | 0.0100 | 0.0002 | 0.5552 | n/a | n/a | 8.9392 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0054 | 0.0054 | 0.0053 | 0.0055 | 0.0001 | 0.5552 | n/a | n/a | 8.9392 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0866 | 0.0879 | 0.0853 | 0.0946 | 0.0034 | 0.5552 | n/a | n/a | 8.9392 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0026 | 0.0026 | 0.0025 | 0.0028 | 0.0001 | 0.2500 | n/a | n/a | 4.5588 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0007 | 0.0000 | 0.2500 | n/a | n/a | 4.5588 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0117 | 0.0135 | 0.0112 | 0.0174 | 0.0026 | 0.2500 | n/a | n/a | 4.5588 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0026 | 0.0027 | 0.0025 | 0.0029 | 0.0001 | 0.2685 | n/a | n/a | 4.3732 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2685 | n/a | n/a | 4.3732 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0114 | 0.0115 | 0.0109 | 0.0122 | 0.0004 | 0.2685 | n/a | n/a | 4.3732 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0026 | 0.0026 | 0.0025 | 0.0028 | 0.0001 | 0.2685 | n/a | n/a | 4.3188 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2685 | n/a | n/a | 4.3188 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0114 | 0.0114 | 0.0112 | 0.0116 | 0.0002 | 0.2685 | n/a | n/a | 4.3188 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0025 | 0.0025 | 0.0025 | 0.0026 | 0.0000 | 0.2907 | n/a | n/a | 3.4709 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0013 | 0.0002 | 0.2907 | n/a | n/a | 3.4709 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0088 | 0.0088 | 0.0086 | 0.0090 | 0.0001 | 0.2907 | n/a | n/a | 3.4709 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0096 | 0.0096 | 0.0096 | 0.0097 | 0.0001 | 0.5616 | n/a | n/a | 9.0605 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0054 | 0.0055 | 0.0052 | 0.0061 | 0.0003 | 0.5616 | n/a | n/a | 9.0605 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0872 | 0.0877 | 0.0853 | 0.0922 | 0.0024 | 0.5616 | n/a | n/a | 9.0605 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0027 | 0.0028 | 0.0026 | 0.0030 | 0.0002 | 0.2932 | n/a | n/a | 4.6597 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2932 | n/a | n/a | 4.6597 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0125 | 0.0126 | 0.0114 | 0.0139 | 0.0009 | 0.2932 | n/a | n/a | 4.6597 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.2489 | n/a | n/a | 4.3298 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2489 | n/a | n/a | 4.3298 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0118 | 0.0119 | 0.0115 | 0.0122 | 0.0002 | 0.2489 | n/a | n/a | 4.3298 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0028 | 0.0028 | 0.0026 | 0.0030 | 0.0001 | 0.2489 | n/a | n/a | 4.3641 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2489 | n/a | n/a | 4.3641 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0120 | 0.0124 | 0.0118 | 0.0138 | 0.0007 | 0.2489 | n/a | n/a | 4.3641 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0026 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.2810 | n/a | n/a | 3.3462 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2810 | n/a | n/a | 3.3462 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0088 | 0.0088 | 0.0087 | 0.0091 | 0.0002 | 0.2810 | n/a | n/a | 3.3462 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0097 | 0.0096 | 0.0096 | 0.0097 | 0.0000 | 0.5735 | n/a | n/a | 9.0016 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0055 | 0.0058 | 0.0054 | 0.0068 | 0.0005 | 0.5735 | n/a | n/a | 9.0016 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0870 | 0.0872 | 0.0857 | 0.0895 | 0.0013 | 0.5735 | n/a | n/a | 9.0016 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.2499 | n/a | n/a | 4.1616 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0009 | 0.0001 | 0.2499 | n/a | n/a | 4.1616 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0114 | 0.0115 | 0.0113 | 0.0117 | 0.0002 | 0.2499 | n/a | n/a | 4.1616 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0027 | 0.0000 | 0.2668 | n/a | n/a | 4.3693 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2668 | n/a | n/a | 4.3693 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0116 | 0.0116 | 0.0112 | 0.0118 | 0.0002 | 0.2668 | n/a | n/a | 4.3693 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0000 | 0.2675 | n/a | n/a | 4.5237 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2675 | n/a | n/a | 4.5237 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0118 | 0.0118 | 0.0117 | 0.0120 | 0.0001 | 0.2675 | n/a | n/a | 4.5237 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0028 | 0.0030 | 0.0026 | 0.0040 | 0.0005 | 0.2712 | n/a | n/a | 3.1160 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0008 | 0.0009 | 0.0007 | 0.0014 | 0.0002 | 0.2712 | n/a | n/a | 3.1160 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0088 | 0.0098 | 0.0086 | 0.0140 | 0.0021 | 0.2712 | n/a | n/a | 3.1160 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0101 | 0.0101 | 0.0100 | 0.0103 | 0.0001 | 0.5357 | n/a | n/a | 8.7850 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0054 | 0.0054 | 0.0054 | 0.0055 | 0.0000 | 0.5357 | n/a | n/a | 8.7850 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0889 | 0.0907 | 0.0872 | 0.0963 | 0.0032 | 0.5357 | n/a | n/a | 8.7850 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0032 | 0.0031 | 0.0030 | 0.0032 | 0.0001 | 0.3721 | n/a | n/a | 4.5073 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0012 | 0.0011 | 0.0009 | 0.0012 | 0.0001 | 0.3721 | n/a | n/a | 4.5073 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0142 | 0.0149 | 0.0123 | 0.0192 | 0.0024 | 0.3721 | n/a | n/a | 4.5073 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0000 | 0.3246 | n/a | n/a | 4.5470 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3246 | n/a | n/a | 4.5470 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0137 | 0.0142 | 0.0116 | 0.0176 | 0.0020 | 0.3246 | n/a | n/a | 4.5470 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0031 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 0.3336 | n/a | n/a | 4.8682 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3336 | n/a | n/a | 4.8682 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0149 | 0.0146 | 0.0125 | 0.0169 | 0.0017 | 0.3336 | n/a | n/a | 4.8682 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0032 | 0.0033 | 0.0032 | 0.0038 | 0.0002 | 0.6986 | n/a | n/a | 3.1039 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0022 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.6986 | n/a | n/a | 3.1039 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0100 | 0.0105 | 0.0097 | 0.0128 | 0.0012 | 0.6986 | n/a | n/a | 3.1039 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0119 | 0.0130 | 0.0118 | 0.0171 | 0.0021 | 0.6389 | n/a | n/a | 11.5433 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0076 | 0.0075 | 0.0072 | 0.0077 | 0.0002 | 0.6389 | n/a | n/a | 11.5433 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1371 | 0.1317 | 0.1147 | 0.1389 | 0.0089 | 0.6389 | n/a | n/a | 11.5433 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0065 | 0.0075 | 0.0049 | 0.0130 | 0.0029 | 0.4728 | n/a | n/a | 3.2641 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0031 | 0.0030 | 0.0026 | 0.0035 | 0.0003 | 0.4728 | n/a | n/a | 3.2641 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0212 | 0.0228 | 0.0160 | 0.0296 | 0.0049 | 0.4728 | n/a | n/a | 3.2641 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0049 | 0.0050 | 0.0048 | 0.0054 | 0.0002 | 0.3232 | n/a | n/a | 3.1234 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.3232 | n/a | n/a | 3.1234 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0153 | 0.0160 | 0.0130 | 0.0204 | 0.0028 | 0.3232 | n/a | n/a | 3.1234 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0049 | 0.0049 | 0.0047 | 0.0050 | 0.0001 | 0.3132 | n/a | n/a | 2.8687 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.3132 | n/a | n/a | 2.8687 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0141 | 0.0140 | 0.0126 | 0.0155 | 0.0010 | 0.3132 | n/a | n/a | 2.8687 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0057 | 0.0057 | 0.0056 | 0.0061 | 0.0002 | 0.2883 | n/a | n/a | 1.8886 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.2883 | n/a | n/a | 1.8886 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0107 | 0.0112 | 0.0101 | 0.0136 | 0.0012 | 0.2883 | n/a | n/a | 1.8886 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0191 | 0.0192 | 0.0189 | 0.0195 | 0.0003 | 0.5461 | n/a | n/a | 5.8840 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0104 | 0.0106 | 0.0102 | 0.0117 | 0.0005 | 0.5461 | n/a | n/a | 5.8840 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1121 | 0.1135 | 0.1055 | 0.1250 | 0.0068 | 0.5461 | n/a | n/a | 5.8840 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0127 | 0.0133 | 0.0123 | 0.0160 | 0.0014 | 0.2819 | n/a | n/a | 1.0329 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0000 | 0.2819 | n/a | n/a | 1.0329 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0132 | 0.0156 | 0.0126 | 0.0239 | 0.0043 | 0.2819 | n/a | n/a | 1.0329 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0095 | 0.0095 | 0.0095 | 0.0096 | 0.0000 | 0.3980 | n/a | n/a | 1.3705 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0038 | 0.0038 | 0.0000 | 0.3980 | n/a | n/a | 1.3705 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0131 | 0.0139 | 0.0125 | 0.0184 | 0.0022 | 0.3980 | n/a | n/a | 1.3705 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0095 | 0.0096 | 0.0095 | 0.0096 | 0.0000 | 0.3896 | n/a | n/a | 1.8563 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0038 | 0.0000 | 0.3896 | n/a | n/a | 1.8563 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0177 | 0.0169 | 0.0129 | 0.0218 | 0.0035 | 0.3896 | n/a | n/a | 1.8563 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0132 | 0.0132 | 0.0131 | 0.0132 | 0.0000 | 0.3201 | n/a | n/a | 0.8437 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0044 | 0.0042 | 0.0050 | 0.0003 | 0.3201 | n/a | n/a | 0.8437 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0111 | 0.0113 | 0.0111 | 0.0118 | 0.0003 | 0.3201 | n/a | n/a | 0.8437 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0431 | 0.0433 | 0.0428 | 0.0442 | 0.0005 | 0.4885 | n/a | n/a | 3.3933 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0210 | 0.0213 | 0.0210 | 0.0220 | 0.0004 | 0.4885 | n/a | n/a | 3.3933 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1462 | 0.1470 | 0.1383 | 0.1534 | 0.0057 | 0.4885 | n/a | n/a | 3.3933 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.9606 | n/a | n/a | 24.5550 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0019 | 0.0001 | 0.9606 | n/a | n/a | 24.5550 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0434 | 0.0438 | 0.0426 | 0.0455 | 0.0010 | 0.9606 | n/a | n/a | 24.5550 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0020 | 0.0001 | 2.4894 | n/a | n/a | 37.6658 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0045 | 0.0045 | 0.0043 | 0.0046 | 0.0001 | 2.4894 | n/a | n/a | 37.6658 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0687 | 0.0697 | 0.0685 | 0.0724 | 0.0015 | 2.4894 | n/a | n/a | 37.6658 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0019 | 0.0000 | 0.6766 | n/a | n/a | 16.3403 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.6766 | n/a | n/a | 16.3403 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0293 | 0.0293 | 0.0286 | 0.0296 | 0.0004 | 0.6766 | n/a | n/a | 16.3403 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 0.6618 | n/a | n/a | 16.1313 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.6618 | n/a | n/a | 16.1313 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0295 | 0.0296 | 0.0288 | 0.0306 | 0.0007 | 0.6618 | n/a | n/a | 16.1313 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 0.6834 | n/a | n/a | 15.8139 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.6834 | n/a | n/a | 15.8139 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0294 | 0.0296 | 0.0287 | 0.0307 | 0.0007 | 0.6834 | n/a | n/a | 15.8139 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.9366 | n/a | n/a | 24.0948 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.9366 | n/a | n/a | 24.0948 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0442 | 0.0442 | 0.0425 | 0.0461 | 0.0013 | 0.9366 | n/a | n/a | 24.0948 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0019 | 0.0000 | 2.4066 | n/a | n/a | 38.4883 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0045 | 0.0045 | 0.0043 | 0.0045 | 0.0001 | 2.4066 | n/a | n/a | 38.4883 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0717 | 0.0718 | 0.0709 | 0.0725 | 0.0006 | 2.4066 | n/a | n/a | 38.4883 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 0.6357 | n/a | n/a | 15.1449 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.6357 | n/a | n/a | 15.1449 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0285 | 0.0286 | 0.0283 | 0.0293 | 0.0003 | 0.6357 | n/a | n/a | 15.1449 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0019 | 0.0020 | 0.0019 | 0.0022 | 0.0001 | 0.6618 | n/a | n/a | 15.3268 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.6618 | n/a | n/a | 15.3268 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0298 | 0.0297 | 0.0290 | 0.0304 | 0.0006 | 0.6618 | n/a | n/a | 15.3268 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0020 | 0.0021 | 0.0020 | 0.0024 | 0.0002 | 0.6648 | n/a | n/a | 14.3260 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0018 | 0.0012 | 0.0027 | 0.0006 | 0.6648 | n/a | n/a | 14.3260 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0290 | 0.0290 | 0.0283 | 0.0298 | 0.0005 | 0.6648 | n/a | n/a | 14.3260 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0017 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 0.9603 | n/a | n/a | 24.9638 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.9603 | n/a | n/a | 24.9638 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0434 | 0.0437 | 0.0430 | 0.0451 | 0.0008 | 0.9603 | n/a | n/a | 24.9638 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0000 | 2.2200 | n/a | n/a | 34.2413 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0045 | 0.0045 | 0.0044 | 0.0048 | 0.0002 | 2.2200 | n/a | n/a | 34.2413 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0693 | 0.0701 | 0.0688 | 0.0726 | 0.0014 | 2.2200 | n/a | n/a | 34.2413 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0021 | 0.0022 | 0.0021 | 0.0022 | 0.0001 | 0.6005 | n/a | n/a | 13.7565 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.6005 | n/a | n/a | 13.7565 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0292 | 0.0294 | 0.0289 | 0.0304 | 0.0005 | 0.6005 | n/a | n/a | 13.7565 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0020 | 0.0021 | 0.0020 | 0.0023 | 0.0001 | 0.5972 | n/a | n/a | 14.4123 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.5972 | n/a | n/a | 14.4123 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0294 | 0.0295 | 0.0291 | 0.0303 | 0.0004 | 0.5972 | n/a | n/a | 14.4123 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0000 | 0.6403 | n/a | n/a | 14.4213 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0015 | 0.0001 | 0.6403 | n/a | n/a | 14.4213 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0293 | 0.0294 | 0.0287 | 0.0300 | 0.0004 | 0.6403 | n/a | n/a | 14.4213 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0020 | 0.0000 | 0.9287 | n/a | n/a | 23.6020 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 0.9287 | n/a | n/a | 23.6020 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0444 | 0.0493 | 0.0434 | 0.0699 | 0.0103 | 0.9287 | n/a | n/a | 23.6020 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0029 | 0.0029 | 0.0028 | 0.0030 | 0.0001 | 1.6456 | n/a | n/a | 24.7498 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0048 | 0.0047 | 0.0045 | 0.0049 | 0.0002 | 1.6456 | n/a | n/a | 24.7498 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0723 | 0.0715 | 0.0701 | 0.0725 | 0.0011 | 1.6456 | n/a | n/a | 24.7498 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0031 | 0.0031 | 0.0030 | 0.0031 | 0.0000 | 0.4739 | n/a | n/a | 10.0428 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0017 | 0.0001 | 0.4739 | n/a | n/a | 10.0428 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0310 | 0.0308 | 0.0285 | 0.0339 | 0.0019 | 0.4739 | n/a | n/a | 10.0428 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0027 | 0.0000 | 0.5096 | n/a | n/a | 11.1536 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.5096 | n/a | n/a | 11.1536 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0296 | 0.0296 | 0.0293 | 0.0298 | 0.0002 | 0.5096 | n/a | n/a | 11.1536 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0027 | 0.0028 | 0.0027 | 0.0031 | 0.0001 | 0.5246 | n/a | n/a | 10.6159 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.5246 | n/a | n/a | 10.6159 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0288 | 0.0288 | 0.0284 | 0.0294 | 0.0004 | 0.5246 | n/a | n/a | 10.6159 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0026 | 0.0000 | 0.9419 | n/a | n/a | 23.6470 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0024 | 0.0023 | 0.0026 | 0.0001 | 0.9419 | n/a | n/a | 23.6470 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0580 | 0.0605 | 0.0535 | 0.0710 | 0.0063 | 0.9419 | n/a | n/a | 23.6470 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0093 | 0.0093 | 0.0092 | 0.0094 | 0.0001 | 0.7352 | n/a | n/a | 9.4155 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0068 | 0.0069 | 0.0068 | 0.0070 | 0.0001 | 0.7352 | n/a | n/a | 9.4155 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0875 | 0.0891 | 0.0863 | 0.0950 | 0.0033 | 0.7352 | n/a | n/a | 9.4155 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0136 | 0.0136 | 0.0133 | 0.0137 | 0.0001 | 0.2726 | n/a | n/a | 2.1402 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0038 | 0.0000 | 0.2726 | n/a | n/a | 2.1402 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0291 | 0.0293 | 0.0287 | 0.0303 | 0.0005 | 0.2726 | n/a | n/a | 2.1402 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0084 | 0.0084 | 0.0082 | 0.0087 | 0.0002 | 0.3102 | n/a | n/a | 3.4620 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0026 | 0.0000 | 0.3102 | n/a | n/a | 3.4620 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0289 | 0.0290 | 0.0288 | 0.0293 | 0.0002 | 0.3102 | n/a | n/a | 3.4620 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0088 | 0.0088 | 0.0086 | 0.0089 | 0.0001 | 0.3289 | n/a | n/a | 3.3826 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0028 | 0.0030 | 0.0000 | 0.3289 | n/a | n/a | 3.3826 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0298 | 0.0299 | 0.0293 | 0.0306 | 0.0004 | 0.3289 | n/a | n/a | 3.3826 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0050 | 0.0050 | 0.0050 | 0.0052 | 0.0001 | 0.7380 | n/a | n/a | 11.5708 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0037 | 0.0037 | 0.0034 | 0.0039 | 0.0002 | 0.7380 | n/a | n/a | 11.5708 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0574 | 0.0603 | 0.0532 | 0.0689 | 0.0062 | 0.7380 | n/a | n/a | 11.5708 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0323 | 0.0324 | 0.0322 | 0.0327 | 0.0002 | 0.3651 | n/a | n/a | 3.1205 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0118 | 0.0118 | 0.0116 | 0.0119 | 0.0001 | 0.3651 | n/a | n/a | 3.1205 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1008 | 0.1014 | 0.0952 | 0.1115 | 0.0055 | 0.3651 | n/a | n/a | 3.1205 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0582 | 0.0584 | 0.0580 | 0.0589 | 0.0004 | 0.1876 | n/a | n/a | 0.5549 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0109 | 0.0110 | 0.0108 | 0.0111 | 0.0001 | 0.1876 | n/a | n/a | 0.5549 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0323 | 0.0323 | 0.0319 | 0.0329 | 0.0003 | 0.1876 | n/a | n/a | 0.5549 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0281 | 0.0281 | 0.0275 | 0.0284 | 0.0003 | 0.2129 | n/a | n/a | 1.1640 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0060 | 0.0060 | 0.0059 | 0.0062 | 0.0001 | 0.2129 | n/a | n/a | 1.1640 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0327 | 0.0357 | 0.0323 | 0.0424 | 0.0041 | 0.2129 | n/a | n/a | 1.1640 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0300 | 0.0299 | 0.0296 | 0.0300 | 0.0002 | 0.2505 | n/a | n/a | 1.1738 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0075 | 0.0075 | 0.0073 | 0.0077 | 0.0001 | 0.2505 | n/a | n/a | 1.1738 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0352 | 0.0353 | 0.0335 | 0.0387 | 0.0018 | 0.2505 | n/a | n/a | 1.1738 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0156 | 0.0157 | 0.0156 | 0.0162 | 0.0002 | 0.5540 | n/a | n/a | 5.5171 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0087 | 0.0088 | 0.0087 | 0.0091 | 0.0002 | 0.5540 | n/a | n/a | 5.5171 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0863 | 0.0847 | 0.0683 | 0.0991 | 0.0103 | 0.5540 | n/a | n/a | 5.5171 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1208 | 0.1214 | 0.1205 | 0.1241 | 0.0014 | 0.2716 | n/a | n/a | 1.0294 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0328 | 0.0334 | 0.0327 | 0.0353 | 0.0010 | 0.2716 | n/a | n/a | 1.0294 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1244 | 0.1243 | 0.1011 | 0.1538 | 0.0173 | 0.2716 | n/a | n/a | 1.0294 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2368 | 0.2382 | 0.2354 | 0.2447 | 0.0034 | 0.1592 | n/a | n/a | 0.1664 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0377 | 0.0380 | 0.0374 | 0.0389 | 0.0006 | 0.1592 | n/a | n/a | 0.1664 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0394 | 0.0389 | 0.0374 | 0.0397 | 0.0009 | 0.1592 | n/a | n/a | 0.1664 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1052 | 0.1052 | 0.1049 | 0.1055 | 0.0002 | 0.1789 | n/a | n/a | 0.3746 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0188 | 0.0188 | 0.0186 | 0.0192 | 0.0002 | 0.1789 | n/a | n/a | 0.3746 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0394 | 0.0397 | 0.0392 | 0.0409 | 0.0006 | 0.1789 | n/a | n/a | 0.3746 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1134 | 0.1130 | 0.1105 | 0.1152 | 0.0019 | 0.2144 | n/a | n/a | 0.3702 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0243 | 0.0246 | 0.0243 | 0.0259 | 0.0006 | 0.2144 | n/a | n/a | 0.3702 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0420 | 0.0425 | 0.0415 | 0.0445 | 0.0012 | 0.2144 | n/a | n/a | 0.3702 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0024 | 0.0001 | 0.9709 | n/a | n/a | 5.3188 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.9709 | n/a | n/a | 5.3188 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0118 | 0.0118 | 0.0115 | 0.0121 | 0.0002 | 0.9709 | n/a | n/a | 5.3188 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0022 | 0.0022 | 0.0021 | 0.0022 | 0.0000 | 3.5155 | n/a | n/a | 4.8989 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0077 | 0.0078 | 0.0077 | 0.0079 | 0.0001 | 3.5155 | n/a | n/a | 4.8989 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0108 | 0.0111 | 0.0106 | 0.0121 | 0.0005 | 3.5155 | n/a | n/a | 4.8989 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.9601 | n/a | n/a | 5.4383 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.9601 | n/a | n/a | 5.4383 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0125 | 0.0125 | 0.0119 | 0.0132 | 0.0004 | 0.9601 | n/a | n/a | 5.4383 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0001 | 3.4584 | n/a | n/a | 5.0478 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0078 | 0.0078 | 0.0075 | 0.0081 | 0.0002 | 3.4584 | n/a | n/a | 5.0478 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0114 | 0.0118 | 0.0111 | 0.0140 | 0.0011 | 3.4584 | n/a | n/a | 5.0478 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 0.9453 | n/a | n/a | 5.0853 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0023 | 0.0021 | 0.0025 | 0.0001 | 0.9453 | n/a | n/a | 5.0853 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0121 | 0.0121 | 0.0118 | 0.0123 | 0.0002 | 0.9453 | n/a | n/a | 5.0853 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 3.2742 | n/a | n/a | 5.6939 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0077 | 0.0078 | 0.0077 | 0.0080 | 0.0001 | 3.2742 | n/a | n/a | 5.6939 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0135 | 0.0164 | 0.0113 | 0.0289 | 0.0065 | 3.2742 | n/a | n/a | 5.6939 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0029 | 0.0036 | 0.0029 | 0.0059 | 0.0011 | 0.7399 | n/a | n/a | 4.6559 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0022 | 0.0000 | 0.7399 | n/a | n/a | 4.6559 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0136 | 0.0135 | 0.0132 | 0.0137 | 0.0002 | 0.7399 | n/a | n/a | 4.6559 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0030 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 2.6167 | n/a | n/a | 4.2260 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0078 | 0.0078 | 0.0077 | 0.0078 | 0.0001 | 2.6167 | n/a | n/a | 4.2260 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0126 | 0.0129 | 0.0124 | 0.0141 | 0.0006 | 2.6167 | n/a | n/a | 4.2260 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0082 | 0.0083 | 0.0081 | 0.0084 | 0.0001 | 0.3090 | n/a | n/a | 1.7267 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0025 | 0.0026 | 0.0025 | 0.0026 | 0.0000 | 0.3090 | n/a | n/a | 1.7267 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0142 | 0.0150 | 0.0135 | 0.0178 | 0.0016 | 0.3090 | n/a | n/a | 1.7267 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0082 | 0.0085 | 0.0081 | 0.0092 | 0.0004 | 1.0007 | n/a | n/a | 1.8594 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0082 | 0.0083 | 0.0081 | 0.0086 | 0.0002 | 1.0007 | n/a | n/a | 1.8594 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0153 | 0.0154 | 0.0126 | 0.0189 | 0.0022 | 1.0007 | n/a | n/a | 1.8594 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0269 | 0.0271 | 0.0266 | 0.0276 | 0.0004 | 0.1248 | n/a | n/a | 0.5840 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0034 | 0.0034 | 0.0033 | 0.0036 | 0.0001 | 0.1248 | n/a | n/a | 0.5840 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0157 | 0.0166 | 0.0151 | 0.0212 | 0.0023 | 0.1248 | n/a | n/a | 0.5840 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0303 | 0.0300 | 0.0263 | 0.0334 | 0.0031 | 0.3103 | n/a | n/a | 0.5122 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0094 | 0.0100 | 0.0093 | 0.0119 | 0.0010 | 0.3103 | n/a | n/a | 0.5122 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0155 | 0.0158 | 0.0149 | 0.0172 | 0.0009 | 0.3103 | n/a | n/a | 0.5122 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1001 | 0.1018 | 0.1000 | 0.1084 | 0.0033 | 0.0664 | n/a | n/a | 0.3348 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0066 | 0.0067 | 0.0066 | 0.0068 | 0.0001 | 0.0664 | n/a | n/a | 0.3348 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0335 | 0.0311 | 0.0211 | 0.0358 | 0.0054 | 0.0664 | n/a | n/a | 0.3348 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1008 | 0.1010 | 0.0996 | 0.1033 | 0.0013 | 0.1348 | n/a | n/a | 0.3345 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0136 | 0.0138 | 0.0130 | 0.0150 | 0.0007 | 0.1348 | n/a | n/a | 0.3345 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0337 | 0.0299 | 0.0182 | 0.0363 | 0.0069 | 0.1348 | n/a | n/a | 0.3345 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 1.0183 | n/a | n/a | 5.0925 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0027 | 0.0028 | 0.0026 | 0.0032 | 0.0002 | 1.0183 | n/a | n/a | 5.0925 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0135 | 0.0135 | 0.0124 | 0.0151 | 0.0010 | 1.0183 | n/a | n/a | 5.0925 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0027 | 0.0028 | 0.0026 | 0.0033 | 0.0003 | 3.8536 | n/a | n/a | 4.4498 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0102 | 0.0106 | 0.0097 | 0.0124 | 0.0009 | 3.8536 | n/a | n/a | 4.4498 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0118 | 0.0120 | 0.0114 | 0.0131 | 0.0006 | 3.8536 | n/a | n/a | 4.4498 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0056 | 0.0058 | 0.0054 | 0.0068 | 0.0005 | 0.7290 | n/a | n/a | 4.8945 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0041 | 0.0042 | 0.0040 | 0.0046 | 0.0002 | 0.7290 | n/a | n/a | 4.8945 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0274 | 0.0240 | 0.0146 | 0.0325 | 0.0076 | 0.7290 | n/a | n/a | 4.8945 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0044 | 0.0045 | 0.0039 | 0.0057 | 0.0006 | 2.7570 | n/a | n/a | 2.9597 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0120 | 0.0132 | 0.0115 | 0.0185 | 0.0027 | 2.7570 | n/a | n/a | 2.9597 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0129 | 0.0132 | 0.0127 | 0.0140 | 0.0005 | 2.7570 | n/a | n/a | 2.9597 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0163 | 0.0173 | 0.0160 | 0.0213 | 0.0020 | 0.4686 | n/a | n/a | 1.7575 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0077 | 0.0084 | 0.0076 | 0.0116 | 0.0016 | 0.4686 | n/a | n/a | 1.7575 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0287 | 0.0277 | 0.0187 | 0.0353 | 0.0054 | 0.4686 | n/a | n/a | 1.7575 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0068 | 0.0068 | 0.0068 | 0.0069 | 0.0000 | 2.1600 | n/a | n/a | 2.4238 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0147 | 0.0153 | 0.0143 | 0.0178 | 0.0013 | 2.1600 | n/a | n/a | 2.4238 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0165 | 0.0171 | 0.0126 | 0.0219 | 0.0041 | 2.1600 | n/a | n/a | 2.4238 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0646 | 0.0646 | 0.0641 | 0.0652 | 0.0004 | 0.3328 | n/a | n/a | 1.1015 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0215 | 0.0222 | 0.0212 | 0.0251 | 0.0015 | 0.3328 | n/a | n/a | 1.1015 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0711 | 0.0704 | 0.0586 | 0.0782 | 0.0074 | 0.3328 | n/a | n/a | 1.1015 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0183 | 0.0188 | 0.0183 | 0.0205 | 0.0009 | 1.1799 | n/a | n/a | 1.3230 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0216 | 0.0224 | 0.0211 | 0.0260 | 0.0018 | 1.1799 | n/a | n/a | 1.3230 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0243 | 0.0263 | 0.0239 | 0.0328 | 0.0034 | 1.1799 | n/a | n/a | 1.3230 | yes | n/a | n/a | yes | TensorStudio | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0056 | 0.0054 | 0.0037 | 0.0069 | 0.0012 | 0.4206 | n/a | n/a | 3.6137 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0024 | 0.0025 | 0.0021 | 0.0030 | 0.0003 | 0.4206 | n/a | n/a | 3.6137 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0203 | 0.0257 | 0.0163 | 0.0405 | 0.0096 | 0.4206 | n/a | n/a | 3.6137 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0632 | 0.0630 | 0.0618 | 0.0636 | 0.0007 | 0.2410 | n/a | n/a | 0.2547 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0152 | 0.0167 | 0.0112 | 0.0268 | 0.0058 | 0.2410 | n/a | n/a | 0.2547 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0161 | 0.0246 | 0.0159 | 0.0583 | 0.0169 | 0.2410 | n/a | n/a | 0.2547 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.2656 | 0.2723 | 0.2550 | 0.3089 | 0.0200 | 1.4666 | n/a | n/a | 0.5691 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3895 | 0.3565 | 0.2553 | 0.4328 | 0.0735 | 1.4666 | n/a | n/a | 0.5691 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1512 | 0.1511 | 0.1250 | 0.1843 | 0.0236 | 1.4666 | n/a | n/a | 0.5691 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.7138 | 2.7048 | 2.2742 | 2.9971 | 0.2427 | 0.1909 | n/a | n/a | 0.1210 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.5182 | 0.5053 | 0.4288 | 0.5589 | 0.0508 | 0.1909 | n/a | n/a | 0.1210 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.3283 | 0.3232 | 0.2753 | 0.3924 | 0.0417 | 0.1909 | n/a | n/a | 0.1210 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1505 | 0.1582 | 0.1426 | 0.1789 | 0.0140 | 8.4519 | n/a | n/a | 0.6564 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2718 | 1.2741 | 1.2369 | 1.3080 | 0.0264 | 8.4519 | n/a | n/a | 0.6564 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.0988 | 0.1030 | 0.0935 | 0.1232 | 0.0106 | 8.4519 | n/a | n/a | 0.6564 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.1059 | 7.0926 | 7.0340 | 7.1305 | 0.0354 | 8.7270 | n/a | n/a | 0.0469 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 62.0132 | 61.8515 | 61.3995 | 62.0491 | 0.2460 | 8.7270 | n/a | n/a | 0.0469 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3335 | 0.3489 | 0.3249 | 0.3945 | 0.0273 | 8.7270 | n/a | n/a | 0.0469 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0135 | 0.0138 | 0.0135 | 0.0145 | 0.0004 | 11.3282 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1532 | 0.1531 | 0.1517 | 0.1545 | 0.0010 | 11.3282 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0134 | 0.0138 | 0.0133 | 0.0154 | 0.0008 | 41.0677 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5499 | 0.5502 | 0.5456 | 0.5561 | 0.0035 | 41.0677 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5329 | 0.5457 | 0.5308 | 0.5920 | 0.0235 | 14.0147 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.4683 | 7.4573 | 7.3658 | 7.5043 | 0.0487 | 14.0147 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5554 | 0.5527 | 0.5249 | 0.5898 | 0.0229 | 49.2709 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 27.3663 | 27.4102 | 26.6733 | 28.2549 | 0.5050 | 49.2709 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0148 | 0.0153 | 0.0143 | 0.0165 | 0.0009 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0316 | 0.0312 | 0.0302 | 0.0320 | 0.0007 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1457 | 0.1451 | 0.1439 | 0.1458 | 0.0008 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3343 | 0.3342 | 0.3317 | 0.3373 | 0.0020 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1318 | 1.1721 | 1.0634 | 1.2921 | 0.0984 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.4474 | 2.4465 | 2.4368 | 2.4596 | 0.0080 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.6089 | 1.6070 | 1.5921 | 1.6237 | 0.0121 | 0.0974 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1567 | 0.1651 | 0.1548 | 0.1902 | 0.0135 | 0.0974 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
