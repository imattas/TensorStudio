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

- TensorStudio wins versus NumPy: `8`
- TensorStudio losses versus NumPy: `6`
- TensorStudio wins versus PyTorch CPU: `9`
- TensorStudio losses versus PyTorch CPU: `5`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| reductions | sum | `(1,)` | TensorStudio | 0.0032 | 0.0029 | 0.0017 | 0.0036 | 0.0007 | 1.1229 | n/a | 2.6778 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0036 | 0.0034 | 0.0026 | 0.0040 | 0.0006 | 1.1229 | n/a | 2.6778 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0086 | 0.0083 | 0.0056 | 0.0116 | 0.0022 | 1.1229 | n/a | 2.6778 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0018 | 0.0020 | 0.0016 | 0.0030 | 0.0005 | 6.9508 | n/a | 7.1360 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0122 | 0.0121 | 0.0095 | 0.0151 | 0.0020 | 6.9508 | n/a | 7.1360 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0125 | 0.0128 | 0.0111 | 0.0143 | 0.0011 | 6.9508 | n/a | 7.1360 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 1.1839 | n/a | 4.1154 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0021 | 0.0002 | 1.1839 | n/a | 4.1154 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0059 | 0.0059 | 0.0047 | 0.0076 | 0.0010 | 1.1839 | n/a | 4.1154 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 4.8337 | n/a | 7.9050 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0072 | 0.0080 | 0.0070 | 0.0112 | 0.0016 | 4.8337 | n/a | 7.9050 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0118 | 0.0134 | 0.0114 | 0.0192 | 0.0030 | 4.8337 | n/a | 7.9050 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 1.0772 | n/a | 3.1285 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 1.0772 | n/a | 3.1285 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0049 | 0.0065 | 0.0044 | 0.0093 | 0.0022 | 1.0772 | n/a | 3.1285 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0030 | 0.0026 | 0.0015 | 0.0031 | 0.0006 | 2.8672 | n/a | 3.9111 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0087 | 0.0095 | 0.0070 | 0.0154 | 0.0031 | 2.8672 | n/a | 3.9111 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0118 | 0.0125 | 0.0111 | 0.0142 | 0.0013 | 2.8672 | n/a | 3.9111 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0022 | 0.0001 | 0.8416 | n/a | 2.2674 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0018 | 0.0024 | 0.0017 | 0.0034 | 0.0008 | 0.8416 | n/a | 2.2674 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0048 | 0.0048 | 0.0045 | 0.0054 | 0.0003 | 0.8416 | n/a | 2.2674 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0041 | 0.0042 | 0.0041 | 0.0044 | 0.0002 | 1.8809 | n/a | 3.1911 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0077 | 0.0099 | 0.0071 | 0.0140 | 0.0031 | 1.8809 | n/a | 3.1911 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0131 | 0.0137 | 0.0113 | 0.0179 | 0.0024 | 1.8809 | n/a | 3.1911 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0070 | 0.0071 | 0.0069 | 0.0074 | 0.0002 | 0.2883 | n/a | 0.6542 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0020 | 0.0022 | 0.0020 | 0.0025 | 0.0002 | 0.2883 | n/a | 0.6542 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0046 | 0.0047 | 0.0046 | 0.0049 | 0.0001 | 0.2883 | n/a | 0.6542 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0138 | 0.0138 | 0.0137 | 0.0138 | 0.0001 | 1.1605 | n/a | 1.3475 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0160 | 0.0159 | 0.0150 | 0.0165 | 0.0005 | 1.1605 | n/a | 1.3475 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0186 | 0.0170 | 0.0114 | 0.0221 | 0.0046 | 1.1605 | n/a | 1.3475 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0458 | 0.0420 | 0.0240 | 0.0478 | 0.0091 | 0.0660 | n/a | 0.1094 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0030 | 0.0031 | 0.0029 | 0.0032 | 0.0001 | 0.0660 | n/a | 0.1094 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0050 | 0.0054 | 0.0050 | 0.0063 | 0.0005 | 0.0660 | n/a | 0.1094 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0310 | 0.0333 | 0.0261 | 0.0473 | 0.0073 | 0.3221 | n/a | 0.8233 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0100 | 0.0113 | 0.0093 | 0.0171 | 0.0029 | 0.3221 | n/a | 0.8233 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0256 | 0.0224 | 0.0141 | 0.0259 | 0.0046 | 0.3221 | n/a | 0.8233 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1090 | 0.1129 | 0.0976 | 0.1345 | 0.0121 | 0.0537 | n/a | 0.0681 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0059 | 0.0059 | 0.0058 | 0.0059 | 0.0000 | 0.0537 | n/a | 0.0681 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0074 | 0.0078 | 0.0069 | 0.0091 | 0.0009 | 0.0537 | n/a | 0.0681 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1009 | 0.1026 | 0.0963 | 0.1139 | 0.0060 | 0.2482 | n/a | 0.2913 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0250 | 0.0252 | 0.0250 | 0.0260 | 0.0004 | 0.2482 | n/a | 0.2913 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0294 | 0.0302 | 0.0286 | 0.0331 | 0.0018 | 0.2482 | n/a | 0.2913 | n/a | no | n/a | no | n/a | NumPy | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
