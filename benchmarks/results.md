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

- TensorStudio wins versus NumPy: `7`
- TensorStudio losses versus NumPy: `96`
- TensorStudio wins versus JAX CPU dispatch: `45`
- TensorStudio losses versus JAX CPU dispatch: `53`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0364 | 0.0392 | 0.0319 | 0.0548 | 0.0081 | 0.0561 | n/a | n/a | 0.8881 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0001 | 0.0561 | n/a | n/a | 0.8881 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0323 | 0.0350 | 0.0267 | 0.0523 | 0.0089 | 0.0561 | n/a | n/a | 0.8881 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0397 | 0.0397 | 0.0348 | 0.0449 | 0.0035 | 0.0512 | n/a | n/a | 0.7214 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0020 | 0.0021 | 0.0015 | 0.0024 | 0.0003 | 0.0512 | n/a | n/a | 0.7214 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0286 | 0.0265 | 0.0223 | 0.0295 | 0.0030 | 0.0512 | n/a | n/a | 0.7214 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0285 | 0.0276 | 0.0251 | 0.0296 | 0.0019 | 0.0764 | n/a | n/a | 0.7796 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0022 | 0.0021 | 0.0015 | 0.0025 | 0.0004 | 0.0764 | n/a | n/a | 0.7796 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0222 | 0.0238 | 0.0213 | 0.0307 | 0.0035 | 0.0764 | n/a | n/a | 0.7796 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0306 | 0.0299 | 0.0264 | 0.0325 | 0.0020 | 0.0432 | n/a | n/a | 0.5716 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0007 | 0.0023 | 0.0005 | 0.0432 | n/a | n/a | 0.5716 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0175 | 0.0182 | 0.0168 | 0.0210 | 0.0016 | 0.0432 | n/a | n/a | 0.5716 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.1601 | 0.1651 | 0.1537 | 0.1873 | 0.0117 | 0.0598 | n/a | n/a | 1.0646 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0096 | 0.0099 | 0.0088 | 0.0123 | 0.0013 | 0.0598 | n/a | n/a | 1.0646 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.1705 | 0.1739 | 0.1455 | 0.2188 | 0.0243 | 0.0598 | n/a | n/a | 1.0646 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0293 | 0.0294 | 0.0258 | 0.0321 | 0.0023 | 0.0663 | n/a | n/a | 0.6813 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0019 | 0.0018 | 0.0010 | 0.0023 | 0.0005 | 0.0663 | n/a | n/a | 0.6813 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0200 | 0.0206 | 0.0181 | 0.0246 | 0.0022 | 0.0663 | n/a | n/a | 0.6813 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0307 | 0.0326 | 0.0294 | 0.0399 | 0.0039 | 0.0565 | n/a | n/a | 0.8585 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0012 | 0.0020 | 0.0003 | 0.0565 | n/a | n/a | 0.8585 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0264 | 0.0283 | 0.0260 | 0.0337 | 0.0029 | 0.0565 | n/a | n/a | 0.8585 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0310 | 0.0317 | 0.0282 | 0.0360 | 0.0028 | 0.0596 | n/a | n/a | 0.8790 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0018 | 0.0019 | 0.0013 | 0.0030 | 0.0006 | 0.0596 | n/a | n/a | 0.8790 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0272 | 0.0265 | 0.0209 | 0.0351 | 0.0052 | 0.0596 | n/a | n/a | 0.8790 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0289 | 0.0291 | 0.0254 | 0.0328 | 0.0024 | 0.0467 | n/a | n/a | 0.5758 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0014 | 0.0015 | 0.0009 | 0.0021 | 0.0004 | 0.0467 | n/a | n/a | 0.5758 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0167 | 0.0164 | 0.0148 | 0.0176 | 0.0009 | 0.0467 | n/a | n/a | 0.5758 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.1565 | 0.1553 | 0.1469 | 0.1617 | 0.0060 | 0.0691 | n/a | n/a | 1.1225 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0108 | 0.0103 | 0.0087 | 0.0116 | 0.0012 | 0.0691 | n/a | n/a | 1.1225 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1757 | 0.1826 | 0.1570 | 0.2254 | 0.0231 | 0.0691 | n/a | n/a | 1.1225 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0294 | 0.0305 | 0.0276 | 0.0339 | 0.0024 | 0.0475 | n/a | n/a | 0.6656 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0010 | 0.0016 | 0.0002 | 0.0475 | n/a | n/a | 0.6656 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0196 | 0.0201 | 0.0181 | 0.0228 | 0.0016 | 0.0475 | n/a | n/a | 0.6656 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0316 | 0.0324 | 0.0297 | 0.0384 | 0.0031 | 0.0593 | n/a | n/a | 0.9492 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0019 | 0.0018 | 0.0013 | 0.0025 | 0.0004 | 0.0593 | n/a | n/a | 0.9492 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0300 | 0.0285 | 0.0215 | 0.0367 | 0.0054 | 0.0593 | n/a | n/a | 0.9492 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0302 | 0.0319 | 0.0283 | 0.0390 | 0.0039 | 0.0427 | n/a | n/a | 0.9049 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0008 | 0.0023 | 0.0005 | 0.0427 | n/a | n/a | 0.9049 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0273 | 0.0269 | 0.0201 | 0.0336 | 0.0049 | 0.0427 | n/a | n/a | 0.9049 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0281 | 0.0289 | 0.0274 | 0.0318 | 0.0016 | 0.0595 | n/a | n/a | 0.6345 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0017 | 0.0016 | 0.0012 | 0.0020 | 0.0003 | 0.0595 | n/a | n/a | 0.6345 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0178 | 0.0191 | 0.0175 | 0.0224 | 0.0019 | 0.0595 | n/a | n/a | 0.6345 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.1561 | 0.1572 | 0.1390 | 0.1694 | 0.0106 | 0.0894 | n/a | n/a | 1.1459 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0140 | 0.0138 | 0.0121 | 0.0158 | 0.0015 | 0.0894 | n/a | n/a | 1.1459 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.1788 | 0.1713 | 0.1540 | 0.1822 | 0.0112 | 0.0894 | n/a | n/a | 1.1459 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0327 | 0.0319 | 0.0281 | 0.0336 | 0.0020 | 0.0456 | n/a | n/a | 0.7171 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0015 | 0.0014 | 0.0008 | 0.0018 | 0.0003 | 0.0456 | n/a | n/a | 0.7171 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0235 | 0.0225 | 0.0187 | 0.0255 | 0.0025 | 0.0456 | n/a | n/a | 0.7171 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0292 | 0.0302 | 0.0263 | 0.0371 | 0.0037 | 0.0523 | n/a | n/a | 0.8312 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0010 | 0.0020 | 0.0004 | 0.0523 | n/a | n/a | 0.8312 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0243 | 0.0243 | 0.0230 | 0.0251 | 0.0008 | 0.0523 | n/a | n/a | 0.8312 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0307 | 0.0329 | 0.0270 | 0.0445 | 0.0061 | 0.0448 | n/a | n/a | 0.8233 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0010 | 0.0018 | 0.0003 | 0.0448 | n/a | n/a | 0.8233 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0252 | 0.0261 | 0.0191 | 0.0337 | 0.0049 | 0.0448 | n/a | n/a | 0.8233 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0263 | 0.0265 | 0.0244 | 0.0296 | 0.0019 | 0.0409 | n/a | n/a | 0.6211 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0011 | 0.0011 | 0.0009 | 0.0015 | 0.0002 | 0.0409 | n/a | n/a | 0.6211 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0163 | 0.0172 | 0.0163 | 0.0186 | 0.0011 | 0.0409 | n/a | n/a | 0.6211 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.1596 | 0.1580 | 0.1401 | 0.1776 | 0.0127 | 0.0621 | n/a | n/a | 1.1892 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0099 | 0.0097 | 0.0088 | 0.0104 | 0.0006 | 0.0621 | n/a | n/a | 1.1892 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.1898 | 0.1846 | 0.1617 | 0.2025 | 0.0144 | 0.0621 | n/a | n/a | 1.1892 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0353 | 0.0403 | 0.0252 | 0.0626 | 0.0151 | 0.0809 | n/a | n/a | 0.8162 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0029 | 0.0033 | 0.0026 | 0.0051 | 0.0009 | 0.0809 | n/a | n/a | 0.8162 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0288 | 0.0305 | 0.0261 | 0.0350 | 0.0037 | 0.0809 | n/a | n/a | 0.8162 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0278 | 0.0282 | 0.0246 | 0.0320 | 0.0024 | 0.0511 | n/a | n/a | 0.9643 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0014 | 0.0017 | 0.0012 | 0.0023 | 0.0005 | 0.0511 | n/a | n/a | 0.9643 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0268 | 0.0275 | 0.0252 | 0.0316 | 0.0023 | 0.0511 | n/a | n/a | 0.9643 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0289 | 0.0278 | 0.0237 | 0.0312 | 0.0027 | 0.0793 | n/a | n/a | 1.0206 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0023 | 0.0024 | 0.0016 | 0.0034 | 0.0007 | 0.0793 | n/a | n/a | 1.0206 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0295 | 0.0282 | 0.0235 | 0.0299 | 0.0024 | 0.0793 | n/a | n/a | 1.0206 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0285 | 0.0291 | 0.0258 | 0.0334 | 0.0025 | 0.0669 | n/a | n/a | 0.9190 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0019 | 0.0021 | 0.0016 | 0.0025 | 0.0004 | 0.0669 | n/a | n/a | 0.9190 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0262 | 0.0264 | 0.0220 | 0.0318 | 0.0032 | 0.0669 | n/a | n/a | 0.9190 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.1594 | 0.1599 | 0.1422 | 0.1880 | 0.0158 | 0.1203 | n/a | n/a | 1.6323 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0192 | 0.0183 | 0.0104 | 0.0240 | 0.0053 | 0.1203 | n/a | n/a | 1.6323 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.2602 | 0.2616 | 0.2233 | 0.2937 | 0.0251 | 0.1203 | n/a | n/a | 1.6323 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0260 | 0.0275 | 0.0251 | 0.0302 | 0.0022 | 0.0870 | n/a | n/a | 1.1575 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0023 | 0.0028 | 0.0020 | 0.0038 | 0.0008 | 0.0870 | n/a | n/a | 1.1575 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0301 | 0.0297 | 0.0261 | 0.0325 | 0.0021 | 0.0870 | n/a | n/a | 1.1575 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0313 | 0.0328 | 0.0268 | 0.0421 | 0.0058 | 0.0801 | n/a | n/a | 0.9144 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0025 | 0.0024 | 0.0021 | 0.0026 | 0.0002 | 0.0801 | n/a | n/a | 0.9144 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0286 | 0.0310 | 0.0268 | 0.0402 | 0.0049 | 0.0801 | n/a | n/a | 0.9144 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0316 | 0.0309 | 0.0266 | 0.0348 | 0.0035 | 0.1105 | n/a | n/a | 1.1032 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0035 | 0.0037 | 0.0031 | 0.0048 | 0.0006 | 0.1105 | n/a | n/a | 1.1032 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0349 | 0.0344 | 0.0315 | 0.0356 | 0.0015 | 0.1105 | n/a | n/a | 1.1032 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0282 | 0.0311 | 0.0256 | 0.0392 | 0.0052 | 0.0903 | n/a | n/a | 1.2679 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0025 | 0.0028 | 0.0023 | 0.0036 | 0.0005 | 0.0903 | n/a | n/a | 1.2679 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0358 | 0.0410 | 0.0298 | 0.0581 | 0.0110 | 0.0903 | n/a | n/a | 1.2679 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.1738 | 0.1775 | 0.1612 | 0.2004 | 0.0152 | 0.1058 | n/a | n/a | 1.8640 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0184 | 0.0192 | 0.0138 | 0.0243 | 0.0035 | 0.1058 | n/a | n/a | 1.8640 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.3240 | 0.3154 | 0.2434 | 0.3691 | 0.0429 | 0.1058 | n/a | n/a | 1.8640 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0436 | 0.0457 | 0.0286 | 0.0699 | 0.0138 | 0.2197 | n/a | n/a | 0.7627 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0096 | 0.0091 | 0.0069 | 0.0120 | 0.0018 | 0.2197 | n/a | n/a | 0.7627 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0332 | 0.0412 | 0.0291 | 0.0639 | 0.0135 | 0.2197 | n/a | n/a | 0.7627 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0369 | 0.0422 | 0.0323 | 0.0601 | 0.0106 | 0.3220 | n/a | n/a | 0.8872 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0119 | 0.0121 | 0.0093 | 0.0174 | 0.0029 | 0.3220 | n/a | n/a | 0.8872 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0327 | 0.0418 | 0.0305 | 0.0799 | 0.0191 | 0.3220 | n/a | n/a | 0.8872 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0395 | 0.0398 | 0.0292 | 0.0515 | 0.0077 | 0.2465 | n/a | n/a | 0.7664 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0097 | 0.0097 | 0.0076 | 0.0116 | 0.0013 | 0.2465 | n/a | n/a | 0.7664 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0303 | 0.0372 | 0.0292 | 0.0658 | 0.0143 | 0.2465 | n/a | n/a | 0.7664 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0402 | 0.0415 | 0.0323 | 0.0483 | 0.0059 | 0.1884 | n/a | n/a | 0.8676 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0076 | 0.0090 | 0.0055 | 0.0128 | 0.0030 | 0.1884 | n/a | n/a | 0.8676 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0349 | 0.0371 | 0.0239 | 0.0503 | 0.0112 | 0.1884 | n/a | n/a | 0.8676 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1903 | 0.1893 | 0.1518 | 0.2274 | 0.0248 | 0.1934 | n/a | n/a | 1.3996 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0368 | 0.0415 | 0.0286 | 0.0709 | 0.0156 | 0.1934 | n/a | n/a | 1.3996 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.2663 | 0.2674 | 0.2147 | 0.3109 | 0.0329 | 0.1934 | n/a | n/a | 1.3996 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0240 | 0.0241 | 0.0229 | 0.0257 | 0.0010 | 0.1149 | n/a | n/a | 4.0555 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0028 | 0.0028 | 0.0027 | 0.0029 | 0.0001 | 0.1149 | n/a | n/a | 4.0555 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0972 | 0.0957 | 0.0702 | 0.1188 | 0.0164 | 0.1149 | n/a | n/a | 4.0555 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0263 | 0.0262 | 0.0243 | 0.0284 | 0.0014 | 0.3260 | n/a | n/a | 5.5935 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0086 | 0.0086 | 0.0072 | 0.0100 | 0.0011 | 0.3260 | n/a | n/a | 5.5935 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.1470 | 0.1415 | 0.1213 | 0.1589 | 0.0153 | 0.3260 | n/a | n/a | 5.5935 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0287 | 0.0295 | 0.0271 | 0.0329 | 0.0020 | 0.0654 | n/a | n/a | 2.2533 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0019 | 0.0020 | 0.0017 | 0.0028 | 0.0004 | 0.0654 | n/a | n/a | 2.2533 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0647 | 0.0667 | 0.0581 | 0.0812 | 0.0086 | 0.0654 | n/a | n/a | 2.2533 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0277 | 0.0277 | 0.0246 | 0.0312 | 0.0022 | 0.0670 | n/a | n/a | 2.0680 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0019 | 0.0021 | 0.0014 | 0.0036 | 0.0008 | 0.0670 | n/a | n/a | 2.0680 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0572 | 0.0579 | 0.0547 | 0.0632 | 0.0031 | 0.0670 | n/a | n/a | 2.0680 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0268 | 0.0264 | 0.0245 | 0.0275 | 0.0011 | 0.1042 | n/a | n/a | 2.3251 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0028 | 0.0028 | 0.0022 | 0.0033 | 0.0004 | 0.1042 | n/a | n/a | 2.3251 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0624 | 0.0657 | 0.0544 | 0.0869 | 0.0113 | 0.1042 | n/a | n/a | 2.3251 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0254 | 0.0254 | 0.0238 | 0.0271 | 0.0011 | 0.1336 | n/a | n/a | 3.1145 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0034 | 0.0031 | 0.0024 | 0.0037 | 0.0006 | 0.1336 | n/a | n/a | 3.1145 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0790 | 0.0821 | 0.0750 | 0.0970 | 0.0081 | 0.1336 | n/a | n/a | 3.1145 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0238 | 0.0238 | 0.0224 | 0.0252 | 0.0009 | 0.2573 | n/a | n/a | 5.1356 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0061 | 0.0061 | 0.0058 | 0.0066 | 0.0003 | 0.2573 | n/a | n/a | 5.1356 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.1223 | 0.1385 | 0.1210 | 0.1977 | 0.0297 | 0.2573 | n/a | n/a | 5.1356 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0242 | 0.0243 | 0.0226 | 0.0262 | 0.0012 | 0.0659 | n/a | n/a | 1.9487 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0016 | 0.0019 | 0.0014 | 0.0033 | 0.0007 | 0.0659 | n/a | n/a | 1.9487 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0471 | 0.0461 | 0.0428 | 0.0489 | 0.0027 | 0.0659 | n/a | n/a | 1.9487 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0238 | 0.0258 | 0.0226 | 0.0336 | 0.0040 | 0.0835 | n/a | n/a | 2.0213 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0020 | 0.0025 | 0.0016 | 0.0037 | 0.0008 | 0.0835 | n/a | n/a | 2.0213 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0481 | 0.0469 | 0.0436 | 0.0485 | 0.0019 | 0.0835 | n/a | n/a | 2.0213 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0257 | 0.0268 | 0.0243 | 0.0300 | 0.0021 | 0.0962 | n/a | n/a | 1.9611 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0025 | 0.0024 | 0.0020 | 0.0029 | 0.0003 | 0.0962 | n/a | n/a | 1.9611 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0504 | 0.0511 | 0.0453 | 0.0600 | 0.0050 | 0.0962 | n/a | n/a | 1.9611 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0250 | 0.0248 | 0.0237 | 0.0261 | 0.0008 | 0.0865 | n/a | n/a | 2.5687 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0022 | 0.0024 | 0.0020 | 0.0029 | 0.0004 | 0.0865 | n/a | n/a | 2.5687 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0643 | 0.0663 | 0.0598 | 0.0736 | 0.0059 | 0.0865 | n/a | n/a | 2.5687 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0221 | 0.0218 | 0.0204 | 0.0233 | 0.0010 | 0.3106 | n/a | n/a | 4.9529 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0068 | 0.0064 | 0.0053 | 0.0070 | 0.0007 | 0.3106 | n/a | n/a | 4.9529 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.1092 | 0.1123 | 0.1042 | 0.1210 | 0.0068 | 0.3106 | n/a | n/a | 4.9529 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0252 | 0.0304 | 0.0245 | 0.0407 | 0.0069 | 0.0865 | n/a | n/a | 1.8458 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0022 | 0.0023 | 0.0016 | 0.0034 | 0.0007 | 0.0865 | n/a | n/a | 1.8458 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0465 | 0.0464 | 0.0440 | 0.0493 | 0.0017 | 0.0865 | n/a | n/a | 1.8458 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0233 | 0.0234 | 0.0220 | 0.0249 | 0.0010 | 0.1451 | n/a | n/a | 1.8928 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0034 | 0.0032 | 0.0018 | 0.0047 | 0.0011 | 0.1451 | n/a | n/a | 1.8928 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0441 | 0.0477 | 0.0431 | 0.0596 | 0.0062 | 0.1451 | n/a | n/a | 1.8928 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0234 | 0.0245 | 0.0222 | 0.0288 | 0.0025 | 0.0889 | n/a | n/a | 2.0344 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0021 | 0.0020 | 0.0017 | 0.0024 | 0.0002 | 0.0889 | n/a | n/a | 2.0344 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0475 | 0.0487 | 0.0426 | 0.0558 | 0.0048 | 0.0889 | n/a | n/a | 2.0344 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0235 | 0.0234 | 0.0224 | 0.0251 | 0.0010 | 0.1031 | n/a | n/a | 3.0207 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0024 | 0.0026 | 0.0021 | 0.0037 | 0.0006 | 0.1031 | n/a | n/a | 3.0207 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0711 | 0.0736 | 0.0648 | 0.0835 | 0.0068 | 0.1031 | n/a | n/a | 3.0207 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0243 | 0.0246 | 0.0228 | 0.0266 | 0.0012 | 0.2875 | n/a | n/a | 4.7149 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0070 | 0.0069 | 0.0060 | 0.0079 | 0.0007 | 0.2875 | n/a | n/a | 4.7149 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.1144 | 0.1173 | 0.1102 | 0.1352 | 0.0092 | 0.2875 | n/a | n/a | 4.7149 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0292 | 0.0292 | 0.0257 | 0.0332 | 0.0024 | 0.0681 | n/a | n/a | 1.5711 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0020 | 0.0022 | 0.0018 | 0.0027 | 0.0004 | 0.0681 | n/a | n/a | 1.5711 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0458 | 0.0467 | 0.0416 | 0.0522 | 0.0038 | 0.0681 | n/a | n/a | 1.5711 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0250 | 0.0296 | 0.0231 | 0.0444 | 0.0079 | 0.0774 | n/a | n/a | 1.7483 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0019 | 0.0020 | 0.0018 | 0.0025 | 0.0002 | 0.0774 | n/a | n/a | 1.7483 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0437 | 0.0443 | 0.0410 | 0.0487 | 0.0026 | 0.0774 | n/a | n/a | 1.7483 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0255 | 0.0260 | 0.0230 | 0.0281 | 0.0019 | 0.0636 | n/a | n/a | 1.9900 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0016 | 0.0020 | 0.0016 | 0.0029 | 0.0005 | 0.0636 | n/a | n/a | 1.9900 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0507 | 0.0549 | 0.0412 | 0.0750 | 0.0114 | 0.0636 | n/a | n/a | 1.9900 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0232 | 0.0234 | 0.0209 | 0.0283 | 0.0026 | 0.1307 | n/a | n/a | 4.1594 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0030 | 0.0034 | 0.0027 | 0.0043 | 0.0007 | 0.1307 | n/a | n/a | 4.1594 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0963 | 0.0935 | 0.0813 | 0.1052 | 0.0084 | 0.1307 | n/a | n/a | 4.1594 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0275 | 0.0283 | 0.0266 | 0.0321 | 0.0020 | 0.3389 | n/a | n/a | 5.4587 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0093 | 0.0103 | 0.0079 | 0.0141 | 0.0023 | 0.3389 | n/a | n/a | 5.4587 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1502 | 0.1540 | 0.1288 | 0.1883 | 0.0203 | 0.3389 | n/a | n/a | 5.4587 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0368 | 0.0380 | 0.0350 | 0.0440 | 0.0031 | 0.1289 | n/a | n/a | 1.1674 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0047 | 0.0054 | 0.0044 | 0.0069 | 0.0010 | 0.1289 | n/a | n/a | 1.1674 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0430 | 0.0437 | 0.0390 | 0.0526 | 0.0047 | 0.1289 | n/a | n/a | 1.1674 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0309 | 0.0312 | 0.0279 | 0.0365 | 0.0029 | 0.1531 | n/a | n/a | 1.4790 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0047 | 0.0049 | 0.0041 | 0.0057 | 0.0006 | 0.1531 | n/a | n/a | 1.4790 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0457 | 0.0478 | 0.0381 | 0.0608 | 0.0075 | 0.1531 | n/a | n/a | 1.4790 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0296 | 0.0308 | 0.0273 | 0.0386 | 0.0041 | 0.1642 | n/a | n/a | 1.8129 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0049 | 0.0052 | 0.0032 | 0.0088 | 0.0020 | 0.1642 | n/a | n/a | 1.8129 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0537 | 0.0521 | 0.0415 | 0.0637 | 0.0086 | 0.1642 | n/a | n/a | 1.8129 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0330 | 0.0323 | 0.0259 | 0.0388 | 0.0042 | 0.1752 | n/a | n/a | 3.1448 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0058 | 0.0056 | 0.0039 | 0.0071 | 0.0012 | 0.1752 | n/a | n/a | 3.1448 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.1039 | 0.1004 | 0.0926 | 0.1066 | 0.0065 | 0.1752 | n/a | n/a | 3.1448 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0505 | 0.0523 | 0.0496 | 0.0561 | 0.0028 | 0.2965 | n/a | n/a | 3.3987 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0150 | 0.0162 | 0.0138 | 0.0204 | 0.0025 | 0.2965 | n/a | n/a | 3.3987 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1716 | 0.1688 | 0.1470 | 0.1943 | 0.0161 | 0.2965 | n/a | n/a | 3.3987 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0975 | 0.0967 | 0.0920 | 0.1022 | 0.0041 | 0.2005 | n/a | n/a | 0.4953 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0195 | 0.0191 | 0.0129 | 0.0235 | 0.0036 | 0.2005 | n/a | n/a | 0.4953 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0483 | 0.0519 | 0.0436 | 0.0704 | 0.0095 | 0.2005 | n/a | n/a | 0.4953 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0511 | 0.0526 | 0.0457 | 0.0614 | 0.0053 | 0.1764 | n/a | n/a | 1.1354 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0090 | 0.0113 | 0.0081 | 0.0164 | 0.0033 | 0.1764 | n/a | n/a | 1.1354 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0580 | 0.0565 | 0.0432 | 0.0674 | 0.0086 | 0.1764 | n/a | n/a | 1.1354 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0528 | 0.0583 | 0.0482 | 0.0717 | 0.0091 | 0.2313 | n/a | n/a | 0.9551 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0122 | 0.0124 | 0.0094 | 0.0157 | 0.0023 | 0.2313 | n/a | n/a | 0.9551 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0504 | 0.0599 | 0.0424 | 0.0863 | 0.0172 | 0.2313 | n/a | n/a | 0.9551 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0300 | 0.0319 | 0.0275 | 0.0382 | 0.0042 | 0.4023 | n/a | n/a | 3.7647 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0121 | 0.0122 | 0.0109 | 0.0143 | 0.0011 | 0.4023 | n/a | n/a | 3.7647 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.1128 | 0.1134 | 0.0941 | 0.1264 | 0.0118 | 0.4023 | n/a | n/a | 3.7647 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1169 | 0.1274 | 0.1061 | 0.1775 | 0.0258 | 0.3635 | n/a | n/a | 1.3650 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0425 | 0.0454 | 0.0386 | 0.0636 | 0.0093 | 0.3635 | n/a | n/a | 1.3650 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1596 | 0.1825 | 0.1494 | 0.2431 | 0.0368 | 0.3635 | n/a | n/a | 1.3650 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2882 | 0.3060 | 0.2776 | 0.3566 | 0.0311 | 0.1912 | n/a | n/a | 0.1847 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0551 | 0.0551 | 0.0492 | 0.0617 | 0.0040 | 0.1912 | n/a | n/a | 0.1847 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0532 | 0.0588 | 0.0427 | 0.0899 | 0.0165 | 0.1912 | n/a | n/a | 0.1847 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1062 | 0.1060 | 0.0993 | 0.1125 | 0.0054 | 0.2149 | n/a | n/a | 0.5317 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0228 | 0.0263 | 0.0225 | 0.0374 | 0.0057 | 0.2149 | n/a | n/a | 0.5317 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0565 | 0.0559 | 0.0472 | 0.0647 | 0.0070 | 0.2149 | n/a | n/a | 0.5317 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1695 | 0.1625 | 0.1323 | 0.1782 | 0.0164 | 0.2244 | n/a | n/a | 0.4846 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0380 | 0.0380 | 0.0299 | 0.0462 | 0.0063 | 0.2244 | n/a | n/a | 0.4846 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0821 | 0.0823 | 0.0762 | 0.0904 | 0.0055 | 0.2244 | n/a | n/a | 0.4846 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0260 | 0.0252 | 0.0221 | 0.0288 | 0.0025 | 0.1269 | n/a | n/a | 0.7584 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0033 | 0.0042 | 0.0027 | 0.0062 | 0.0016 | 0.1269 | n/a | n/a | 0.7584 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0197 | 0.0196 | 0.0170 | 0.0235 | 0.0022 | 0.1269 | n/a | n/a | 0.7584 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0248 | 0.0251 | 0.0233 | 0.0274 | 0.0015 | 0.4921 | n/a | n/a | 0.8573 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0122 | 0.0124 | 0.0110 | 0.0151 | 0.0014 | 0.4921 | n/a | n/a | 0.8573 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0213 | 0.0258 | 0.0171 | 0.0487 | 0.0117 | 0.4921 | n/a | n/a | 0.8573 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0250 | 0.0260 | 0.0248 | 0.0288 | 0.0015 | 0.1860 | n/a | n/a | 0.8792 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0047 | 0.0043 | 0.0027 | 0.0051 | 0.0009 | 0.1860 | n/a | n/a | 0.8792 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0220 | 0.0266 | 0.0179 | 0.0461 | 0.0104 | 0.1860 | n/a | n/a | 0.8792 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0269 | 0.0280 | 0.0234 | 0.0361 | 0.0043 | 0.4534 | n/a | n/a | 0.6875 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0122 | 0.0124 | 0.0112 | 0.0145 | 0.0012 | 0.4534 | n/a | n/a | 0.6875 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0185 | 0.0187 | 0.0157 | 0.0208 | 0.0018 | 0.4534 | n/a | n/a | 0.6875 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0269 | 0.0262 | 0.0225 | 0.0285 | 0.0021 | 0.1305 | n/a | n/a | 0.7343 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0035 | 0.0034 | 0.0026 | 0.0042 | 0.0006 | 0.1305 | n/a | n/a | 0.7343 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0197 | 0.0215 | 0.0190 | 0.0250 | 0.0025 | 0.1305 | n/a | n/a | 0.7343 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0256 | 0.0280 | 0.0229 | 0.0339 | 0.0047 | 0.5052 | n/a | n/a | 0.8221 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0129 | 0.0148 | 0.0129 | 0.0190 | 0.0025 | 0.5052 | n/a | n/a | 0.8221 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0210 | 0.0203 | 0.0161 | 0.0259 | 0.0035 | 0.5052 | n/a | n/a | 0.8221 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0253 | 0.0253 | 0.0226 | 0.0296 | 0.0024 | 0.1110 | n/a | n/a | 1.0837 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0028 | 0.0032 | 0.0027 | 0.0040 | 0.0005 | 0.1110 | n/a | n/a | 1.0837 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0274 | 0.0295 | 0.0217 | 0.0414 | 0.0071 | 0.1110 | n/a | n/a | 1.0837 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0270 | 0.0268 | 0.0249 | 0.0295 | 0.0016 | 0.4654 | n/a | n/a | 0.8743 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0126 | 0.0123 | 0.0104 | 0.0134 | 0.0011 | 0.4654 | n/a | n/a | 0.8743 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0236 | 0.0230 | 0.0212 | 0.0240 | 0.0011 | 0.4654 | n/a | n/a | 0.8743 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0265 | 0.0257 | 0.0195 | 0.0291 | 0.0033 | 0.2508 | n/a | n/a | 0.9931 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0067 | 0.0080 | 0.0046 | 0.0144 | 0.0036 | 0.2508 | n/a | n/a | 0.9931 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0263 | 0.0257 | 0.0205 | 0.0334 | 0.0046 | 0.2508 | n/a | n/a | 0.9931 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0261 | 0.0263 | 0.0235 | 0.0300 | 0.0022 | 0.6069 | n/a | n/a | 0.9108 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0158 | 0.0157 | 0.0117 | 0.0176 | 0.0022 | 0.6069 | n/a | n/a | 0.9108 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0238 | 0.0248 | 0.0229 | 0.0290 | 0.0022 | 0.6069 | n/a | n/a | 0.9108 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0301 | 0.0300 | 0.0276 | 0.0326 | 0.0020 | 0.2606 | n/a | n/a | 1.1609 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0078 | 0.0079 | 0.0050 | 0.0126 | 0.0026 | 0.2606 | n/a | n/a | 1.1609 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0350 | 0.0343 | 0.0282 | 0.0414 | 0.0044 | 0.2606 | n/a | n/a | 1.1609 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0311 | 0.0322 | 0.0306 | 0.0373 | 0.0026 | 0.4333 | n/a | n/a | 0.8818 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0135 | 0.0142 | 0.0130 | 0.0171 | 0.0015 | 0.4333 | n/a | n/a | 0.8818 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0274 | 0.0267 | 0.0224 | 0.0298 | 0.0025 | 0.4333 | n/a | n/a | 0.8818 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0530 | 0.0557 | 0.0466 | 0.0646 | 0.0071 | 0.1648 | n/a | n/a | 0.6824 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0087 | 0.0115 | 0.0087 | 0.0161 | 0.0034 | 0.1648 | n/a | n/a | 0.6824 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0362 | 0.0440 | 0.0300 | 0.0837 | 0.0201 | 0.1648 | n/a | n/a | 0.6824 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0598 | 0.0589 | 0.0453 | 0.0696 | 0.0078 | 0.6453 | n/a | n/a | 0.6560 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0386 | 0.0351 | 0.0206 | 0.0547 | 0.0128 | 0.6453 | n/a | n/a | 0.6560 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0392 | 0.0450 | 0.0302 | 0.0831 | 0.0196 | 0.6453 | n/a | n/a | 0.6560 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0290 | 0.0279 | 0.0225 | 0.0318 | 0.0034 | 0.1170 | n/a | n/a | 0.6453 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0034 | 0.0051 | 0.0033 | 0.0108 | 0.0029 | 0.1170 | n/a | n/a | 0.6453 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0187 | 0.0191 | 0.0167 | 0.0216 | 0.0017 | 0.1170 | n/a | n/a | 0.6453 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0160 | 0.0155 | 0.0137 | 0.0168 | 0.0011 | 1.0732 | n/a | n/a | 1.1332 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0171 | 0.0164 | 0.0133 | 0.0186 | 0.0020 | 1.0732 | n/a | n/a | 1.1332 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0181 | 0.0206 | 0.0167 | 0.0284 | 0.0042 | 1.0732 | n/a | n/a | 1.1332 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0304 | 0.0308 | 0.0291 | 0.0333 | 0.0016 | 0.2060 | n/a | n/a | 0.9207 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0063 | 0.0073 | 0.0054 | 0.0108 | 0.0021 | 0.2060 | n/a | n/a | 0.9207 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0280 | 0.0294 | 0.0269 | 0.0323 | 0.0023 | 0.2060 | n/a | n/a | 0.9207 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0242 | 0.0235 | 0.0193 | 0.0276 | 0.0028 | 0.8549 | n/a | n/a | 1.5012 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0207 | 0.0205 | 0.0160 | 0.0248 | 0.0029 | 0.8549 | n/a | n/a | 1.5012 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0363 | 0.0341 | 0.0253 | 0.0381 | 0.0047 | 0.8549 | n/a | n/a | 1.5012 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0632 | 0.0653 | 0.0585 | 0.0761 | 0.0067 | 0.2698 | n/a | n/a | 0.5263 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0170 | 0.0170 | 0.0144 | 0.0203 | 0.0019 | 0.2698 | n/a | n/a | 0.5263 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0333 | 0.0403 | 0.0283 | 0.0589 | 0.0128 | 0.2698 | n/a | n/a | 0.5263 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0354 | 0.0351 | 0.0310 | 0.0395 | 0.0035 | 0.6397 | n/a | n/a | 0.8081 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0227 | 0.0261 | 0.0199 | 0.0345 | 0.0061 | 0.6397 | n/a | n/a | 0.8081 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0286 | 0.0254 | 0.0184 | 0.0296 | 0.0048 | 0.6397 | n/a | n/a | 0.8081 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.1115 | 0.1130 | 0.1094 | 0.1177 | 0.0034 | 0.2447 | n/a | n/a | 0.6728 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0273 | 0.0288 | 0.0262 | 0.0346 | 0.0031 | 0.2447 | n/a | n/a | 0.6728 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0750 | 0.0748 | 0.0701 | 0.0801 | 0.0041 | 0.2447 | n/a | n/a | 0.6728 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0793 | 0.0814 | 0.0677 | 0.1098 | 0.0152 | 0.4941 | n/a | n/a | 0.3058 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0392 | 0.0359 | 0.0250 | 0.0458 | 0.0078 | 0.4941 | n/a | n/a | 0.3058 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0243 | 0.0296 | 0.0230 | 0.0416 | 0.0076 | 0.4941 | n/a | n/a | 0.3058 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0262 | 0.0281 | 0.0243 | 0.0339 | 0.0038 | 0.0876 | n/a | n/a | 0.7956 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.0876 | n/a | n/a | 0.7956 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0209 | 0.0196 | 0.0132 | 0.0272 | 0.0050 | 0.0876 | n/a | n/a | 0.7956 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1550 | 0.1475 | 0.1322 | 0.1599 | 0.0121 | 0.0817 | n/a | n/a | 0.1363 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0127 | 0.0143 | 0.0125 | 0.0202 | 0.0030 | 0.0817 | n/a | n/a | 0.1363 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0211 | 0.0246 | 0.0189 | 0.0401 | 0.0079 | 0.0817 | n/a | n/a | 0.1363 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4534 | 0.4784 | 0.4391 | 0.5671 | 0.0479 | 0.7439 | n/a | n/a | 0.3004 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3373 | 0.3528 | 0.2426 | 0.5221 | 0.1071 | 0.7439 | n/a | n/a | 0.3004 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1362 | 0.1283 | 0.1017 | 0.1427 | 0.0148 | 0.7439 | n/a | n/a | 0.3004 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 3.6061 | 3.2990 | 2.6264 | 3.9023 | 0.5131 | 0.1413 | n/a | n/a | 0.0994 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.5097 | 0.5470 | 0.4596 | 0.7259 | 0.0973 | 0.1413 | n/a | n/a | 0.0994 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.3583 | 0.4153 | 0.3258 | 0.6527 | 0.1209 | 0.1413 | n/a | n/a | 0.0994 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2448 | 0.2417 | 0.2333 | 0.2470 | 0.0055 | 8.0224 | n/a | n/a | 0.6561 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.9636 | 2.0802 | 1.9260 | 2.4227 | 0.1854 | 8.0224 | n/a | n/a | 0.6561 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1606 | 0.1804 | 0.1464 | 0.2336 | 0.0334 | 8.0224 | n/a | n/a | 0.6561 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 6.6522 | 6.6190 | 6.4917 | 6.6785 | 0.0667 | 14.7299 | n/a | n/a | 0.0565 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 97.9869 | 100.0982 | 92.9422 | 107.5844 | 5.4788 | 14.7299 | n/a | n/a | 0.0565 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3757 | 0.4918 | 0.3580 | 0.7147 | 0.1540 | 14.7299 | n/a | n/a | 0.0565 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0392 | 0.0382 | 0.0328 | 0.0410 | 0.0028 | 8.8440 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.3468 | 0.3462 | 0.2880 | 0.4051 | 0.0465 | 8.8440 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0460 | 0.0464 | 0.0333 | 0.0659 | 0.0111 | 22.4030 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 1.0295 | 1.0283 | 0.8414 | 1.1834 | 0.1097 | 22.4030 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.9724 | 0.9709 | 0.9278 | 1.0194 | 0.0347 | 13.4401 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 13.0692 | 13.0571 | 12.3155 | 13.9354 | 0.5201 | 13.4401 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.9858 | 0.9955 | 0.9551 | 1.0572 | 0.0336 | 45.9119 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 45.2612 | 45.4769 | 44.6702 | 47.4123 | 1.0059 | 45.9119 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.2130 | 0.2493 | 0.2038 | 0.3357 | 0.0514 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.6254 | 0.6111 | 0.5509 | 0.6561 | 0.0381 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.4713 | 0.4800 | 0.4215 | 0.5522 | 0.0529 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 1.0380 | 1.0531 | 0.9138 | 1.2283 | 0.1006 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.6811 | 1.7199 | 1.6249 | 1.8639 | 0.0829 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 4.1790 | 4.3249 | 4.0189 | 5.0202 | 0.3700 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 10.7848 | 11.0360 | 8.4559 | 15.4410 | 2.3819 | 0.0214 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.2309 | 0.2203 | 0.1853 | 0.2656 | 0.0306 | 0.0214 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |
| backends | backend_info | `(1,)` | TensorStudio | 0.0027 | 0.0030 | 0.0026 | 0.0040 | 0.0005 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | available_devices | `(1,)` | TensorStudio | 0.0019 | 0.0020 | 0.0018 | 0.0025 | 0.0002 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cpu_transfer | `(1,)` | TensorStudio | 0.0023 | 0.0030 | 0.0020 | 0.0045 | 0.0012 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cuda_availability_check | `(1,)` | TensorStudio | 0.0046 | 0.0043 | 0.0029 | 0.0054 | 0.0009 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
