# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `2.1.0`
- TensorStudio threads: `12`
- TensorStudio BLAS enabled: `False`
- TensorStudio SIMD level: `sse2`
- TensorStudio storage pool enabled: `True`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `8`
- TensorStudio losses versus NumPy: `95`
- TensorStudio wins versus JAX CPU dispatch: `39`
- TensorStudio losses versus JAX CPU dispatch: `59`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0336 | 0.0325 | 0.0284 | 0.0344 | 0.0023 | 0.0409 | n/a | n/a | 0.9569 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 0.0409 | n/a | n/a | 0.9569 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0321 | 0.0311 | 0.0278 | 0.0339 | 0.0022 | 0.0409 | n/a | n/a | 0.9569 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0357 | 0.0363 | 0.0348 | 0.0381 | 0.0014 | 0.0470 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0017 | 0.0020 | 0.0016 | 0.0031 | 0.0006 | 0.0470 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0304 | 0.0319 | 0.0278 | 0.0387 | 0.0039 | 0.0470 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0363 | 0.0360 | 0.0350 | 0.0368 | 0.0007 | 0.0492 | n/a | n/a | 0.7993 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0018 | 0.0017 | 0.0014 | 0.0018 | 0.0002 | 0.0492 | n/a | n/a | 0.7993 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0290 | 0.0290 | 0.0271 | 0.0312 | 0.0015 | 0.0492 | n/a | n/a | 0.7993 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0352 | 0.0349 | 0.0328 | 0.0360 | 0.0012 | 0.0460 | n/a | n/a | 0.6848 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0016 | 0.0017 | 0.0013 | 0.0022 | 0.0003 | 0.0460 | n/a | n/a | 0.6848 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0241 | 0.0245 | 0.0220 | 0.0271 | 0.0021 | 0.0460 | n/a | n/a | 0.6848 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.1931 | 0.1938 | 0.1707 | 0.2176 | 0.0172 | 0.0502 | n/a | n/a | 1.0679 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0097 | 0.0099 | 0.0085 | 0.0115 | 0.0011 | 0.0502 | n/a | n/a | 1.0679 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.2062 | 0.2047 | 0.1926 | 0.2211 | 0.0097 | 0.0502 | n/a | n/a | 1.0679 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0338 | 0.0350 | 0.0329 | 0.0386 | 0.0023 | 0.0510 | n/a | n/a | 0.8285 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0017 | 0.0016 | 0.0013 | 0.0018 | 0.0002 | 0.0510 | n/a | n/a | 0.8285 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0280 | 0.0280 | 0.0244 | 0.0311 | 0.0022 | 0.0510 | n/a | n/a | 0.8285 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0343 | 0.0340 | 0.0319 | 0.0350 | 0.0011 | 0.0398 | n/a | n/a | 0.7627 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0014 | 0.0015 | 0.0012 | 0.0018 | 0.0002 | 0.0398 | n/a | n/a | 0.7627 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0262 | 0.0264 | 0.0256 | 0.0273 | 0.0006 | 0.0398 | n/a | n/a | 0.7627 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0338 | 0.0338 | 0.0320 | 0.0352 | 0.0010 | 0.0557 | n/a | n/a | 0.8718 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0019 | 0.0021 | 0.0017 | 0.0027 | 0.0003 | 0.0557 | n/a | n/a | 0.8718 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0295 | 0.0288 | 0.0261 | 0.0304 | 0.0017 | 0.0557 | n/a | n/a | 0.8718 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0340 | 0.0338 | 0.0312 | 0.0353 | 0.0015 | 0.0552 | n/a | n/a | 0.7058 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0019 | 0.0017 | 0.0013 | 0.0020 | 0.0003 | 0.0552 | n/a | n/a | 0.7058 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0240 | 0.0228 | 0.0196 | 0.0256 | 0.0024 | 0.0552 | n/a | n/a | 0.7058 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.1776 | 0.1770 | 0.1715 | 0.1823 | 0.0039 | 0.0679 | n/a | n/a | 1.1560 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0121 | 0.0120 | 0.0087 | 0.0154 | 0.0022 | 0.0679 | n/a | n/a | 1.1560 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.2053 | 0.2026 | 0.1872 | 0.2151 | 0.0096 | 0.0679 | n/a | n/a | 1.1560 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0342 | 0.0359 | 0.0325 | 0.0406 | 0.0031 | 0.0438 | n/a | n/a | 0.8773 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0018 | 0.0001 | 0.0438 | n/a | n/a | 0.8773 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0300 | 0.0304 | 0.0244 | 0.0366 | 0.0042 | 0.0438 | n/a | n/a | 0.8773 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0345 | 0.0344 | 0.0338 | 0.0354 | 0.0006 | 0.0534 | n/a | n/a | 0.7990 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0018 | 0.0017 | 0.0015 | 0.0019 | 0.0002 | 0.0534 | n/a | n/a | 0.7990 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0276 | 0.0275 | 0.0236 | 0.0301 | 0.0023 | 0.0534 | n/a | n/a | 0.7990 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0354 | 0.0352 | 0.0348 | 0.0358 | 0.0004 | 0.0452 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0016 | 0.0017 | 0.0014 | 0.0020 | 0.0002 | 0.0452 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0301 | 0.0294 | 0.0275 | 0.0310 | 0.0013 | 0.0452 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0351 | 0.0345 | 0.0324 | 0.0359 | 0.0012 | 0.0529 | n/a | n/a | 0.6456 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0019 | 0.0017 | 0.0015 | 0.0020 | 0.0002 | 0.0529 | n/a | n/a | 0.6456 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0226 | 0.0228 | 0.0204 | 0.0256 | 0.0018 | 0.0529 | n/a | n/a | 0.6456 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.1920 | 0.1905 | 0.1816 | 0.1970 | 0.0052 | 0.0796 | n/a | n/a | 1.0693 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0153 | 0.0152 | 0.0142 | 0.0159 | 0.0006 | 0.0796 | n/a | n/a | 1.0693 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.2053 | 0.2071 | 0.1933 | 0.2356 | 0.0151 | 0.0796 | n/a | n/a | 1.0693 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0354 | 0.0352 | 0.0317 | 0.0379 | 0.0020 | 0.0496 | n/a | n/a | 0.6894 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0018 | 0.0018 | 0.0015 | 0.0024 | 0.0003 | 0.0496 | n/a | n/a | 0.6894 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0244 | 0.0253 | 0.0221 | 0.0310 | 0.0031 | 0.0496 | n/a | n/a | 0.6894 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0343 | 0.0358 | 0.0332 | 0.0436 | 0.0039 | 0.0674 | n/a | n/a | 0.8237 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0023 | 0.0022 | 0.0019 | 0.0025 | 0.0003 | 0.0674 | n/a | n/a | 0.8237 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0282 | 0.0280 | 0.0262 | 0.0292 | 0.0010 | 0.0674 | n/a | n/a | 0.8237 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0353 | 0.0343 | 0.0292 | 0.0364 | 0.0027 | 0.0347 | n/a | n/a | 0.8605 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0012 | 0.0011 | 0.0009 | 0.0013 | 0.0002 | 0.0347 | n/a | n/a | 0.8605 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0304 | 0.0288 | 0.0234 | 0.0324 | 0.0032 | 0.0347 | n/a | n/a | 0.8605 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0333 | 0.0323 | 0.0282 | 0.0355 | 0.0027 | 0.0376 | n/a | n/a | 0.6500 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0012 | 0.0013 | 0.0011 | 0.0015 | 0.0001 | 0.0376 | n/a | n/a | 0.6500 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0216 | 0.0208 | 0.0154 | 0.0236 | 0.0028 | 0.0376 | n/a | n/a | 0.6500 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.1756 | 0.1788 | 0.1716 | 0.1928 | 0.0075 | 0.0907 | n/a | n/a | 1.2924 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0159 | 0.0162 | 0.0146 | 0.0179 | 0.0011 | 0.0907 | n/a | n/a | 1.2924 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.2269 | 0.2248 | 0.2078 | 0.2376 | 0.0109 | 0.0907 | n/a | n/a | 1.2924 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0368 | 0.0367 | 0.0341 | 0.0391 | 0.0018 | 0.0749 | n/a | n/a | 0.7062 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0028 | 0.0027 | 0.0024 | 0.0030 | 0.0002 | 0.0749 | n/a | n/a | 0.7062 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0260 | 0.0267 | 0.0240 | 0.0294 | 0.0019 | 0.0749 | n/a | n/a | 0.7062 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0335 | 0.0342 | 0.0313 | 0.0374 | 0.0022 | 0.1168 | n/a | n/a | 0.8084 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0039 | 0.0040 | 0.0038 | 0.0041 | 0.0001 | 0.1168 | n/a | n/a | 0.8084 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0271 | 0.0329 | 0.0251 | 0.0562 | 0.0118 | 0.1168 | n/a | n/a | 0.8084 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0338 | 0.0342 | 0.0325 | 0.0371 | 0.0015 | 0.0610 | n/a | n/a | 0.9191 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0021 | 0.0023 | 0.0018 | 0.0030 | 0.0005 | 0.0610 | n/a | n/a | 0.9191 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0311 | 0.0338 | 0.0275 | 0.0512 | 0.0089 | 0.0610 | n/a | n/a | 0.9191 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0341 | 0.0347 | 0.0309 | 0.0390 | 0.0028 | 0.0851 | n/a | n/a | 0.6851 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0000 | 0.0851 | n/a | n/a | 0.6851 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0234 | 0.0288 | 0.0198 | 0.0557 | 0.0136 | 0.0851 | n/a | n/a | 0.6851 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.1813 | 0.1877 | 0.1743 | 0.2229 | 0.0178 | 0.0952 | n/a | n/a | 1.2709 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0173 | 0.0171 | 0.0160 | 0.0180 | 0.0008 | 0.0952 | n/a | n/a | 1.2709 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.2305 | 0.2353 | 0.2164 | 0.2708 | 0.0185 | 0.0952 | n/a | n/a | 1.2709 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0375 | 0.0370 | 0.0349 | 0.0381 | 0.0011 | 0.1265 | n/a | n/a | 0.7997 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0047 | 0.0042 | 0.0030 | 0.0051 | 0.0008 | 0.1265 | n/a | n/a | 0.7997 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0300 | 0.0303 | 0.0260 | 0.0353 | 0.0030 | 0.1265 | n/a | n/a | 0.7997 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0363 | 0.0359 | 0.0329 | 0.0383 | 0.0018 | 0.0942 | n/a | n/a | 0.7645 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0034 | 0.0036 | 0.0032 | 0.0041 | 0.0004 | 0.0942 | n/a | n/a | 0.7645 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0277 | 0.0332 | 0.0267 | 0.0490 | 0.0085 | 0.0942 | n/a | n/a | 0.7645 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0347 | 0.0351 | 0.0332 | 0.0375 | 0.0015 | 0.0916 | n/a | n/a | 0.8688 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0032 | 0.0031 | 0.0025 | 0.0033 | 0.0003 | 0.0916 | n/a | n/a | 0.8688 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0302 | 0.0304 | 0.0241 | 0.0380 | 0.0046 | 0.0916 | n/a | n/a | 0.8688 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0380 | 0.0376 | 0.0342 | 0.0395 | 0.0018 | 0.0949 | n/a | n/a | 0.6690 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0036 | 0.0036 | 0.0031 | 0.0042 | 0.0003 | 0.0949 | n/a | n/a | 0.6690 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0254 | 0.0260 | 0.0249 | 0.0277 | 0.0010 | 0.0949 | n/a | n/a | 0.6690 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.1920 | 0.1897 | 0.1758 | 0.1995 | 0.0079 | 0.1195 | n/a | n/a | 1.2836 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0229 | 0.0244 | 0.0199 | 0.0334 | 0.0047 | 0.1195 | n/a | n/a | 1.2836 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.2464 | 0.2442 | 0.2323 | 0.2549 | 0.0079 | 0.1195 | n/a | n/a | 1.2836 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0453 | 0.0446 | 0.0421 | 0.0468 | 0.0017 | 0.2475 | n/a | n/a | 0.6926 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0112 | 0.0109 | 0.0084 | 0.0127 | 0.0014 | 0.2475 | n/a | n/a | 0.6926 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0314 | 0.0324 | 0.0250 | 0.0419 | 0.0066 | 0.2475 | n/a | n/a | 0.6926 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0442 | 0.0439 | 0.0386 | 0.0487 | 0.0037 | 0.2046 | n/a | n/a | 0.8436 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0090 | 0.0092 | 0.0089 | 0.0097 | 0.0003 | 0.2046 | n/a | n/a | 0.8436 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0373 | 0.0363 | 0.0303 | 0.0397 | 0.0035 | 0.2046 | n/a | n/a | 0.8436 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0397 | 0.0409 | 0.0335 | 0.0509 | 0.0062 | 0.1988 | n/a | n/a | 0.7821 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0079 | 0.0083 | 0.0075 | 0.0092 | 0.0007 | 0.1988 | n/a | n/a | 0.7821 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0311 | 0.0325 | 0.0305 | 0.0371 | 0.0025 | 0.1988 | n/a | n/a | 0.7821 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0410 | 0.0422 | 0.0386 | 0.0481 | 0.0034 | 0.1906 | n/a | n/a | 0.6654 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0078 | 0.0079 | 0.0078 | 0.0082 | 0.0002 | 0.1906 | n/a | n/a | 0.6654 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0273 | 0.0276 | 0.0270 | 0.0290 | 0.0008 | 0.1906 | n/a | n/a | 0.6654 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.2122 | 0.2176 | 0.2108 | 0.2343 | 0.0089 | 0.2190 | n/a | n/a | 1.2998 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0465 | 0.0462 | 0.0438 | 0.0496 | 0.0020 | 0.2190 | n/a | n/a | 1.2998 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.2759 | 0.2819 | 0.2355 | 0.3644 | 0.0454 | 0.2190 | n/a | n/a | 1.2998 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0359 | 0.0355 | 0.0321 | 0.0404 | 0.0029 | 0.1094 | n/a | n/a | 2.8123 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0039 | 0.0039 | 0.0023 | 0.0048 | 0.0009 | 0.1094 | n/a | n/a | 2.8123 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.1010 | 0.1008 | 0.0984 | 0.1029 | 0.0019 | 0.1094 | n/a | n/a | 2.8123 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0328 | 0.0317 | 0.0260 | 0.0342 | 0.0029 | 0.4379 | n/a | n/a | 5.3081 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0144 | 0.0142 | 0.0127 | 0.0156 | 0.0009 | 0.4379 | n/a | n/a | 5.3081 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.1740 | 0.1735 | 0.1577 | 0.1932 | 0.0123 | 0.4379 | n/a | n/a | 5.3081 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0347 | 0.0354 | 0.0333 | 0.0380 | 0.0019 | 0.1059 | n/a | n/a | 2.1490 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0037 | 0.0036 | 0.0030 | 0.0043 | 0.0005 | 0.1059 | n/a | n/a | 2.1490 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0747 | 0.0772 | 0.0624 | 0.0960 | 0.0112 | 0.1059 | n/a | n/a | 2.1490 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0332 | 0.0336 | 0.0314 | 0.0368 | 0.0018 | 0.0976 | n/a | n/a | 2.0388 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0032 | 0.0030 | 0.0020 | 0.0037 | 0.0006 | 0.0976 | n/a | n/a | 2.0388 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0678 | 0.0667 | 0.0618 | 0.0710 | 0.0034 | 0.0976 | n/a | n/a | 2.0388 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0310 | 0.0324 | 0.0290 | 0.0361 | 0.0030 | 0.1133 | n/a | n/a | 2.3194 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0035 | 0.0034 | 0.0030 | 0.0037 | 0.0002 | 0.1133 | n/a | n/a | 2.3194 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0718 | 0.0746 | 0.0677 | 0.0850 | 0.0063 | 0.1133 | n/a | n/a | 2.3194 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0317 | 0.0319 | 0.0290 | 0.0359 | 0.0025 | 0.1197 | n/a | n/a | 3.0709 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0038 | 0.0040 | 0.0035 | 0.0047 | 0.0004 | 0.1197 | n/a | n/a | 3.0709 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0972 | 0.0967 | 0.0907 | 0.1024 | 0.0044 | 0.1197 | n/a | n/a | 3.0709 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0298 | 0.0289 | 0.0250 | 0.0307 | 0.0020 | 0.4060 | n/a | n/a | 5.8345 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0121 | 0.0119 | 0.0105 | 0.0135 | 0.0011 | 0.4060 | n/a | n/a | 5.8345 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.1737 | 0.1814 | 0.1679 | 0.2170 | 0.0180 | 0.4060 | n/a | n/a | 5.8345 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0293 | 0.0294 | 0.0265 | 0.0347 | 0.0030 | 0.1000 | n/a | n/a | 2.3644 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0029 | 0.0028 | 0.0020 | 0.0031 | 0.0004 | 0.1000 | n/a | n/a | 2.3644 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0693 | 0.0686 | 0.0615 | 0.0756 | 0.0051 | 0.1000 | n/a | n/a | 2.3644 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0357 | 0.0390 | 0.0311 | 0.0563 | 0.0090 | 0.1305 | n/a | n/a | 2.1642 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0047 | 0.0046 | 0.0038 | 0.0050 | 0.0005 | 0.1305 | n/a | n/a | 2.1642 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0772 | 0.0744 | 0.0487 | 0.0889 | 0.0148 | 0.1305 | n/a | n/a | 2.1642 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0360 | 0.0357 | 0.0328 | 0.0388 | 0.0023 | 0.1183 | n/a | n/a | 1.9880 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0043 | 0.0041 | 0.0036 | 0.0046 | 0.0004 | 0.1183 | n/a | n/a | 1.9880 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0716 | 0.0743 | 0.0665 | 0.0815 | 0.0059 | 0.1183 | n/a | n/a | 1.9880 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0317 | 0.0318 | 0.0307 | 0.0338 | 0.0011 | 0.1461 | n/a | n/a | 3.0110 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0046 | 0.0045 | 0.0037 | 0.0049 | 0.0004 | 0.1461 | n/a | n/a | 3.0110 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0953 | 0.0961 | 0.0875 | 0.1057 | 0.0058 | 0.1461 | n/a | n/a | 3.0110 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0316 | 0.0317 | 0.0300 | 0.0335 | 0.0014 | 0.3360 | n/a | n/a | 5.1782 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0106 | 0.0116 | 0.0094 | 0.0162 | 0.0024 | 0.3360 | n/a | n/a | 5.1782 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.1639 | 0.1637 | 0.1465 | 0.1799 | 0.0106 | 0.3360 | n/a | n/a | 5.1782 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0310 | 0.0310 | 0.0288 | 0.0336 | 0.0016 | 0.0965 | n/a | n/a | 2.0846 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0030 | 0.0031 | 0.0026 | 0.0034 | 0.0003 | 0.0965 | n/a | n/a | 2.0846 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0646 | 0.0659 | 0.0599 | 0.0760 | 0.0055 | 0.0965 | n/a | n/a | 2.0846 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0318 | 0.0313 | 0.0304 | 0.0321 | 0.0007 | 0.1001 | n/a | n/a | 2.1813 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0032 | 0.0032 | 0.0030 | 0.0036 | 0.0002 | 0.1001 | n/a | n/a | 2.1813 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0695 | 0.0696 | 0.0656 | 0.0740 | 0.0032 | 0.1001 | n/a | n/a | 2.1813 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0329 | 0.0330 | 0.0302 | 0.0363 | 0.0020 | 0.0908 | n/a | n/a | 2.1775 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0030 | 0.0031 | 0.0025 | 0.0040 | 0.0005 | 0.0908 | n/a | n/a | 2.1775 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0717 | 0.0713 | 0.0676 | 0.0751 | 0.0025 | 0.0908 | n/a | n/a | 2.1775 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0303 | 0.0292 | 0.0229 | 0.0313 | 0.0032 | 0.1196 | n/a | n/a | 3.1220 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0036 | 0.0037 | 0.0026 | 0.0046 | 0.0007 | 0.1196 | n/a | n/a | 3.1220 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0946 | 0.0949 | 0.0823 | 0.1045 | 0.0073 | 0.1196 | n/a | n/a | 3.1220 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0360 | 0.0364 | 0.0329 | 0.0412 | 0.0032 | 0.3877 | n/a | n/a | 4.7731 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0140 | 0.0137 | 0.0114 | 0.0157 | 0.0014 | 0.3877 | n/a | n/a | 4.7731 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.1718 | 0.1708 | 0.1606 | 0.1777 | 0.0056 | 0.3877 | n/a | n/a | 4.7731 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0339 | 0.0340 | 0.0323 | 0.0362 | 0.0013 | 0.1090 | n/a | n/a | 1.9945 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0037 | 0.0036 | 0.0034 | 0.0039 | 0.0002 | 0.1090 | n/a | n/a | 1.9945 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0677 | 0.0667 | 0.0621 | 0.0691 | 0.0026 | 0.1090 | n/a | n/a | 1.9945 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0336 | 0.0349 | 0.0311 | 0.0398 | 0.0035 | 0.1071 | n/a | n/a | 2.1831 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0036 | 0.0038 | 0.0035 | 0.0045 | 0.0004 | 0.1071 | n/a | n/a | 2.1831 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0733 | 0.0725 | 0.0676 | 0.0748 | 0.0025 | 0.1071 | n/a | n/a | 2.1831 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0336 | 0.0346 | 0.0325 | 0.0405 | 0.0030 | 0.1209 | n/a | n/a | 1.9419 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0041 | 0.0041 | 0.0029 | 0.0051 | 0.0007 | 0.1209 | n/a | n/a | 1.9419 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0653 | 0.0646 | 0.0594 | 0.0693 | 0.0035 | 0.1209 | n/a | n/a | 1.9419 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0319 | 0.0319 | 0.0288 | 0.0345 | 0.0019 | 0.1729 | n/a | n/a | 4.6827 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0055 | 0.0056 | 0.0045 | 0.0063 | 0.0007 | 0.1729 | n/a | n/a | 4.6827 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.1496 | 0.1379 | 0.1159 | 0.1508 | 0.0154 | 0.1729 | n/a | n/a | 4.6827 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0465 | 0.0479 | 0.0408 | 0.0595 | 0.0065 | 0.4167 | n/a | n/a | 4.3925 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0194 | 0.0191 | 0.0162 | 0.0218 | 0.0018 | 0.4167 | n/a | n/a | 4.3925 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.2043 | 0.2041 | 0.1868 | 0.2264 | 0.0149 | 0.4167 | n/a | n/a | 4.3925 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0607 | 0.0603 | 0.0552 | 0.0683 | 0.0047 | 0.1649 | n/a | n/a | 1.2029 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0100 | 0.0101 | 0.0084 | 0.0116 | 0.0011 | 0.1649 | n/a | n/a | 1.2029 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0730 | 0.0734 | 0.0684 | 0.0771 | 0.0029 | 0.1649 | n/a | n/a | 1.2029 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0422 | 0.0426 | 0.0384 | 0.0489 | 0.0037 | 0.1540 | n/a | n/a | 1.6044 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0065 | 0.0065 | 0.0059 | 0.0069 | 0.0003 | 0.1540 | n/a | n/a | 1.6044 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0677 | 0.0684 | 0.0628 | 0.0743 | 0.0038 | 0.1540 | n/a | n/a | 1.6044 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0473 | 0.0491 | 0.0467 | 0.0558 | 0.0034 | 0.1688 | n/a | n/a | 1.5414 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0080 | 0.0080 | 0.0073 | 0.0086 | 0.0004 | 0.1688 | n/a | n/a | 1.5414 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0730 | 0.0718 | 0.0641 | 0.0763 | 0.0042 | 0.1688 | n/a | n/a | 1.5414 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0389 | 0.0396 | 0.0377 | 0.0435 | 0.0021 | 0.2635 | n/a | n/a | 3.7694 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0102 | 0.0118 | 0.0086 | 0.0160 | 0.0027 | 0.2635 | n/a | n/a | 3.7694 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.1465 | 0.1462 | 0.1371 | 0.1557 | 0.0068 | 0.2635 | n/a | n/a | 3.7694 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0746 | 0.0720 | 0.0651 | 0.0787 | 0.0052 | 0.3416 | n/a | n/a | 3.4343 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0255 | 0.0271 | 0.0239 | 0.0319 | 0.0032 | 0.3416 | n/a | n/a | 3.4343 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.2563 | 0.2492 | 0.2235 | 0.2656 | 0.0155 | 0.3416 | n/a | n/a | 3.4343 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.1339 | 0.1355 | 0.1276 | 0.1423 | 0.0056 | 0.1687 | n/a | n/a | 0.5627 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0226 | 0.0225 | 0.0217 | 0.0236 | 0.0007 | 0.1687 | n/a | n/a | 0.5627 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0753 | 0.0770 | 0.0714 | 0.0877 | 0.0058 | 0.1687 | n/a | n/a | 0.5627 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0642 | 0.0614 | 0.0509 | 0.0707 | 0.0076 | 0.2052 | n/a | n/a | 1.2354 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0132 | 0.0131 | 0.0128 | 0.0136 | 0.0003 | 0.2052 | n/a | n/a | 1.2354 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0794 | 0.0872 | 0.0678 | 0.1181 | 0.0173 | 0.2052 | n/a | n/a | 1.2354 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0754 | 0.0757 | 0.0723 | 0.0816 | 0.0033 | 0.2126 | n/a | n/a | 0.8933 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0160 | 0.0162 | 0.0148 | 0.0185 | 0.0013 | 0.2126 | n/a | n/a | 0.8933 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0673 | 0.0756 | 0.0643 | 0.1059 | 0.0155 | 0.2126 | n/a | n/a | 0.8933 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0443 | 0.0440 | 0.0429 | 0.0447 | 0.0007 | 0.5006 | n/a | n/a | 3.0481 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0222 | 0.0215 | 0.0182 | 0.0228 | 0.0017 | 0.5006 | n/a | n/a | 3.0481 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.1350 | 0.1440 | 0.1267 | 0.1842 | 0.0205 | 0.5006 | n/a | n/a | 3.0481 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1710 | 0.1668 | 0.1462 | 0.1849 | 0.0140 | 0.4108 | n/a | n/a | 1.2545 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0703 | 0.0686 | 0.0623 | 0.0712 | 0.0033 | 0.4108 | n/a | n/a | 1.2545 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.2146 | 0.2176 | 0.2012 | 0.2324 | 0.0109 | 0.4108 | n/a | n/a | 1.2545 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3740 | 0.3720 | 0.3574 | 0.3873 | 0.0107 | 0.1881 | n/a | n/a | 0.1880 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0704 | 0.0705 | 0.0662 | 0.0775 | 0.0039 | 0.1881 | n/a | n/a | 0.1880 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0703 | 0.0753 | 0.0679 | 0.0861 | 0.0076 | 0.1881 | n/a | n/a | 0.1880 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1544 | 0.1651 | 0.1429 | 0.2145 | 0.0254 | 0.2754 | n/a | n/a | 0.5315 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0425 | 0.0428 | 0.0407 | 0.0451 | 0.0014 | 0.2754 | n/a | n/a | 0.5315 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0820 | 0.0857 | 0.0795 | 0.1000 | 0.0075 | 0.2754 | n/a | n/a | 0.5315 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1933 | 0.1876 | 0.1617 | 0.2085 | 0.0197 | 0.2891 | n/a | n/a | 0.5295 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0559 | 0.0548 | 0.0478 | 0.0595 | 0.0042 | 0.2891 | n/a | n/a | 0.5295 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.1023 | 0.1011 | 0.0909 | 0.1107 | 0.0066 | 0.2891 | n/a | n/a | 0.5295 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0332 | 0.0325 | 0.0304 | 0.0338 | 0.0013 | 0.1680 | n/a | n/a | 0.8415 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0056 | 0.0055 | 0.0047 | 0.0062 | 0.0006 | 0.1680 | n/a | n/a | 0.8415 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0279 | 0.0276 | 0.0256 | 0.0292 | 0.0015 | 0.1680 | n/a | n/a | 0.8415 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0304 | 0.0314 | 0.0300 | 0.0339 | 0.0015 | 0.5982 | n/a | n/a | 0.8898 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0182 | 0.0187 | 0.0173 | 0.0205 | 0.0012 | 0.5982 | n/a | n/a | 0.8898 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0270 | 0.0262 | 0.0228 | 0.0282 | 0.0021 | 0.5982 | n/a | n/a | 0.8898 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0334 | 0.0333 | 0.0325 | 0.0344 | 0.0007 | 0.1665 | n/a | n/a | 0.9767 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0056 | 0.0057 | 0.0054 | 0.0061 | 0.0003 | 0.1665 | n/a | n/a | 0.9767 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0326 | 0.0318 | 0.0297 | 0.0335 | 0.0016 | 0.1665 | n/a | n/a | 0.9767 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0368 | 0.0381 | 0.0348 | 0.0413 | 0.0027 | 0.5966 | n/a | n/a | 0.9421 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0220 | 0.0222 | 0.0184 | 0.0260 | 0.0024 | 0.5966 | n/a | n/a | 0.9421 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0347 | 0.0348 | 0.0301 | 0.0403 | 0.0033 | 0.5966 | n/a | n/a | 0.9421 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0342 | 0.0344 | 0.0317 | 0.0388 | 0.0026 | 0.2309 | n/a | n/a | 0.9748 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0079 | 0.0074 | 0.0056 | 0.0092 | 0.0013 | 0.2309 | n/a | n/a | 0.9748 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0333 | 0.0329 | 0.0304 | 0.0351 | 0.0016 | 0.2309 | n/a | n/a | 0.9748 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0343 | 0.0354 | 0.0339 | 0.0388 | 0.0019 | 0.7336 | n/a | n/a | 0.7619 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0251 | 0.0259 | 0.0235 | 0.0302 | 0.0023 | 0.7336 | n/a | n/a | 0.7619 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0261 | 0.0269 | 0.0250 | 0.0302 | 0.0020 | 0.7336 | n/a | n/a | 0.7619 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0331 | 0.0331 | 0.0321 | 0.0342 | 0.0008 | 0.1927 | n/a | n/a | 0.9637 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0064 | 0.0061 | 0.0047 | 0.0068 | 0.0007 | 0.1927 | n/a | n/a | 0.9637 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0319 | 0.0327 | 0.0294 | 0.0385 | 0.0032 | 0.1927 | n/a | n/a | 0.9637 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0339 | 0.0341 | 0.0328 | 0.0360 | 0.0011 | 0.5506 | n/a | n/a | 0.8614 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0187 | 0.0188 | 0.0177 | 0.0195 | 0.0006 | 0.5506 | n/a | n/a | 0.8614 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0292 | 0.0292 | 0.0274 | 0.0307 | 0.0013 | 0.5506 | n/a | n/a | 0.8614 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0305 | 0.0317 | 0.0300 | 0.0348 | 0.0019 | 0.1907 | n/a | n/a | 0.9414 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0058 | 0.0064 | 0.0055 | 0.0077 | 0.0009 | 0.1907 | n/a | n/a | 0.9414 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0287 | 0.0297 | 0.0264 | 0.0337 | 0.0029 | 0.1907 | n/a | n/a | 0.9414 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0344 | 0.0349 | 0.0334 | 0.0366 | 0.0013 | 0.6447 | n/a | n/a | 0.8044 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0222 | 0.0219 | 0.0198 | 0.0250 | 0.0018 | 0.6447 | n/a | n/a | 0.8044 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0276 | 0.0296 | 0.0231 | 0.0438 | 0.0074 | 0.6447 | n/a | n/a | 0.8044 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0388 | 0.0389 | 0.0382 | 0.0403 | 0.0007 | 0.2114 | n/a | n/a | 0.7663 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0082 | 0.0087 | 0.0078 | 0.0113 | 0.0013 | 0.2114 | n/a | n/a | 0.7663 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0297 | 0.0293 | 0.0268 | 0.0309 | 0.0015 | 0.2114 | n/a | n/a | 0.7663 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0381 | 0.0381 | 0.0366 | 0.0400 | 0.0012 | 0.7157 | n/a | n/a | 0.9184 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0273 | 0.0283 | 0.0255 | 0.0332 | 0.0029 | 0.7157 | n/a | n/a | 0.9184 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0350 | 0.0396 | 0.0329 | 0.0581 | 0.0094 | 0.7157 | n/a | n/a | 0.9184 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0589 | 0.0585 | 0.0556 | 0.0608 | 0.0022 | 0.2670 | n/a | n/a | 0.7541 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0157 | 0.0165 | 0.0140 | 0.0214 | 0.0026 | 0.2670 | n/a | n/a | 0.7541 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0445 | 0.0441 | 0.0381 | 0.0468 | 0.0031 | 0.2670 | n/a | n/a | 0.7541 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0585 | 0.0578 | 0.0542 | 0.0600 | 0.0021 | 0.7915 | n/a | n/a | 0.6636 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0463 | 0.0467 | 0.0388 | 0.0598 | 0.0073 | 0.7915 | n/a | n/a | 0.6636 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0388 | 0.0389 | 0.0335 | 0.0431 | 0.0035 | 0.7915 | n/a | n/a | 0.6636 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0342 | 0.0337 | 0.0318 | 0.0353 | 0.0013 | 0.1338 | n/a | n/a | 0.8565 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0046 | 0.0058 | 0.0040 | 0.0081 | 0.0018 | 0.1338 | n/a | n/a | 0.8565 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0293 | 0.0324 | 0.0288 | 0.0388 | 0.0042 | 0.1338 | n/a | n/a | 0.8565 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0239 | 0.0245 | 0.0220 | 0.0280 | 0.0020 | 1.3236 | n/a | n/a | 1.2918 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0316 | 0.0321 | 0.0290 | 0.0367 | 0.0025 | 1.3236 | n/a | n/a | 1.2918 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0309 | 0.0301 | 0.0281 | 0.0318 | 0.0016 | 1.3236 | n/a | n/a | 1.2918 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0386 | 0.0393 | 0.0366 | 0.0426 | 0.0022 | 0.2835 | n/a | n/a | 0.9107 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0109 | 0.0107 | 0.0095 | 0.0118 | 0.0008 | 0.2835 | n/a | n/a | 0.9107 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0351 | 0.0353 | 0.0319 | 0.0384 | 0.0026 | 0.2835 | n/a | n/a | 0.9107 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0342 | 0.0347 | 0.0297 | 0.0392 | 0.0036 | 1.0481 | n/a | n/a | 1.0308 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0359 | 0.0347 | 0.0298 | 0.0385 | 0.0031 | 1.0481 | n/a | n/a | 1.0308 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0353 | 0.0380 | 0.0346 | 0.0455 | 0.0041 | 1.0481 | n/a | n/a | 1.0308 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0594 | 0.0592 | 0.0561 | 0.0623 | 0.0022 | 0.3170 | n/a | n/a | 0.5472 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0188 | 0.0189 | 0.0177 | 0.0214 | 0.0013 | 0.3170 | n/a | n/a | 0.5472 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0325 | 0.0383 | 0.0321 | 0.0475 | 0.0074 | 0.3170 | n/a | n/a | 0.5472 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0515 | 0.0497 | 0.0451 | 0.0537 | 0.0035 | 0.8380 | n/a | n/a | 0.7195 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0432 | 0.0441 | 0.0343 | 0.0639 | 0.0106 | 0.8380 | n/a | n/a | 0.7195 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0371 | 0.0374 | 0.0317 | 0.0422 | 0.0040 | 0.8380 | n/a | n/a | 0.7195 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.1276 | 0.1270 | 0.1249 | 0.1286 | 0.0014 | 0.3751 | n/a | n/a | 0.7962 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0479 | 0.0548 | 0.0354 | 0.0876 | 0.0178 | 0.3751 | n/a | n/a | 0.7962 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.1016 | 0.1053 | 0.0967 | 0.1183 | 0.0083 | 0.3751 | n/a | n/a | 0.7962 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.1132 | 0.1134 | 0.1092 | 0.1184 | 0.0039 | 0.4970 | n/a | n/a | 0.4143 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0563 | 0.0568 | 0.0484 | 0.0646 | 0.0059 | 0.4970 | n/a | n/a | 0.4143 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0469 | 0.0712 | 0.0361 | 0.1794 | 0.0544 | 0.4970 | n/a | n/a | 0.4143 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0401 | 0.0428 | 0.0397 | 0.0482 | 0.0036 | 0.0828 | n/a | n/a | 0.6387 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0033 | 0.0041 | 0.0029 | 0.0056 | 0.0012 | 0.0828 | n/a | n/a | 0.6387 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0256 | 0.0268 | 0.0217 | 0.0335 | 0.0039 | 0.0828 | n/a | n/a | 0.6387 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1839 | 0.2009 | 0.1660 | 0.2861 | 0.0433 | 0.1469 | n/a | n/a | 0.3592 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0270 | 0.0270 | 0.0229 | 0.0304 | 0.0029 | 0.1469 | n/a | n/a | 0.3592 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0661 | 0.1104 | 0.0646 | 0.2564 | 0.0742 | 0.1469 | n/a | n/a | 0.3592 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.5188 | 0.5410 | 0.4871 | 0.6505 | 0.0589 | 0.8722 | n/a | n/a | 0.2814 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.4525 | 0.4900 | 0.3073 | 0.6526 | 0.1255 | 0.8722 | n/a | n/a | 0.2814 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1460 | 0.2177 | 0.1405 | 0.4597 | 0.1228 | 0.8722 | n/a | n/a | 0.2814 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 3.4714 | 3.4362 | 3.2322 | 3.5864 | 0.1237 | 0.2234 | n/a | n/a | 0.0972 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.7755 | 1.1548 | 0.5949 | 2.8398 | 0.8522 | 0.2234 | n/a | n/a | 0.0972 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.3376 | 0.3384 | 0.2751 | 0.4107 | 0.0441 | 0.2234 | n/a | n/a | 0.0972 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.3263 | 0.3392 | 0.2983 | 0.3856 | 0.0339 | 10.7889 | n/a | n/a | 0.5738 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 3.5200 | 3.6822 | 3.1130 | 4.9095 | 0.6320 | 10.7889 | n/a | n/a | 0.5738 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1872 | 0.1868 | 0.1735 | 0.1965 | 0.0075 | 10.7889 | n/a | n/a | 0.5738 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 8.3772 | 8.3692 | 7.9608 | 8.9948 | 0.3842 | 20.7797 | n/a | n/a | 0.0595 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 174.0763 | 172.2829 | 161.4306 | 183.6853 | 9.0943 | 20.7797 | n/a | n/a | 0.0595 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.4984 | 0.6313 | 0.4391 | 1.2307 | 0.3020 | 20.7797 | n/a | n/a | 0.0595 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0535 | 0.0536 | 0.0492 | 0.0594 | 0.0036 | 7.9357 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.4245 | 0.4245 | 0.3857 | 0.4774 | 0.0305 | 7.9357 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0520 | 0.0538 | 0.0505 | 0.0589 | 0.0031 | 28.0110 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 1.4566 | 1.4487 | 1.3552 | 1.4911 | 0.0491 | 28.0110 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 1.0199 | 1.0244 | 0.9980 | 1.0550 | 0.0221 | 19.8557 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 20.2507 | 20.5516 | 19.4338 | 22.9265 | 1.2393 | 19.8557 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 1.1132 | 1.1171 | 1.0937 | 1.1444 | 0.0214 | 65.3624 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 72.7589 | 75.0220 | 71.1936 | 84.7716 | 4.9496 | 65.3624 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.2764 | 0.2817 | 0.2637 | 0.3164 | 0.0180 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.6979 | 0.7147 | 0.6854 | 0.7990 | 0.0426 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.5534 | 0.5570 | 0.5251 | 0.6067 | 0.0293 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 1.3145 | 1.3271 | 1.3107 | 1.3572 | 0.0189 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 2.4896 | 2.4309 | 2.1880 | 2.5296 | 0.1243 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 5.6504 | 5.6884 | 5.6382 | 5.7682 | 0.0542 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 10.2570 | 10.2155 | 9.7438 | 10.7510 | 0.3442 | 0.0386 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.3961 | 0.4217 | 0.3112 | 0.5077 | 0.0743 | 0.0386 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |
| backends | backend_info | `(1,)` | TensorStudio | 0.0070 | 0.0070 | 0.0060 | 0.0086 | 0.0009 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | available_devices | `(1,)` | TensorStudio | 0.0040 | 0.0038 | 0.0026 | 0.0045 | 0.0006 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cpu_transfer | `(1,)` | TensorStudio | 0.0034 | 0.0033 | 0.0023 | 0.0041 | 0.0006 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cuda_availability_check | `(1,)` | TensorStudio | 0.0076 | 0.0077 | 0.0061 | 0.0099 | 0.0013 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
