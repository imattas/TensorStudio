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

- TensorStudio wins versus NumPy: `18`
- TensorStudio losses versus NumPy: `74`
- TensorStudio wins versus PyTorch CPU: `73`
- TensorStudio losses versus PyTorch CPU: `19`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0024 | 0.0024 | 0.0022 | 0.0029 | 0.0003 | 0.2889 | n/a | 1.5321 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2889 | n/a | 1.5321 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0035 | 0.0037 | 0.0001 | 0.2889 | n/a | 1.5321 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0026 | 0.0001 | 0.3123 | n/a | 1.5069 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3123 | n/a | 1.5069 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | PyTorch CPU | 0.0035 | 0.0037 | 0.0035 | 0.0041 | 0.0003 | 0.3123 | n/a | 1.5069 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0024 | 0.0001 | 0.3337 | n/a | 1.5901 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3337 | n/a | 1.5901 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0033 | 0.0037 | 0.0001 | 0.3337 | n/a | 1.5901 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0023 | 0.0024 | 0.0022 | 0.0027 | 0.0002 | 0.5856 | n/a | 1.7161 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0013 | 0.0012 | 0.0007 | 0.0019 | 0.0005 | 0.5856 | n/a | 1.7161 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | PyTorch CPU | 0.0039 | 0.0043 | 0.0035 | 0.0054 | 0.0008 | 0.5856 | n/a | 1.7161 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0083 | 0.0084 | 0.0083 | 0.0087 | 0.0002 | 0.4524 | n/a | 6.7530 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0041 | 0.0001 | 0.4524 | n/a | 6.7530 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | PyTorch CPU | 0.0562 | 0.0618 | 0.0554 | 0.0741 | 0.0077 | 0.4524 | n/a | 6.7530 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0023 | 0.0024 | 0.0022 | 0.0027 | 0.0002 | 0.3052 | n/a | 2.2404 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3052 | n/a | 2.2404 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | PyTorch CPU | 0.0050 | 0.0049 | 0.0034 | 0.0073 | 0.0014 | 0.3052 | n/a | 2.2404 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0023 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.3142 | n/a | 1.6395 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3142 | n/a | 1.6395 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | PyTorch CPU | 0.0037 | 0.0037 | 0.0036 | 0.0040 | 0.0001 | 0.3142 | n/a | 1.6395 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3399 | n/a | 1.5442 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.3399 | n/a | 1.5442 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 0.3399 | n/a | 1.5442 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3285 | n/a | 1.5498 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3285 | n/a | 1.5498 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0035 | 0.0038 | 0.0001 | 0.3285 | n/a | 1.5498 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0083 | 0.0083 | 0.0082 | 0.0084 | 0.0001 | 0.4695 | n/a | 6.7727 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0040 | 0.0001 | 0.4695 | n/a | 6.7727 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | PyTorch CPU | 0.0562 | 0.0564 | 0.0559 | 0.0571 | 0.0005 | 0.4695 | n/a | 6.7727 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0000 | 0.3112 | n/a | 1.5187 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3112 | n/a | 1.5187 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0036 | 0.0001 | 0.3112 | n/a | 1.5187 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3214 | n/a | 1.5607 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3214 | n/a | 1.5607 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0034 | 0.0036 | 0.0000 | 0.3214 | n/a | 1.5607 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0025 | 0.0025 | 0.0022 | 0.0029 | 0.0002 | 0.2976 | n/a | 1.3559 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2976 | n/a | 1.3559 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | PyTorch CPU | 0.0033 | 0.0035 | 0.0033 | 0.0039 | 0.0002 | 0.2976 | n/a | 1.3559 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0024 | 0.0001 | 0.3411 | n/a | 1.5289 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3411 | n/a | 1.5289 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 0.3411 | n/a | 1.5289 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0084 | 0.0084 | 0.0083 | 0.0085 | 0.0001 | 0.4570 | n/a | 6.6211 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0039 | 0.0038 | 0.0038 | 0.0039 | 0.0001 | 0.4570 | n/a | 6.6211 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | PyTorch CPU | 0.0558 | 0.0560 | 0.0556 | 0.0572 | 0.0006 | 0.4570 | n/a | 6.6211 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.3143 | n/a | 1.5097 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.3143 | n/a | 1.5097 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | PyTorch CPU | 0.0034 | 0.0036 | 0.0034 | 0.0041 | 0.0003 | 0.3143 | n/a | 1.5097 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0026 | 0.0001 | 0.3082 | n/a | 1.5364 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3082 | n/a | 1.5364 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | PyTorch CPU | 0.0036 | 0.0037 | 0.0036 | 0.0039 | 0.0001 | 0.3082 | n/a | 1.5364 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.3197 | n/a | 1.4443 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3197 | n/a | 1.4443 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | PyTorch CPU | 0.0033 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 0.3197 | n/a | 1.4443 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0026 | 0.0001 | 0.3270 | n/a | 1.5059 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0000 | 0.3270 | n/a | 1.5059 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0034 | 0.0038 | 0.0002 | 0.3270 | n/a | 1.5059 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0086 | 0.0094 | 0.0086 | 0.0123 | 0.0014 | 0.4547 | n/a | 6.5756 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0041 | 0.0001 | 0.4547 | n/a | 6.5756 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | PyTorch CPU | 0.0568 | 0.0584 | 0.0557 | 0.0661 | 0.0039 | 0.4547 | n/a | 6.5756 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0029 | 0.0029 | 0.0026 | 0.0031 | 0.0001 | 0.3871 | n/a | 1.2228 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0011 | 0.0001 | 0.3871 | n/a | 1.2228 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | PyTorch CPU | 0.0036 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 0.3871 | n/a | 1.2228 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0027 | 0.0028 | 0.0027 | 0.0031 | 0.0002 | 0.3603 | n/a | 1.3778 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3603 | n/a | 1.3778 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | PyTorch CPU | 0.0037 | 0.0037 | 0.0037 | 0.0038 | 0.0000 | 0.3603 | n/a | 1.3778 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0026 | 0.0028 | 0.0026 | 0.0031 | 0.0002 | 0.3862 | n/a | 1.3166 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3862 | n/a | 1.3166 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0035 | 0.0037 | 0.0001 | 0.3862 | n/a | 1.3166 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0000 | 0.3687 | n/a | 1.3027 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.3687 | n/a | 1.3027 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | PyTorch CPU | 0.0037 | 0.0040 | 0.0036 | 0.0050 | 0.0005 | 0.3687 | n/a | 1.3027 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0105 | 0.0106 | 0.0103 | 0.0112 | 0.0003 | 0.5387 | n/a | 5.5539 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0056 | 0.0058 | 0.0054 | 0.0064 | 0.0004 | 0.5387 | n/a | 5.5539 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | PyTorch CPU | 0.0581 | 0.0581 | 0.0569 | 0.0601 | 0.0012 | 0.5387 | n/a | 5.5539 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0042 | 0.0042 | 0.0042 | 0.0043 | 0.0000 | 0.3466 | n/a | 1.0941 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.3466 | n/a | 1.0941 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | PyTorch CPU | 0.0046 | 0.0048 | 0.0044 | 0.0054 | 0.0004 | 0.3466 | n/a | 1.0941 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0044 | 0.0044 | 0.0043 | 0.0047 | 0.0001 | 0.3353 | n/a | 1.0412 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.3353 | n/a | 1.0412 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | PyTorch CPU | 0.0046 | 0.0047 | 0.0045 | 0.0054 | 0.0004 | 0.3353 | n/a | 1.0412 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0043 | 0.0043 | 0.0042 | 0.0045 | 0.0001 | 0.3575 | n/a | 1.0164 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.3575 | n/a | 1.0164 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | PyTorch CPU | 0.0044 | 0.0044 | 0.0043 | 0.0046 | 0.0001 | 0.3575 | n/a | 1.0164 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0050 | 0.0051 | 0.0050 | 0.0053 | 0.0001 | 0.3355 | n/a | 0.9131 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.3355 | n/a | 0.9131 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | PyTorch CPU | 0.0046 | 0.0049 | 0.0044 | 0.0060 | 0.0006 | 0.3355 | n/a | 0.9131 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0176 | 0.0176 | 0.0172 | 0.0179 | 0.0002 | 0.5035 | n/a | 3.4611 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0089 | 0.0089 | 0.0084 | 0.0095 | 0.0004 | 0.5035 | n/a | 3.4611 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | PyTorch CPU | 0.0610 | 0.0614 | 0.0609 | 0.0629 | 0.0007 | 0.5035 | n/a | 3.4611 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0106 | 0.0108 | 0.0105 | 0.0115 | 0.0003 | 0.3373 | n/a | 0.6826 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0000 | 0.3373 | n/a | 0.6826 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | PyTorch CPU | 0.0073 | 0.0074 | 0.0072 | 0.0077 | 0.0002 | 0.3373 | n/a | 0.6826 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0100 | 0.0101 | 0.0099 | 0.0106 | 0.0002 | 0.3762 | n/a | 0.7247 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0038 | 0.0000 | 0.3762 | n/a | 0.7247 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | PyTorch CPU | 0.0072 | 0.0073 | 0.0072 | 0.0074 | 0.0001 | 0.3762 | n/a | 0.7247 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0094 | 0.0094 | 0.0093 | 0.0095 | 0.0001 | 0.3963 | n/a | 0.7654 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0038 | 0.0001 | 0.3963 | n/a | 0.7654 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | PyTorch CPU | 0.0072 | 0.0073 | 0.0071 | 0.0080 | 0.0004 | 0.3963 | n/a | 0.7654 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0126 | 0.0126 | 0.0125 | 0.0127 | 0.0001 | 0.3333 | n/a | 0.5789 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0042 | 0.0042 | 0.0042 | 0.0000 | 0.3333 | n/a | 0.5789 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | PyTorch CPU | 0.0073 | 0.0075 | 0.0071 | 0.0085 | 0.0005 | 0.3333 | n/a | 0.5789 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0430 | 0.0432 | 0.0426 | 0.0439 | 0.0004 | 0.4410 | n/a | 1.6188 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0190 | 0.0193 | 0.0186 | 0.0210 | 0.0009 | 0.4410 | n/a | 1.6188 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | PyTorch CPU | 0.0697 | 0.0704 | 0.0691 | 0.0724 | 0.0013 | 0.4410 | n/a | 1.6188 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.8762 | n/a | 14.2346 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.8762 | n/a | 14.2346 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1,)` | PyTorch CPU | 0.0218 | 0.0220 | 0.0216 | 0.0231 | 0.0005 | 0.8762 | n/a | 14.2346 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0021 | 0.0002 | 2.4006 | n/a | 38.8278 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0036 | 0.0036 | 0.0034 | 0.0038 | 0.0001 | 2.4006 | n/a | 38.8278 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | PyTorch CPU | 0.0581 | 0.0587 | 0.0576 | 0.0613 | 0.0014 | 2.4006 | n/a | 38.8278 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.8224 | n/a | 14.4568 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.8224 | n/a | 14.4568 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0213 | 0.0227 | 0.0004 | 0.8224 | n/a | 14.4568 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 0.8142 | n/a | 14.2076 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.8142 | n/a | 14.2076 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0215 | 0.0220 | 0.0002 | 0.8142 | n/a | 14.2076 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 0.8962 | n/a | 13.7611 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0014 | 0.0012 | 0.0014 | 0.0001 | 0.8962 | n/a | 13.7611 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0217 | 0.0220 | 0.0001 | 0.8962 | n/a | 13.7611 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.8787 | n/a | 14.2937 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.8787 | n/a | 14.2937 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(8,)` | PyTorch CPU | 0.0217 | 0.0217 | 0.0214 | 0.0219 | 0.0002 | 0.8787 | n/a | 14.2937 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 2.3633 | n/a | 38.0353 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0037 | 0.0001 | 2.3633 | n/a | 38.0353 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | PyTorch CPU | 0.0586 | 0.0587 | 0.0584 | 0.0590 | 0.0002 | 2.3633 | n/a | 38.0353 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.8052 | n/a | 14.0745 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0013 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.8052 | n/a | 14.0745 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | PyTorch CPU | 0.0219 | 0.0217 | 0.0215 | 0.0219 | 0.0002 | 0.8052 | n/a | 14.0745 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.7623 | n/a | 13.4623 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7623 | n/a | 13.4623 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(8,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0217 | 0.0221 | 0.0002 | 0.7623 | n/a | 13.4623 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.7534 | n/a | 13.0485 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.7534 | n/a | 13.0485 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(8,)` | PyTorch CPU | 0.0217 | 0.0217 | 0.0216 | 0.0219 | 0.0001 | 0.7534 | n/a | 13.0485 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | 0.9341 | n/a | 14.2691 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0014 | 0.0016 | 0.0013 | 0.0024 | 0.0004 | 0.9341 | n/a | 14.2691 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(32,)` | PyTorch CPU | 0.0215 | 0.0215 | 0.0212 | 0.0218 | 0.0002 | 0.9341 | n/a | 14.2691 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0017 | 0.0017 | 0.0017 | 0.0019 | 0.0001 | 2.1046 | n/a | 33.9043 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0040 | 0.0002 | 2.1046 | n/a | 33.9043 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | PyTorch CPU | 0.0583 | 0.0585 | 0.0573 | 0.0607 | 0.0012 | 2.1046 | n/a | 33.9043 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0023 | 0.0002 | 0.7068 | n/a | 11.8996 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0015 | 0.0001 | 0.7068 | n/a | 11.8996 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | PyTorch CPU | 0.0217 | 0.0216 | 0.0213 | 0.0219 | 0.0002 | 0.7068 | n/a | 11.8996 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0022 | 0.0002 | 0.7976 | n/a | 12.3133 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0014 | 0.0015 | 0.0013 | 0.0020 | 0.0003 | 0.7976 | n/a | 12.3133 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(32,)` | PyTorch CPU | 0.0217 | 0.0217 | 0.0214 | 0.0221 | 0.0002 | 0.7976 | n/a | 12.3133 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0018 | 0.0000 | 0.7302 | n/a | 12.3661 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.7302 | n/a | 12.3661 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(32,)` | PyTorch CPU | 0.0219 | 0.0220 | 0.0218 | 0.0223 | 0.0002 | 0.7302 | n/a | 12.3661 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0018 | 0.0001 | 0.8624 | n/a | 13.6349 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.8624 | n/a | 13.6349 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(128,)` | PyTorch CPU | 0.0216 | 0.0216 | 0.0216 | 0.0218 | 0.0001 | 0.8624 | n/a | 13.6349 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0024 | 0.0025 | 0.0024 | 0.0026 | 0.0001 | 1.4899 | n/a | 24.0753 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0041 | 0.0002 | 1.4899 | n/a | 24.0753 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | PyTorch CPU | 0.0589 | 0.0588 | 0.0581 | 0.0592 | 0.0004 | 1.4899 | n/a | 24.0753 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0028 | 0.0028 | 0.0026 | 0.0032 | 0.0002 | 0.6218 | n/a | 7.8736 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0018 | 0.0017 | 0.0015 | 0.0020 | 0.0002 | 0.6218 | n/a | 7.8736 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | PyTorch CPU | 0.0222 | 0.0222 | 0.0221 | 0.0225 | 0.0002 | 0.6218 | n/a | 7.8736 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 0.5917 | n/a | 9.5093 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.5917 | n/a | 9.5093 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(128,)` | PyTorch CPU | 0.0223 | 0.0223 | 0.0221 | 0.0226 | 0.0002 | 0.5917 | n/a | 9.5093 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0000 | 0.5778 | n/a | 9.2412 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0018 | 0.0002 | 0.5778 | n/a | 9.2412 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(128,)` | PyTorch CPU | 0.0226 | 0.0227 | 0.0221 | 0.0234 | 0.0004 | 0.5778 | n/a | 9.2412 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0000 | 0.8679 | n/a | 10.0649 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0021 | 0.0001 | 0.8679 | n/a | 10.0649 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | PyTorch CPU | 0.0221 | 0.0222 | 0.0220 | 0.0226 | 0.0002 | 0.8679 | n/a | 10.0649 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0089 | 0.0090 | 0.0088 | 0.0093 | 0.0002 | 0.6574 | n/a | 6.7573 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0059 | 0.0059 | 0.0058 | 0.0062 | 0.0001 | 0.6574 | n/a | 6.7573 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | PyTorch CPU | 0.0602 | 0.0610 | 0.0602 | 0.0630 | 0.0011 | 0.6574 | n/a | 6.7573 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0133 | 0.0135 | 0.0132 | 0.0145 | 0.0005 | 0.2824 | n/a | 1.7958 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.2824 | n/a | 1.7958 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | PyTorch CPU | 0.0239 | 0.0240 | 0.0238 | 0.0246 | 0.0003 | 0.2824 | n/a | 1.7958 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0081 | 0.0081 | 0.0079 | 0.0082 | 0.0001 | 0.3238 | n/a | 2.8090 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0000 | 0.3238 | n/a | 2.8090 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | PyTorch CPU | 0.0228 | 0.0227 | 0.0223 | 0.0231 | 0.0003 | 0.3238 | n/a | 2.8090 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0085 | 0.0085 | 0.0083 | 0.0087 | 0.0001 | 0.3488 | n/a | 2.6909 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0001 | 0.3488 | n/a | 2.6909 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1024,)` | PyTorch CPU | 0.0228 | 0.0231 | 0.0228 | 0.0240 | 0.0005 | 0.3488 | n/a | 2.6909 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0046 | 0.0046 | 0.0045 | 0.0046 | 0.0000 | 0.7046 | n/a | 4.9818 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0032 | 0.0032 | 0.0032 | 0.0034 | 0.0001 | 0.7046 | n/a | 4.9818 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | PyTorch CPU | 0.0227 | 0.0227 | 0.0226 | 0.0228 | 0.0001 | 0.7046 | n/a | 4.9818 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0315 | 0.0318 | 0.0314 | 0.0331 | 0.0007 | 0.3427 | n/a | 2.9753 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0108 | 0.0108 | 0.0106 | 0.0110 | 0.0001 | 0.3427 | n/a | 2.9753 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | PyTorch CPU | 0.0937 | 0.0937 | 0.0825 | 0.1061 | 0.0083 | 0.3427 | n/a | 2.9753 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0937 | 0.0913 | 0.0599 | 0.1089 | 0.0171 | 0.1117 | n/a | 0.4012 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0105 | 0.0105 | 0.0104 | 0.0106 | 0.0001 | 0.1117 | n/a | 0.4012 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | PyTorch CPU | 0.0376 | 0.0386 | 0.0328 | 0.0449 | 0.0045 | 0.1117 | n/a | 0.4012 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0618 | 0.0580 | 0.0438 | 0.0638 | 0.0074 | 0.1453 | n/a | 0.7207 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0090 | 0.0100 | 0.0078 | 0.0130 | 0.0020 | 0.1453 | n/a | 0.7207 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | PyTorch CPU | 0.0446 | 0.0453 | 0.0443 | 0.0486 | 0.0016 | 0.1453 | n/a | 0.7207 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0452 | 0.0468 | 0.0342 | 0.0597 | 0.0086 | 0.1673 | n/a | 0.6498 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0076 | 0.0079 | 0.0073 | 0.0089 | 0.0006 | 0.1673 | n/a | 0.6498 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(4096,)` | PyTorch CPU | 0.0294 | 0.0295 | 0.0288 | 0.0303 | 0.0005 | 0.1673 | n/a | 0.6498 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0130 | 0.0129 | 0.0126 | 0.0131 | 0.0002 | 1.3354 | n/a | 2.8440 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0174 | 0.0168 | 0.0101 | 0.0204 | 0.0035 | 1.3354 | n/a | 2.8440 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | relu | `(16384,)` | PyTorch CPU | 0.0370 | 0.0373 | 0.0268 | 0.0510 | 0.0094 | 1.3354 | n/a | 2.8440 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1472 | 0.1720 | 0.1335 | 0.2397 | 0.0426 | 0.2534 | n/a | 0.6988 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0373 | 0.0367 | 0.0320 | 0.0402 | 0.0033 | 0.2534 | n/a | 0.6988 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | PyTorch CPU | 0.1029 | 0.1039 | 0.0826 | 0.1267 | 0.0140 | 0.2534 | n/a | 0.6988 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.4267 | 0.4004 | 0.3515 | 0.4426 | 0.0396 | 0.0933 | n/a | 0.0832 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0398 | 0.0399 | 0.0390 | 0.0411 | 0.0008 | 0.0933 | n/a | 0.0832 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| activations | tanh | `(16384,)` | PyTorch CPU | 0.0355 | 0.1071 | 0.0326 | 0.3833 | 0.1383 | 0.0933 | n/a | 0.0832 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1442 | 0.1999 | 0.1179 | 0.3294 | 0.0826 | 0.2165 | n/a | 0.3517 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0312 | 0.0309 | 0.0204 | 0.0434 | 0.0094 | 0.2165 | n/a | 0.3517 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | PyTorch CPU | 0.0507 | 0.0495 | 0.0454 | 0.0536 | 0.0030 | 0.2165 | n/a | 0.3517 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.2379 | 0.2180 | 0.1174 | 0.2551 | 0.0507 | 0.1348 | n/a | 0.2050 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0321 | 0.0312 | 0.0278 | 0.0339 | 0.0021 | 0.1348 | n/a | 0.2050 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(16384,)` | PyTorch CPU | 0.0488 | 0.0455 | 0.0348 | 0.0546 | 0.0073 | 0.1348 | n/a | 0.2050 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0031 | 0.0033 | 0.0028 | 0.0040 | 0.0004 | 1.5467 | n/a | 2.8153 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0049 | 0.0051 | 0.0048 | 0.0058 | 0.0004 | 1.5467 | n/a | 2.8153 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0088 | 0.0079 | 0.0053 | 0.0101 | 0.0019 | 1.5467 | n/a | 2.8153 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0019 | 0.0021 | 0.0018 | 0.0026 | 0.0004 | 4.3281 | n/a | 6.5819 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0080 | 0.0083 | 0.0080 | 0.0092 | 0.0005 | 4.3281 | n/a | 6.5819 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0122 | 0.0124 | 0.0122 | 0.0127 | 0.0002 | 4.3281 | n/a | 6.5819 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 1.2804 | n/a | 2.9934 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0022 | 0.0000 | 1.2804 | n/a | 2.9934 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0051 | 0.0052 | 0.0049 | 0.0057 | 0.0003 | 1.2804 | n/a | 2.9934 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 4.6527 | n/a | 7.5228 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0077 | 0.0078 | 0.0076 | 0.0080 | 0.0002 | 4.6527 | n/a | 7.5228 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0124 | 0.0130 | 0.0123 | 0.0147 | 0.0009 | 4.6527 | n/a | 7.5228 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0019 | 0.0000 | 1.1851 | n/a | 2.8026 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0022 | 0.0000 | 1.1851 | n/a | 2.8026 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0051 | 0.0053 | 0.0050 | 0.0062 | 0.0004 | 1.1851 | n/a | 2.8026 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0018 | 0.0000 | 4.2222 | n/a | 6.8059 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0077 | 0.0078 | 0.0076 | 0.0080 | 0.0001 | 4.2222 | n/a | 6.8059 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0124 | 0.0131 | 0.0123 | 0.0162 | 0.0015 | 4.2222 | n/a | 6.8059 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.9247 | n/a | 2.2082 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0021 | 0.0021 | 0.0021 | 0.0022 | 0.0000 | 0.9247 | n/a | 2.2082 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0051 | 0.0052 | 0.0050 | 0.0054 | 0.0001 | 0.9247 | n/a | 2.2082 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0028 | 0.0002 | 3.3522 | n/a | 5.4085 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0078 | 0.0081 | 0.0077 | 0.0092 | 0.0006 | 3.3522 | n/a | 5.4085 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0126 | 0.0125 | 0.0124 | 0.0127 | 0.0001 | 3.3522 | n/a | 5.4085 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0071 | 0.0071 | 0.0070 | 0.0072 | 0.0001 | 0.3597 | n/a | 0.7424 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0025 | 0.0029 | 0.0001 | 0.3597 | n/a | 0.7424 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0053 | 0.0053 | 0.0052 | 0.0056 | 0.0002 | 0.3597 | n/a | 0.7424 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0071 | 0.0071 | 0.0070 | 0.0073 | 0.0001 | 1.1641 | n/a | 1.7822 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0083 | 0.0083 | 0.0081 | 0.0084 | 0.0001 | 1.1641 | n/a | 1.7822 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0127 | 0.0126 | 0.0123 | 0.0128 | 0.0002 | 1.1641 | n/a | 1.7822 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0236 | 0.0237 | 0.0235 | 0.0239 | 0.0002 | 0.1431 | n/a | 0.2460 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0034 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 0.1431 | n/a | 0.2460 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0058 | 0.0058 | 0.0056 | 0.0059 | 0.0001 | 0.1431 | n/a | 0.2460 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0237 | 0.0237 | 0.0235 | 0.0239 | 0.0001 | 0.3910 | n/a | 0.5626 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0092 | 0.0092 | 0.0090 | 0.0094 | 0.0001 | 0.3910 | n/a | 0.5626 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0133 | 0.0133 | 0.0131 | 0.0135 | 0.0001 | 0.3910 | n/a | 0.5626 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0952 | 0.0948 | 0.0938 | 0.0953 | 0.0006 | 0.0693 | n/a | 0.0789 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0066 | 0.0067 | 0.0065 | 0.0069 | 0.0002 | 0.0693 | n/a | 0.0789 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0075 | 0.0076 | 0.0074 | 0.0077 | 0.0001 | 0.0693 | n/a | 0.0789 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0941 | 0.1000 | 0.0907 | 0.1135 | 0.0101 | 0.1495 | n/a | 0.1668 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0141 | 0.0141 | 0.0129 | 0.0154 | 0.0011 | 0.1495 | n/a | 0.1668 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0157 | 0.0158 | 0.0152 | 0.0165 | 0.0005 | 0.1495 | n/a | 0.1668 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0000 | 1.7500 | n/a | 4.2805 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0036 | 0.0034 | 0.0025 | 0.0042 | 0.0006 | 1.7500 | n/a | 4.2805 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | PyTorch CPU | 0.0088 | 0.0089 | 0.0085 | 0.0097 | 0.0004 | 1.7500 | n/a | 4.2805 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0020 | 0.0021 | 0.0020 | 0.0022 | 0.0001 | 4.0549 | n/a | 7.1000 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0081 | 0.0100 | 0.0079 | 0.0138 | 0.0026 | 4.0549 | n/a | 7.1000 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | PyTorch CPU | 0.0142 | 0.0145 | 0.0141 | 0.0156 | 0.0006 | 4.0549 | n/a | 7.1000 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0049 | 0.0049 | 0.0048 | 0.0050 | 0.0001 | 0.8183 | n/a | 1.7816 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0040 | 0.0040 | 0.0039 | 0.0042 | 0.0001 | 0.8183 | n/a | 1.7816 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | PyTorch CPU | 0.0087 | 0.0089 | 0.0085 | 0.0099 | 0.0005 | 0.8183 | n/a | 1.7816 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0030 | 0.0030 | 0.0030 | 0.0030 | 0.0000 | 3.2326 | n/a | 4.9861 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0096 | 0.0096 | 0.0094 | 0.0100 | 0.0002 | 3.2326 | n/a | 4.9861 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | PyTorch CPU | 0.0149 | 0.0150 | 0.0147 | 0.0153 | 0.0003 | 3.2326 | n/a | 4.9861 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0156 | 0.0160 | 0.0153 | 0.0172 | 0.0007 | 0.4761 | n/a | 0.7976 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0074 | 0.0075 | 0.0073 | 0.0077 | 0.0002 | 0.4761 | n/a | 0.7976 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | PyTorch CPU | 0.0125 | 0.0126 | 0.0124 | 0.0131 | 0.0003 | 0.4761 | n/a | 0.7976 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0062 | 0.0063 | 0.0059 | 0.0068 | 0.0004 | 2.1011 | n/a | 2.9137 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0131 | 0.0132 | 0.0129 | 0.0136 | 0.0002 | 2.1011 | n/a | 2.9137 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | PyTorch CPU | 0.0182 | 0.0198 | 0.0171 | 0.0263 | 0.0034 | 2.1011 | n/a | 2.9137 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0638 | 0.0640 | 0.0635 | 0.0652 | 0.0006 | 0.3357 | n/a | 0.2590 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0214 | 0.0217 | 0.0211 | 0.0225 | 0.0006 | 0.3357 | n/a | 0.2590 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | PyTorch CPU | 0.0165 | 0.0167 | 0.0131 | 0.0212 | 0.0027 | 0.3357 | n/a | 0.2590 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0183 | 0.0189 | 0.0177 | 0.0205 | 0.0011 | 1.1958 | n/a | 1.3157 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0219 | 0.0230 | 0.0213 | 0.0261 | 0.0019 | 1.1958 | n/a | 1.3157 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | PyTorch CPU | 0.0240 | 0.0263 | 0.0236 | 0.0317 | 0.0032 | 1.1958 | n/a | 1.3157 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
