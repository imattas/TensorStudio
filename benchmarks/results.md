# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.0`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: available (2.12.1+cpu)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `13`
- TensorStudio losses versus NumPy: `76`
- TensorStudio wins versus PyTorch CPU: `68`
- TensorStudio losses versus PyTorch CPU: `26`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.2886 | n/a | 1.5426 | n/a | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2886 | n/a | 1.5426 | n/a | NumPy baseline |
| elementwise | add | `(1,)` | PyTorch CPU | 0.0035 | 0.0042 | 0.0033 | 0.0057 | 0.0009 | 0.2886 | n/a | 1.5426 | n/a | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0025 | 0.0001 | 0.2954 | n/a | 1.6269 | n/a | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2954 | n/a | 1.6269 | n/a | NumPy baseline |
| elementwise | sub | `(1,)` | PyTorch CPU | 0.0037 | 0.0044 | 0.0035 | 0.0065 | 0.0011 | 0.2954 | n/a | 1.6269 | n/a | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.3074 | n/a | 1.4575 | n/a | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3074 | n/a | 1.4575 | n/a | NumPy baseline |
| elementwise | mul | `(1,)` | PyTorch CPU | 0.0033 | 0.0033 | 0.0032 | 0.0034 | 0.0001 | 0.3074 | n/a | 1.4575 | n/a | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0028 | 0.0030 | 0.0022 | 0.0042 | 0.0008 | 0.2706 | n/a | 1.2114 | n/a | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2706 | n/a | 1.2114 | n/a | NumPy baseline |
| elementwise | div | `(1,)` | PyTorch CPU | 0.0033 | 0.0034 | 0.0033 | 0.0038 | 0.0002 | 0.2706 | n/a | 1.2114 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0085 | 0.0091 | 0.0085 | 0.0116 | 0.0012 | 0.4590 | n/a | 6.5087 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0039 | 0.0042 | 0.0038 | 0.0055 | 0.0006 | 0.4590 | n/a | 6.5087 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | PyTorch CPU | 0.0556 | 0.0582 | 0.0527 | 0.0673 | 0.0053 | 0.4590 | n/a | 6.5087 | n/a | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0001 | 0.3118 | n/a | 1.4986 | n/a | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0014 | 0.0003 | 0.3118 | n/a | 1.4986 | n/a | NumPy baseline |
| elementwise | add | `(8,)` | PyTorch CPU | 0.0034 | 0.0042 | 0.0034 | 0.0072 | 0.0015 | 0.3118 | n/a | 1.4986 | n/a | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0024 | 0.0027 | 0.0023 | 0.0043 | 0.0008 | 0.6534 | n/a | 1.5163 | n/a | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0015 | 0.0014 | 0.0007 | 0.0016 | 0.0004 | 0.6534 | n/a | 1.5163 | n/a | NumPy baseline |
| elementwise | sub | `(8,)` | PyTorch CPU | 0.0036 | 0.0037 | 0.0035 | 0.0044 | 0.0003 | 0.6534 | n/a | 1.5163 | n/a | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0026 | 0.0001 | 0.3483 | n/a | 1.5105 | n/a | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0001 | 0.3483 | n/a | 1.5105 | n/a | NumPy baseline |
| elementwise | mul | `(8,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0036 | 0.0001 | 0.3483 | n/a | 1.5105 | n/a | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0043 | 0.0038 | 0.0023 | 0.0043 | 0.0008 | 0.3221 | n/a | 0.7973 | n/a | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.3221 | n/a | 0.7973 | n/a | NumPy baseline |
| elementwise | div | `(8,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0034 | 0.0000 | 0.3221 | n/a | 0.7973 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0084 | 0.0084 | 0.0083 | 0.0086 | 0.0001 | 0.4665 | n/a | 6.1822 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0040 | 0.0001 | 0.4665 | n/a | 6.1822 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | PyTorch CPU | 0.0521 | 0.0522 | 0.0518 | 0.0527 | 0.0003 | 0.4665 | n/a | 6.1822 | n/a | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0026 | 0.0001 | 0.2996 | n/a | 1.5429 | n/a | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.2996 | n/a | 1.5429 | n/a | NumPy baseline |
| elementwise | add | `(32,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0034 | 0.0037 | 0.0001 | 0.2996 | n/a | 1.5429 | n/a | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0024 | 0.0029 | 0.0023 | 0.0051 | 0.0011 | 0.2971 | n/a | 1.4938 | n/a | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2971 | n/a | 1.4938 | n/a | NumPy baseline |
| elementwise | sub | `(32,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0033 | 0.0037 | 0.0001 | 0.2971 | n/a | 1.4938 | n/a | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0001 | 0.3025 | n/a | 1.4291 | n/a | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3025 | n/a | 1.4291 | n/a | NumPy baseline |
| elementwise | mul | `(32,)` | PyTorch CPU | 0.0033 | 0.0034 | 0.0033 | 0.0036 | 0.0001 | 0.3025 | n/a | 1.4291 | n/a | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0023 | 0.0027 | 0.0022 | 0.0043 | 0.0008 | 0.3090 | n/a | 1.4560 | n/a | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0015 | 0.0003 | 0.3090 | n/a | 1.4560 | n/a | NumPy baseline |
| elementwise | div | `(32,)` | PyTorch CPU | 0.0034 | 0.0033 | 0.0033 | 0.0034 | 0.0000 | 0.3090 | n/a | 1.4560 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0086 | 0.0092 | 0.0085 | 0.0104 | 0.0009 | 0.4927 | n/a | 6.1047 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0042 | 0.0051 | 0.0038 | 0.0081 | 0.0016 | 0.4927 | n/a | 6.1047 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | PyTorch CPU | 0.0525 | 0.0541 | 0.0514 | 0.0576 | 0.0027 | 0.4927 | n/a | 6.1047 | n/a | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.2736 | n/a | 1.4439 | n/a | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2736 | n/a | 1.4439 | n/a | NumPy baseline |
| elementwise | add | `(128,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0034 | 0.0036 | 0.0001 | 0.2736 | n/a | 1.4439 | n/a | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0000 | 0.2878 | n/a | 1.4693 | n/a | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2878 | n/a | 1.4693 | n/a | NumPy baseline |
| elementwise | sub | `(128,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0035 | 0.0036 | 0.0000 | 0.2878 | n/a | 1.4693 | n/a | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0030 | 0.0030 | 0.0023 | 0.0043 | 0.0008 | 0.2722 | n/a | 2.1341 | n/a | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0008 | 0.0009 | 0.0007 | 0.0014 | 0.0003 | 0.2722 | n/a | 2.1341 | n/a | NumPy baseline |
| elementwise | mul | `(128,)` | PyTorch CPU | 0.0063 | 0.0060 | 0.0033 | 0.0084 | 0.0023 | 0.2722 | n/a | 2.1341 | n/a | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 0.3016 | n/a | 1.8958 | n/a | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.3016 | n/a | 1.8958 | n/a | NumPy baseline |
| elementwise | div | `(128,)` | PyTorch CPU | 0.0045 | 0.0052 | 0.0033 | 0.0084 | 0.0019 | 0.3016 | n/a | 1.8958 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0089 | 0.0100 | 0.0087 | 0.0129 | 0.0016 | 0.4410 | n/a | 6.1384 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0039 | 0.0040 | 0.0039 | 0.0042 | 0.0001 | 0.4410 | n/a | 6.1384 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | PyTorch CPU | 0.0545 | 0.0543 | 0.0506 | 0.0590 | 0.0030 | 0.4410 | n/a | 6.1384 | n/a | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0029 | 0.0029 | 0.0027 | 0.0031 | 0.0001 | 0.3131 | n/a | 1.2799 | n/a | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0009 | 0.0009 | 0.0009 | 0.0009 | 0.0000 | 0.3131 | n/a | 1.2799 | n/a | NumPy baseline |
| elementwise | add | `(1024,)` | PyTorch CPU | 0.0038 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.3131 | n/a | 1.2799 | n/a | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0027 | 0.0027 | 0.0027 | 0.0027 | 0.0000 | 0.3494 | n/a | 1.3776 | n/a | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0009 | 0.0014 | 0.0009 | 0.0029 | 0.0008 | 0.3494 | n/a | 1.3776 | n/a | NumPy baseline |
| elementwise | sub | `(1024,)` | PyTorch CPU | 0.0037 | 0.0044 | 0.0036 | 0.0058 | 0.0009 | 0.3494 | n/a | 1.3776 | n/a | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0029 | 0.0029 | 0.0026 | 0.0033 | 0.0002 | 0.3274 | n/a | 1.2127 | n/a | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0010 | 0.0000 | 0.3274 | n/a | 1.2127 | n/a | NumPy baseline |
| elementwise | mul | `(1024,)` | PyTorch CPU | 0.0035 | 0.0038 | 0.0035 | 0.0046 | 0.0004 | 0.3274 | n/a | 1.2127 | n/a | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0029 | 0.0032 | 0.0001 | 0.3435 | n/a | 1.1715 | n/a | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3435 | n/a | 1.1715 | n/a | NumPy baseline |
| elementwise | div | `(1024,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0041 | 0.0002 | 0.3435 | n/a | 1.1715 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0108 | 0.0107 | 0.0103 | 0.0110 | 0.0003 | 0.4953 | n/a | 4.9261 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0054 | 0.0055 | 0.0053 | 0.0058 | 0.0002 | 0.4953 | n/a | 4.9261 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | PyTorch CPU | 0.0533 | 0.0581 | 0.0531 | 0.0775 | 0.0097 | 0.4953 | n/a | 4.9261 | n/a | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0043 | 0.0043 | 0.0043 | 0.0045 | 0.0001 | 0.3421 | n/a | 1.0626 | n/a | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.3421 | n/a | 1.0626 | n/a | NumPy baseline |
| elementwise | add | `(4096,)` | PyTorch CPU | 0.0046 | 0.0046 | 0.0045 | 0.0047 | 0.0001 | 0.3421 | n/a | 1.0626 | n/a | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0085 | 0.0074 | 0.0043 | 0.0089 | 0.0018 | 0.1944 | n/a | 0.5581 | n/a | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.1944 | n/a | 0.5581 | n/a | NumPy baseline |
| elementwise | sub | `(4096,)` | PyTorch CPU | 0.0047 | 0.0047 | 0.0045 | 0.0049 | 0.0001 | 0.1944 | n/a | 0.5581 | n/a | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0042 | 0.0043 | 0.0042 | 0.0045 | 0.0001 | 0.3487 | n/a | 1.0703 | n/a | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | 0.3487 | n/a | 1.0703 | n/a | NumPy baseline |
| elementwise | mul | `(4096,)` | PyTorch CPU | 0.0045 | 0.0046 | 0.0044 | 0.0047 | 0.0001 | 0.3487 | n/a | 1.0703 | n/a | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0053 | 0.0053 | 0.0051 | 0.0057 | 0.0002 | 0.3025 | n/a | 0.8526 | n/a | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0001 | 0.3025 | n/a | 0.8526 | n/a | NumPy baseline |
| elementwise | div | `(4096,)` | PyTorch CPU | 0.0045 | 0.0046 | 0.0045 | 0.0050 | 0.0002 | 0.3025 | n/a | 0.8526 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0230 | 0.0243 | 0.0183 | 0.0359 | 0.0065 | 0.4993 | n/a | 2.7790 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0115 | 0.0126 | 0.0102 | 0.0169 | 0.0026 | 0.4993 | n/a | 2.7790 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | PyTorch CPU | 0.0639 | 0.0622 | 0.0567 | 0.0679 | 0.0042 | 0.4993 | n/a | 2.7790 | n/a | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0108 | 0.0108 | 0.0108 | 0.0109 | 0.0000 | 0.3307 | n/a | 0.6973 | n/a | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0036 | 0.0000 | 0.3307 | n/a | 0.6973 | n/a | NumPy baseline |
| elementwise | add | `(16384,)` | PyTorch CPU | 0.0075 | 0.0077 | 0.0074 | 0.0086 | 0.0004 | 0.3307 | n/a | 0.6973 | n/a | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0108 | 0.0109 | 0.0104 | 0.0120 | 0.0006 | 0.3368 | n/a | 0.6652 | n/a | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0037 | 0.0000 | 0.3368 | n/a | 0.6652 | n/a | NumPy baseline |
| elementwise | sub | `(16384,)` | PyTorch CPU | 0.0072 | 0.0073 | 0.0072 | 0.0079 | 0.0003 | 0.3368 | n/a | 0.6652 | n/a | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0094 | 0.0094 | 0.0093 | 0.0095 | 0.0001 | 0.4577 | n/a | 0.8560 | n/a | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0043 | 0.0042 | 0.0037 | 0.0044 | 0.0002 | 0.4577 | n/a | 0.8560 | n/a | NumPy baseline |
| elementwise | mul | `(16384,)` | PyTorch CPU | 0.0080 | 0.0080 | 0.0079 | 0.0082 | 0.0001 | 0.4577 | n/a | 0.8560 | n/a | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0139 | 0.0155 | 0.0127 | 0.0199 | 0.0030 | 0.3026 | n/a | 0.5307 | n/a | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0043 | 0.0042 | 0.0046 | 0.0002 | 0.3026 | n/a | 0.5307 | n/a | NumPy baseline |
| elementwise | div | `(16384,)` | PyTorch CPU | 0.0074 | 0.0074 | 0.0071 | 0.0080 | 0.0004 | 0.3026 | n/a | 0.5307 | n/a | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0752 | 0.0731 | 0.0638 | 0.0757 | 0.0046 | 0.4506 | n/a | 0.8548 | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0339 | 0.0290 | 0.0189 | 0.0360 | 0.0077 | 0.4506 | n/a | 0.8548 | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | PyTorch CPU | 0.0643 | 0.0648 | 0.0630 | 0.0675 | 0.0017 | 0.4506 | n/a | 0.8548 | n/a | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0015 | 0.0018 | 0.0015 | 0.0028 | 0.0005 | 0.8906 | n/a | 17.0187 | n/a | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.8906 | n/a | 17.0187 | n/a | NumPy baseline |
| activations | relu | `(1,)` | PyTorch CPU | 0.0254 | 0.0270 | 0.0214 | 0.0352 | 0.0054 | 0.8906 | n/a | 17.0187 | n/a | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0016 | 0.0020 | 0.0015 | 0.0034 | 0.0007 | 2.1844 | n/a | 36.5668 | n/a | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0034 | 0.0035 | 0.0034 | 0.0038 | 0.0002 | 2.1844 | n/a | 36.5668 | n/a | NumPy baseline |
| activations | sigmoid | `(1,)` | PyTorch CPU | 0.0570 | 0.0574 | 0.0534 | 0.0640 | 0.0038 | 2.1844 | n/a | 36.5668 | n/a | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0020 | 0.0022 | 0.0015 | 0.0035 | 0.0007 | 0.6185 | n/a | 10.9567 | n/a | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.6185 | n/a | 10.9567 | n/a | NumPy baseline |
| activations | tanh | `(1,)` | PyTorch CPU | 0.0219 | 0.0225 | 0.0215 | 0.0255 | 0.0015 | 0.6185 | n/a | 10.9567 | n/a | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.7794 | n/a | 15.3604 | n/a | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.7794 | n/a | 15.3604 | n/a | NumPy baseline |
| activations | exp | `(1,)` | PyTorch CPU | 0.0239 | 0.0239 | 0.0218 | 0.0253 | 0.0013 | 0.7794 | n/a | 15.3604 | n/a | reference |
| activations | log | `(1,)` | TensorStudio | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.7380 | n/a | 13.1260 | n/a | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.7380 | n/a | 13.1260 | n/a | NumPy baseline |
| activations | log | `(1,)` | PyTorch CPU | 0.0219 | 0.0233 | 0.0216 | 0.0287 | 0.0027 | 0.7380 | n/a | 13.1260 | n/a | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.9174 | n/a | 14.8571 | n/a | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0015 | 0.0013 | 0.0018 | 0.0002 | 0.9174 | n/a | 14.8571 | n/a | NumPy baseline |
| activations | relu | `(8,)` | PyTorch CPU | 0.0222 | 0.0239 | 0.0218 | 0.0268 | 0.0023 | 0.9174 | n/a | 14.8571 | n/a | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 2.1224 | n/a | 37.5254 | n/a | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0034 | 0.0036 | 0.0033 | 0.0039 | 0.0002 | 2.1224 | n/a | 37.5254 | n/a | NumPy baseline |
| activations | sigmoid | `(8,)` | PyTorch CPU | 0.0606 | 0.0658 | 0.0527 | 0.0945 | 0.0152 | 2.1224 | n/a | 37.5254 | n/a | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0028 | 0.0032 | 0.0016 | 0.0049 | 0.0014 | 0.4364 | n/a | 8.1661 | n/a | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.4364 | n/a | 8.1661 | n/a | NumPy baseline |
| activations | tanh | `(8,)` | PyTorch CPU | 0.0232 | 0.0262 | 0.0221 | 0.0338 | 0.0046 | 0.4364 | n/a | 8.1661 | n/a | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0016 | 0.0018 | 0.0016 | 0.0028 | 0.0005 | 0.7445 | n/a | 15.7422 | n/a | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.7445 | n/a | 15.7422 | n/a | NumPy baseline |
| activations | exp | `(8,)` | PyTorch CPU | 0.0253 | 0.0260 | 0.0223 | 0.0322 | 0.0036 | 0.7445 | n/a | 15.7422 | n/a | reference |
| activations | log | `(8,)` | TensorStudio | 0.0016 | 0.0018 | 0.0016 | 0.0025 | 0.0004 | 0.7520 | n/a | 13.8021 | n/a | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0001 | 0.7520 | n/a | 13.8021 | n/a | NumPy baseline |
| activations | log | `(8,)` | PyTorch CPU | 0.0224 | 0.0236 | 0.0216 | 0.0281 | 0.0025 | 0.7520 | n/a | 13.8021 | n/a | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.9301 | n/a | 14.3140 | n/a | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.9301 | n/a | 14.3140 | n/a | NumPy baseline |
| activations | relu | `(32,)` | PyTorch CPU | 0.0213 | 0.0221 | 0.0212 | 0.0255 | 0.0017 | 0.9301 | n/a | 14.3140 | n/a | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 2.7548 | n/a | 35.6408 | n/a | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0049 | 0.0053 | 0.0034 | 0.0084 | 0.0018 | 2.7548 | n/a | 35.6408 | n/a | NumPy baseline |
| activations | sigmoid | `(32,)` | PyTorch CPU | 0.0638 | 0.0641 | 0.0532 | 0.0840 | 0.0109 | 2.7548 | n/a | 35.6408 | n/a | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0019 | 0.0021 | 0.0018 | 0.0032 | 0.0005 | 0.7068 | n/a | 12.2096 | n/a | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0019 | 0.0013 | 0.0031 | 0.0008 | 0.7068 | n/a | 12.2096 | n/a | NumPy baseline |
| activations | tanh | `(32,)` | PyTorch CPU | 0.0229 | 0.0258 | 0.0218 | 0.0388 | 0.0065 | 0.7068 | n/a | 12.2096 | n/a | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0023 | 0.0024 | 0.0020 | 0.0033 | 0.0005 | 0.7034 | n/a | 11.3295 | n/a | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0016 | 0.0016 | 0.0014 | 0.0019 | 0.0002 | 0.7034 | n/a | 11.3295 | n/a | NumPy baseline |
| activations | exp | `(32,)` | PyTorch CPU | 0.0257 | 0.0257 | 0.0213 | 0.0336 | 0.0044 | 0.7034 | n/a | 11.3295 | n/a | reference |
| activations | log | `(32,)` | TensorStudio | 0.0040 | 0.0033 | 0.0018 | 0.0041 | 0.0010 | 0.5416 | n/a | 5.4579 | n/a | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0022 | 0.0022 | 0.0013 | 0.0030 | 0.0007 | 0.5416 | n/a | 5.4579 | n/a | NumPy baseline |
| activations | log | `(32,)` | PyTorch CPU | 0.0220 | 0.0231 | 0.0213 | 0.0265 | 0.0020 | 0.5416 | n/a | 5.4579 | n/a | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0018 | 0.0001 | 1.0326 | n/a | 14.6298 | n/a | win vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0017 | 0.0020 | 0.0014 | 0.0029 | 0.0006 | 1.0326 | n/a | 14.6298 | n/a | NumPy baseline |
| activations | relu | `(128,)` | PyTorch CPU | 0.0237 | 0.0234 | 0.0213 | 0.0256 | 0.0014 | 1.0326 | n/a | 14.6298 | n/a | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0025 | 0.0026 | 0.0025 | 0.0027 | 0.0001 | 1.4501 | n/a | 22.9591 | n/a | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0037 | 0.0037 | 0.0035 | 0.0038 | 0.0001 | 1.4501 | n/a | 22.9591 | n/a | NumPy baseline |
| activations | sigmoid | `(128,)` | PyTorch CPU | 0.0580 | 0.0571 | 0.0533 | 0.0608 | 0.0029 | 1.4501 | n/a | 22.9591 | n/a | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0027 | 0.0038 | 0.0026 | 0.0060 | 0.0014 | 0.5416 | n/a | 8.5790 | n/a | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0020 | 0.0015 | 0.0033 | 0.0007 | 0.5416 | n/a | 8.5790 | n/a | NumPy baseline |
| activations | tanh | `(128,)` | PyTorch CPU | 0.0235 | 0.0237 | 0.0218 | 0.0272 | 0.0020 | 0.5416 | n/a | 8.5790 | n/a | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0024 | 0.0025 | 0.0024 | 0.0027 | 0.0001 | 0.5630 | n/a | 9.3226 | n/a | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0017 | 0.0013 | 0.0030 | 0.0007 | 0.5630 | n/a | 9.3226 | n/a | NumPy baseline |
| activations | exp | `(128,)` | PyTorch CPU | 0.0227 | 0.0229 | 0.0222 | 0.0240 | 0.0007 | 0.5630 | n/a | 9.3226 | n/a | reference |
| activations | log | `(128,)` | TensorStudio | 0.0024 | 0.0025 | 0.0024 | 0.0029 | 0.0002 | 0.5696 | n/a | 9.0961 | n/a | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0019 | 0.0002 | 0.5696 | n/a | 9.0961 | n/a | NumPy baseline |
| activations | log | `(128,)` | PyTorch CPU | 0.0221 | 0.0237 | 0.0219 | 0.0278 | 0.0023 | 0.5696 | n/a | 9.0961 | n/a | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0024 | 0.0001 | 1.2343 | n/a | 10.3822 | n/a | win vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0019 | 0.0038 | 0.0007 | 1.2343 | n/a | 10.3822 | n/a | NumPy baseline |
| activations | relu | `(1024,)` | PyTorch CPU | 0.0226 | 0.0223 | 0.0216 | 0.0227 | 0.0005 | 1.2343 | n/a | 10.3822 | n/a | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0091 | 0.0103 | 0.0089 | 0.0152 | 0.0025 | 0.6245 | n/a | 6.4452 | n/a | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0057 | 0.0057 | 0.0055 | 0.0061 | 0.0002 | 0.6245 | n/a | 6.4452 | n/a | NumPy baseline |
| activations | sigmoid | `(1024,)` | PyTorch CPU | 0.0586 | 0.0599 | 0.0554 | 0.0664 | 0.0036 | 0.6245 | n/a | 6.4452 | n/a | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0142 | 0.0141 | 0.0132 | 0.0153 | 0.0008 | 0.2720 | n/a | 1.6805 | n/a | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0039 | 0.0038 | 0.0037 | 0.0039 | 0.0001 | 0.2720 | n/a | 1.6805 | n/a | NumPy baseline |
| activations | tanh | `(1024,)` | PyTorch CPU | 0.0238 | 0.0239 | 0.0236 | 0.0242 | 0.0002 | 0.2720 | n/a | 1.6805 | n/a | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0079 | 0.0083 | 0.0078 | 0.0095 | 0.0007 | 0.3203 | n/a | 2.9125 | n/a | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0025 | 0.0025 | 0.0025 | 0.0026 | 0.0000 | 0.3203 | n/a | 2.9125 | n/a | NumPy baseline |
| activations | exp | `(1024,)` | PyTorch CPU | 0.0231 | 0.0239 | 0.0224 | 0.0278 | 0.0020 | 0.3203 | n/a | 2.9125 | n/a | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0086 | 0.0086 | 0.0085 | 0.0088 | 0.0001 | 0.3365 | n/a | 2.6665 | n/a | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0031 | 0.0029 | 0.0037 | 0.0003 | 0.3365 | n/a | 2.6665 | n/a | NumPy baseline |
| activations | log | `(1024,)` | PyTorch CPU | 0.0230 | 0.0284 | 0.0224 | 0.0432 | 0.0080 | 0.3365 | n/a | 2.6665 | n/a | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0043 | 0.0043 | 0.0043 | 0.0044 | 0.0000 | 0.7174 | n/a | 5.3588 | n/a | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0031 | 0.0032 | 0.0030 | 0.0038 | 0.0003 | 0.7174 | n/a | 5.3588 | n/a | NumPy baseline |
| activations | relu | `(4096,)` | PyTorch CPU | 0.0232 | 0.0235 | 0.0226 | 0.0246 | 0.0007 | 0.7174 | n/a | 5.3588 | n/a | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0321 | 0.0321 | 0.0319 | 0.0322 | 0.0001 | 0.3263 | n/a | 2.7974 | n/a | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0105 | 0.0107 | 0.0102 | 0.0113 | 0.0004 | 0.3263 | n/a | 2.7974 | n/a | NumPy baseline |
| activations | sigmoid | `(4096,)` | PyTorch CPU | 0.0897 | 0.0950 | 0.0830 | 0.1106 | 0.0105 | 0.3263 | n/a | 2.7974 | n/a | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0925 | 0.0931 | 0.0630 | 0.1170 | 0.0187 | 0.1160 | n/a | 0.5596 | n/a | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0107 | 0.0109 | 0.0104 | 0.0118 | 0.0005 | 0.1160 | n/a | 0.5596 | n/a | NumPy baseline |
| activations | tanh | `(4096,)` | PyTorch CPU | 0.0518 | 0.0491 | 0.0355 | 0.0595 | 0.0092 | 0.1160 | n/a | 0.5596 | n/a | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0614 | 0.0569 | 0.0445 | 0.0688 | 0.0097 | 0.1548 | n/a | 0.8095 | n/a | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0095 | 0.0100 | 0.0070 | 0.0135 | 0.0028 | 0.1548 | n/a | 0.8095 | n/a | NumPy baseline |
| activations | exp | `(4096,)` | PyTorch CPU | 0.0497 | 0.0504 | 0.0431 | 0.0575 | 0.0060 | 0.1548 | n/a | 0.8095 | n/a | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0379 | 0.0380 | 0.0371 | 0.0391 | 0.0007 | 0.4065 | n/a | 1.2688 | n/a | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0154 | 0.0152 | 0.0143 | 0.0157 | 0.0005 | 0.4065 | n/a | 1.2688 | n/a | NumPy baseline |
| activations | log | `(4096,)` | PyTorch CPU | 0.0481 | 0.0466 | 0.0332 | 0.0593 | 0.0090 | 0.4065 | n/a | 1.2688 | n/a | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0214 | 0.0221 | 0.0168 | 0.0287 | 0.0042 | 0.4236 | n/a | 2.3228 | n/a | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0091 | 0.0091 | 0.0085 | 0.0097 | 0.0004 | 0.4236 | n/a | 2.3228 | n/a | NumPy baseline |
| activations | relu | `(16384,)` | PyTorch CPU | 0.0496 | 0.0467 | 0.0349 | 0.0507 | 0.0060 | 0.4236 | n/a | 2.3228 | n/a | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.2325 | 0.2339 | 0.1665 | 0.3050 | 0.0450 | 0.2689 | n/a | 0.5659 | n/a | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0625 | 0.0635 | 0.0599 | 0.0680 | 0.0027 | 0.2689 | n/a | 0.5659 | n/a | NumPy baseline |
| activations | sigmoid | `(16384,)` | PyTorch CPU | 0.1316 | 0.1264 | 0.0990 | 0.1369 | 0.0139 | 0.2689 | n/a | 0.5659 | n/a | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3265 | 0.3503 | 0.3219 | 0.4420 | 0.0462 | 0.2000 | n/a | 0.1662 | n/a | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0653 | 0.0630 | 0.0400 | 0.0854 | 0.0193 | 0.2000 | n/a | 0.1662 | n/a | NumPy baseline |
| activations | tanh | `(16384,)` | PyTorch CPU | 0.0543 | 0.0521 | 0.0368 | 0.0610 | 0.0084 | 0.2000 | n/a | 0.1662 | n/a | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.2154 | 0.2147 | 0.1464 | 0.2608 | 0.0409 | 0.1115 | n/a | 0.2260 | n/a | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0240 | 0.0256 | 0.0202 | 0.0321 | 0.0052 | 0.1115 | n/a | 0.2260 | n/a | NumPy baseline |
| activations | exp | `(16384,)` | PyTorch CPU | 0.0487 | 0.0502 | 0.0476 | 0.0543 | 0.0027 | 0.1115 | n/a | 0.2260 | n/a | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1690 | 0.1739 | 0.1299 | 0.2425 | 0.0388 | 0.2576 | n/a | 0.3083 | n/a | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0436 | 0.0425 | 0.0284 | 0.0518 | 0.0081 | 0.2576 | n/a | 0.3083 | n/a | NumPy baseline |
| activations | log | `(16384,)` | PyTorch CPU | 0.0521 | 0.0515 | 0.0389 | 0.0601 | 0.0071 | 0.2576 | n/a | 0.3083 | n/a | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0024 | 0.0023 | 0.0017 | 0.0030 | 0.0004 | 0.8597 | n/a | 3.8522 | n/a | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0020 | 0.0021 | 0.0018 | 0.0026 | 0.0003 | 0.8597 | n/a | 3.8522 | n/a | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0091 | 0.0085 | 0.0055 | 0.0126 | 0.0025 | 0.8597 | n/a | 3.8522 | n/a | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0023 | 0.0025 | 0.0021 | 0.0037 | 0.0006 | 3.4575 | n/a | 4.8701 | n/a | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0079 | 0.0092 | 0.0074 | 0.0120 | 0.0019 | 3.4575 | n/a | 4.8701 | n/a | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0112 | 0.0112 | 0.0107 | 0.0116 | 0.0003 | 3.4575 | n/a | 4.8701 | n/a | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 1.1904 | n/a | 3.3904 | n/a | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0019 | 0.0001 | 1.1904 | n/a | 3.3904 | n/a | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0048 | 0.0048 | 0.0045 | 0.0054 | 0.0003 | 1.1904 | n/a | 3.3904 | n/a | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 5.0002 | n/a | 7.6832 | n/a | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0071 | 0.0082 | 0.0070 | 0.0128 | 0.0023 | 5.0002 | n/a | 7.6832 | n/a | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0110 | 0.0109 | 0.0108 | 0.0111 | 0.0001 | 5.0002 | n/a | 7.6832 | n/a | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 1.1023 | n/a | 5.6143 | n/a | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0002 | 1.1023 | n/a | 5.6143 | n/a | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0089 | 0.0080 | 0.0045 | 0.0111 | 0.0023 | 1.1023 | n/a | 5.6143 | n/a | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0016 | 0.0020 | 0.0016 | 0.0035 | 0.0007 | 4.3608 | n/a | 6.7855 | n/a | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0071 | 0.0072 | 0.0070 | 0.0076 | 0.0002 | 4.3608 | n/a | 6.7855 | n/a | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0110 | 0.0119 | 0.0108 | 0.0152 | 0.0017 | 4.3608 | n/a | 6.7855 | n/a | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0021 | 0.0000 | 0.8096 | n/a | 2.1941 | n/a | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 0.8096 | n/a | 2.1941 | n/a | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0046 | 0.0045 | 0.0044 | 0.0046 | 0.0001 | 0.8096 | n/a | 2.1941 | n/a | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0023 | 0.0001 | 3.4400 | n/a | 5.1524 | n/a | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0072 | 0.0076 | 0.0071 | 0.0095 | 0.0009 | 3.4400 | n/a | 5.1524 | n/a | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0108 | 0.0127 | 0.0107 | 0.0202 | 0.0038 | 3.4400 | n/a | 5.1524 | n/a | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0073 | 0.0072 | 0.0069 | 0.0074 | 0.0002 | 0.2797 | n/a | 0.6940 | n/a | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0000 | 0.2797 | n/a | 0.6940 | n/a | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0051 | 0.0051 | 0.0046 | 0.0056 | 0.0004 | 0.2797 | n/a | 0.6940 | n/a | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0071 | 0.0073 | 0.0070 | 0.0077 | 0.0003 | 1.0679 | n/a | 1.7859 | n/a | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0076 | 0.0084 | 0.0073 | 0.0123 | 0.0019 | 1.0679 | n/a | 1.7859 | n/a | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0127 | 0.0123 | 0.0106 | 0.0138 | 0.0013 | 1.0679 | n/a | 1.7859 | n/a | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0246 | 0.0265 | 0.0242 | 0.0347 | 0.0041 | 0.1255 | n/a | 0.2101 | n/a | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0031 | 0.0031 | 0.0028 | 0.0034 | 0.0002 | 0.1255 | n/a | 0.2101 | n/a | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0052 | 0.0054 | 0.0051 | 0.0063 | 0.0005 | 0.1255 | n/a | 0.2101 | n/a | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0245 | 0.0245 | 0.0242 | 0.0247 | 0.0002 | 0.3568 | n/a | 0.4702 | n/a | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0087 | 0.0087 | 0.0085 | 0.0090 | 0.0002 | 0.3568 | n/a | 0.4702 | n/a | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0115 | 0.0120 | 0.0111 | 0.0132 | 0.0008 | 0.3568 | n/a | 0.4702 | n/a | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0966 | 0.1084 | 0.0928 | 0.1333 | 0.0165 | 0.0613 | n/a | 0.0710 | n/a | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0059 | 0.0059 | 0.0059 | 0.0060 | 0.0000 | 0.0613 | n/a | 0.0710 | n/a | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0069 | 0.0069 | 0.0068 | 0.0071 | 0.0001 | 0.0613 | n/a | 0.0710 | n/a | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1283 | 0.1347 | 0.1109 | 0.1663 | 0.0194 | 0.1118 | n/a | 0.1185 | n/a | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0143 | 0.0146 | 0.0128 | 0.0159 | 0.0012 | 0.1118 | n/a | 0.1185 | n/a | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0152 | 0.0152 | 0.0139 | 0.0176 | 0.0013 | 0.1118 | n/a | 0.1185 | n/a | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0031 | 0.0033 | 0.0030 | 0.0038 | 0.0004 | 0.6256 | n/a | 1.6938 | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0021 | 0.0019 | 0.0025 | 0.0002 | 0.6256 | n/a | 1.6938 | n/a | NumPy baseline |
| matmul | matmul | `(16, 16)` | PyTorch CPU | 0.0053 | 0.0049 | 0.0040 | 0.0054 | 0.0005 | 0.6256 | n/a | 1.6938 | n/a | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0412 | 0.0389 | 0.0343 | 0.0430 | 0.0036 | 0.2514 | n/a | 0.3274 | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0104 | 0.0104 | 0.0103 | 0.0106 | 0.0001 | 0.2514 | n/a | 0.3274 | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | PyTorch CPU | 0.0135 | 0.0134 | 0.0127 | 0.0137 | 0.0004 | 0.2514 | n/a | 0.3274 | n/a | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.5742 | 0.5507 | 0.4636 | 0.5846 | 0.0449 | 0.2842 | n/a | 0.0565 | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1632 | 0.1645 | 0.1374 | 0.1877 | 0.0176 | 0.2842 | n/a | 0.0565 | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | PyTorch CPU | 0.0324 | 0.0310 | 0.0256 | 0.0349 | 0.0032 | 0.2842 | n/a | 0.0565 | n/a | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 4.3941 | 4.3956 | 4.2338 | 4.5186 | 0.1098 | 0.0937 | n/a | 0.0319 | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4116 | 0.4122 | 0.3108 | 0.5232 | 0.0712 | 0.0937 | n/a | 0.0319 | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | PyTorch CPU | 0.1402 | 0.1685 | 0.1380 | 0.2263 | 0.0372 | 0.0937 | n/a | 0.0319 | n/a | reference |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0153 | 0.0171 | 0.0146 | 0.0245 | 0.0037 | n/a | n/a | 10.0788 | n/a | no NumPy reference |
| autograd | scalar_backward | `(1,)` | PyTorch CPU | 0.1538 | 0.1514 | 0.1116 | 0.1730 | 0.0223 | n/a | n/a | 10.0788 | n/a | reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0322 | 0.0374 | 0.0316 | 0.0457 | 0.0067 | n/a | n/a | 7.6228 | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | PyTorch CPU | 0.2454 | 0.2430 | 0.1898 | 0.3052 | 0.0386 | n/a | n/a | 7.6228 | n/a | reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1397 | 0.1417 | 0.1369 | 0.1522 | 0.0056 | n/a | n/a | 0.6880 | n/a | no NumPy reference |
| autograd | scalar_backward | `(128,)` | PyTorch CPU | 0.0961 | 0.1238 | 0.0936 | 0.1800 | 0.0364 | n/a | n/a | 0.6880 | n/a | reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3325 | 0.3543 | 0.3181 | 0.4519 | 0.0501 | n/a | n/a | 0.5821 | n/a | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | PyTorch CPU | 0.1936 | 0.1980 | 0.1755 | 0.2311 | 0.0195 | n/a | n/a | 0.5821 | n/a | reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.0033 | 1.0028 | 0.9977 | 1.0056 | 0.0027 | n/a | n/a | 0.0971 | n/a | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | PyTorch CPU | 0.0974 | 0.0993 | 0.0964 | 0.1032 | 0.0030 | n/a | n/a | 0.0971 | n/a | reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.3844 | 2.5269 | 2.3738 | 3.0891 | 0.2813 | n/a | n/a | 0.0767 | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | PyTorch CPU | 0.1830 | 0.1832 | 0.1791 | 0.1883 | 0.0030 | n/a | n/a | 0.0767 | n/a | reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.5469 | 1.5484 | 1.5298 | 1.5744 | 0.0147 | 0.0860 | n/a | n/a | n/a | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1331 | 0.1345 | 0.1328 | 0.1404 | 0.0029 | 0.0860 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
