# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.1.0`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: available (2.12.1+cpu)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `23`
- TensorStudio losses versus NumPy: `80`
- TensorStudio wins versus PyTorch CPU: `74`
- TensorStudio losses versus PyTorch CPU: `34`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0027 | 0.0033 | 0.0027 | 0.0055 | 0.0011 | 0.2854 | n/a | 1.4350 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2854 | n/a | 1.4350 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | PyTorch CPU | 0.0039 | 0.0041 | 0.0033 | 0.0055 | 0.0008 | 0.2854 | n/a | 1.4350 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3031 | n/a | 1.6071 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3031 | n/a | 1.6071 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | PyTorch CPU | 0.0037 | 0.0037 | 0.0035 | 0.0040 | 0.0002 | 0.3031 | n/a | 1.6071 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0023 | 0.0024 | 0.0022 | 0.0029 | 0.0003 | 0.3220 | n/a | 1.4474 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3220 | n/a | 1.4474 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | PyTorch CPU | 0.0033 | 0.0033 | 0.0032 | 0.0036 | 0.0001 | 0.3220 | n/a | 1.4474 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3273 | n/a | 1.5019 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.3273 | n/a | 1.5019 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | PyTorch CPU | 0.0034 | 0.0035 | 0.0034 | 0.0037 | 0.0001 | 0.3273 | n/a | 1.5019 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0087 | 0.0087 | 0.0085 | 0.0090 | 0.0002 | 0.4425 | n/a | 6.4211 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0038 | 0.0039 | 0.0038 | 0.0040 | 0.0001 | 0.4425 | n/a | 6.4211 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | PyTorch CPU | 0.0556 | 0.0558 | 0.0552 | 0.0571 | 0.0007 | 0.4425 | n/a | 6.4211 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.2920 | n/a | 1.5123 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2920 | n/a | 1.5123 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0040 | 0.0002 | 0.2920 | n/a | 1.5123 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.3246 | n/a | 1.6023 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3246 | n/a | 1.6023 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0034 | 0.0038 | 0.0001 | 0.3246 | n/a | 1.6023 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.3226 | n/a | 1.4962 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3226 | n/a | 1.4962 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | PyTorch CPU | 0.0033 | 0.0033 | 0.0033 | 0.0034 | 0.0000 | 0.3226 | n/a | 1.4962 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3189 | n/a | 1.5628 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3189 | n/a | 1.5628 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0041 | 0.0003 | 0.3189 | n/a | 1.5628 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0089 | 0.0093 | 0.0088 | 0.0109 | 0.0008 | 0.4403 | n/a | 6.2911 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0039 | 0.0041 | 0.0038 | 0.0049 | 0.0004 | 0.4403 | n/a | 6.2911 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | PyTorch CPU | 0.0560 | 0.0561 | 0.0556 | 0.0568 | 0.0005 | 0.4403 | n/a | 6.2911 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0000 | 0.2933 | n/a | 1.4956 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2933 | n/a | 1.4956 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 0.2933 | n/a | 1.4956 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.3038 | n/a | 1.5596 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3038 | n/a | 1.5596 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0034 | 0.0038 | 0.0001 | 0.3038 | n/a | 1.5596 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3215 | n/a | 1.4590 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.3215 | n/a | 1.4590 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | PyTorch CPU | 0.0032 | 0.0033 | 0.0032 | 0.0037 | 0.0002 | 0.3215 | n/a | 1.4590 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3174 | n/a | 1.4424 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3174 | n/a | 1.4424 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | PyTorch CPU | 0.0033 | 0.0033 | 0.0033 | 0.0034 | 0.0001 | 0.3174 | n/a | 1.4424 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0088 | 0.0088 | 0.0087 | 0.0089 | 0.0001 | 0.4584 | n/a | 6.4373 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0040 | 0.0040 | 0.0038 | 0.0042 | 0.0001 | 0.4584 | n/a | 6.4373 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | PyTorch CPU | 0.0566 | 0.0633 | 0.0564 | 0.0794 | 0.0091 | 0.4584 | n/a | 6.4373 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.2875 | n/a | 1.4667 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2875 | n/a | 1.4667 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0034 | 0.0039 | 0.0002 | 0.2875 | n/a | 1.4667 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0025 | 0.0001 | 0.3232 | n/a | 1.5314 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3232 | n/a | 1.5314 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0035 | 0.0038 | 0.0001 | 0.3232 | n/a | 1.5314 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.3123 | n/a | 1.4178 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3123 | n/a | 1.4178 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | PyTorch CPU | 0.0033 | 0.0033 | 0.0032 | 0.0033 | 0.0000 | 0.3123 | n/a | 1.4178 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.3110 | n/a | 1.4256 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3110 | n/a | 1.4256 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | PyTorch CPU | 0.0034 | 0.0033 | 0.0032 | 0.0034 | 0.0001 | 0.3110 | n/a | 1.4256 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0091 | 0.0091 | 0.0090 | 0.0092 | 0.0001 | 0.4301 | n/a | 6.1693 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0041 | 0.0001 | 0.4301 | n/a | 6.1693 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | PyTorch CPU | 0.0560 | 0.0567 | 0.0556 | 0.0589 | 0.0012 | 0.4301 | n/a | 6.1693 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0026 | 0.0033 | 0.0002 | 0.3637 | n/a | 1.2992 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0011 | 0.0010 | 0.0009 | 0.0011 | 0.0001 | 0.3637 | n/a | 1.2992 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | PyTorch CPU | 0.0039 | 0.0039 | 0.0035 | 0.0048 | 0.0004 | 0.3637 | n/a | 1.2992 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0026 | 0.0026 | 0.0026 | 0.0026 | 0.0000 | 0.3536 | n/a | 1.4318 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0009 | 0.0009 | 0.0009 | 0.0009 | 0.0000 | 0.3536 | n/a | 1.4318 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | PyTorch CPU | 0.0037 | 0.0043 | 0.0037 | 0.0064 | 0.0011 | 0.3536 | n/a | 1.4318 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0027 | 0.0000 | 0.4063 | n/a | 1.3075 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0012 | 0.0001 | 0.4063 | n/a | 1.3075 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0035 | 0.0035 | 0.0000 | 0.4063 | n/a | 1.3075 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0029 | 0.0031 | 0.0029 | 0.0038 | 0.0004 | 0.3530 | n/a | 1.2123 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.3530 | n/a | 1.2123 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | PyTorch CPU | 0.0036 | 0.0038 | 0.0035 | 0.0051 | 0.0006 | 0.3530 | n/a | 1.2123 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0110 | 0.0111 | 0.0110 | 0.0113 | 0.0001 | 0.4874 | n/a | 5.3448 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0054 | 0.0053 | 0.0052 | 0.0054 | 0.0001 | 0.4874 | n/a | 5.3448 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | PyTorch CPU | 0.0588 | 0.0602 | 0.0585 | 0.0658 | 0.0028 | 0.4874 | n/a | 5.3448 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0047 | 0.0048 | 0.0046 | 0.0050 | 0.0002 | 0.3161 | n/a | 0.9595 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | 0.3161 | n/a | 0.9595 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | PyTorch CPU | 0.0045 | 0.0047 | 0.0043 | 0.0053 | 0.0004 | 0.3161 | n/a | 0.9595 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0046 | 0.0047 | 0.0045 | 0.0051 | 0.0002 | 0.3009 | n/a | 1.0069 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0016 | 0.0001 | 0.3009 | n/a | 1.0069 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | PyTorch CPU | 0.0046 | 0.0047 | 0.0046 | 0.0048 | 0.0001 | 0.3009 | n/a | 1.0069 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0046 | 0.0045 | 0.0045 | 0.0046 | 0.0001 | 0.3123 | n/a | 0.9534 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.3123 | n/a | 0.9534 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | PyTorch CPU | 0.0044 | 0.0043 | 0.0042 | 0.0045 | 0.0001 | 0.3123 | n/a | 0.9534 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0050 | 0.0051 | 0.0050 | 0.0052 | 0.0001 | 0.3152 | n/a | 0.8980 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0001 | 0.3152 | n/a | 0.8980 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | PyTorch CPU | 0.0045 | 0.0047 | 0.0042 | 0.0056 | 0.0005 | 0.3152 | n/a | 0.8980 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0188 | 0.0189 | 0.0187 | 0.0192 | 0.0002 | 0.4136 | n/a | 3.2161 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0078 | 0.0078 | 0.0076 | 0.0079 | 0.0001 | 0.4136 | n/a | 3.2161 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | PyTorch CPU | 0.0603 | 0.0604 | 0.0595 | 0.0615 | 0.0008 | 0.4136 | n/a | 3.2161 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0108 | 0.0110 | 0.0108 | 0.0114 | 0.0002 | 0.3207 | n/a | 0.6651 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0035 | 0.0035 | 0.0035 | 0.0035 | 0.0000 | 0.3207 | n/a | 0.6651 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | PyTorch CPU | 0.0072 | 0.0073 | 0.0072 | 0.0075 | 0.0001 | 0.3207 | n/a | 0.6651 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0102 | 0.0104 | 0.0102 | 0.0108 | 0.0003 | 0.3614 | n/a | 0.7116 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0037 | 0.0000 | 0.3614 | n/a | 0.7116 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | PyTorch CPU | 0.0072 | 0.0074 | 0.0072 | 0.0080 | 0.0003 | 0.3614 | n/a | 0.7116 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0094 | 0.0094 | 0.0094 | 0.0094 | 0.0000 | 0.3955 | n/a | 0.7630 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0038 | 0.0037 | 0.0039 | 0.0001 | 0.3955 | n/a | 0.7630 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | PyTorch CPU | 0.0072 | 0.0072 | 0.0071 | 0.0073 | 0.0001 | 0.3955 | n/a | 0.7630 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0126 | 0.0130 | 0.0125 | 0.0143 | 0.0007 | 0.3306 | n/a | 0.5730 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0041 | 0.0042 | 0.0041 | 0.0044 | 0.0001 | 0.3306 | n/a | 0.5730 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | PyTorch CPU | 0.0072 | 0.0072 | 0.0072 | 0.0073 | 0.0001 | 0.3306 | n/a | 0.5730 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0439 | 0.0442 | 0.0435 | 0.0458 | 0.0008 | 0.4246 | n/a | 1.5787 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0186 | 0.0188 | 0.0186 | 0.0192 | 0.0002 | 0.4246 | n/a | 1.5787 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | PyTorch CPU | 0.0692 | 0.0710 | 0.0682 | 0.0796 | 0.0043 | 0.4246 | n/a | 1.5787 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0017 | 0.0001 | 0.8886 | n/a | 14.1501 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.8886 | n/a | 14.1501 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1,)` | PyTorch CPU | 0.0215 | 0.0216 | 0.0213 | 0.0221 | 0.0003 | 0.8886 | n/a | 14.1501 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 2.3637 | n/a | 37.4849 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0037 | 0.0001 | 2.3637 | n/a | 37.4849 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | PyTorch CPU | 0.0579 | 0.0589 | 0.0574 | 0.0640 | 0.0025 | 2.3637 | n/a | 37.4849 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.7942 | n/a | 13.7324 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7942 | n/a | 13.7324 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | PyTorch CPU | 0.0215 | 0.0216 | 0.0213 | 0.0223 | 0.0003 | 0.7942 | n/a | 13.7324 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.7668 | n/a | 13.6638 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7668 | n/a | 13.6638 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1,)` | PyTorch CPU | 0.0216 | 0.0217 | 0.0215 | 0.0221 | 0.0002 | 0.7668 | n/a | 13.6638 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.7677 | n/a | 13.2992 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.7677 | n/a | 13.2992 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1,)` | PyTorch CPU | 0.0215 | 0.0216 | 0.0215 | 0.0219 | 0.0002 | 0.7677 | n/a | 13.2992 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.9228 | n/a | 14.5070 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0017 | 0.0001 | 0.9228 | n/a | 14.5070 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(8,)` | PyTorch CPU | 0.0219 | 0.0219 | 0.0216 | 0.0221 | 0.0002 | 0.9228 | n/a | 14.5070 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 2.2843 | n/a | 36.4423 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0037 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 2.2843 | n/a | 36.4423 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | PyTorch CPU | 0.0588 | 0.0588 | 0.0580 | 0.0600 | 0.0007 | 2.2843 | n/a | 36.4423 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.7563 | n/a | 13.5333 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7563 | n/a | 13.5333 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | PyTorch CPU | 0.0219 | 0.0219 | 0.0217 | 0.0221 | 0.0001 | 0.7563 | n/a | 13.5333 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.7228 | n/a | 13.3430 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.7228 | n/a | 13.3430 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(8,)` | PyTorch CPU | 0.0220 | 0.0219 | 0.0218 | 0.0221 | 0.0001 | 0.7228 | n/a | 13.3430 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.7441 | n/a | 13.3707 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7441 | n/a | 13.3707 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(8,)` | PyTorch CPU | 0.0220 | 0.0220 | 0.0218 | 0.0221 | 0.0001 | 0.7441 | n/a | 13.3707 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0015 | 0.0017 | 0.0015 | 0.0022 | 0.0002 | 0.8959 | n/a | 13.8052 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.8959 | n/a | 13.8052 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(32,)` | PyTorch CPU | 0.0214 | 0.0215 | 0.0212 | 0.0219 | 0.0002 | 0.8959 | n/a | 13.8052 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0017 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 2.1053 | n/a | 33.3251 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0037 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 2.1053 | n/a | 33.3251 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | PyTorch CPU | 0.0578 | 0.0579 | 0.0575 | 0.0583 | 0.0003 | 2.1053 | n/a | 33.3251 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0019 | 0.0019 | 0.0019 | 0.0020 | 0.0000 | 0.6577 | n/a | 11.2555 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.6577 | n/a | 11.2555 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | PyTorch CPU | 0.0216 | 0.0216 | 0.0215 | 0.0217 | 0.0001 | 0.6577 | n/a | 11.2555 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0017 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 0.7028 | n/a | 12.6777 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0012 | 0.0014 | 0.0012 | 0.0019 | 0.0002 | 0.7028 | n/a | 12.6777 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(32,)` | PyTorch CPU | 0.0221 | 0.0222 | 0.0217 | 0.0228 | 0.0004 | 0.7028 | n/a | 12.6777 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0020 | 0.0001 | 0.6952 | n/a | 12.0849 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.6952 | n/a | 12.0849 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(32,)` | PyTorch CPU | 0.0218 | 0.0219 | 0.0215 | 0.0227 | 0.0004 | 0.6952 | n/a | 12.0849 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.8778 | n/a | 13.2308 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.8778 | n/a | 13.2308 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(128,)` | PyTorch CPU | 0.0216 | 0.0216 | 0.0215 | 0.0217 | 0.0001 | 0.8778 | n/a | 13.2308 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0025 | 0.0000 | 1.5201 | n/a | 23.9182 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0038 | 0.0037 | 0.0037 | 0.0038 | 0.0001 | 1.5201 | n/a | 23.9182 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | PyTorch CPU | 0.0592 | 0.0614 | 0.0591 | 0.0704 | 0.0045 | 1.5201 | n/a | 23.9182 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0027 | 0.0028 | 0.0000 | 0.5428 | n/a | 8.1728 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.5428 | n/a | 8.1728 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | PyTorch CPU | 0.0223 | 0.0226 | 0.0222 | 0.0240 | 0.0007 | 0.5428 | n/a | 8.1728 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0000 | 0.5456 | n/a | 9.2138 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.5456 | n/a | 9.2138 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(128,)` | PyTorch CPU | 0.0224 | 0.0223 | 0.0221 | 0.0225 | 0.0001 | 0.5456 | n/a | 9.2138 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0000 | 0.5849 | n/a | 9.1438 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.5849 | n/a | 9.1438 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(128,)` | PyTorch CPU | 0.0223 | 0.0222 | 0.0219 | 0.0224 | 0.0002 | 0.5849 | n/a | 9.1438 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.8582 | n/a | 9.8143 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 0.8582 | n/a | 9.8143 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0218 | 0.0219 | 0.0001 | 0.8582 | n/a | 9.8143 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0091 | 0.0091 | 0.0089 | 0.0092 | 0.0001 | 0.6450 | n/a | 6.6276 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0059 | 0.0059 | 0.0057 | 0.0064 | 0.0003 | 0.6450 | n/a | 6.6276 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | PyTorch CPU | 0.0603 | 0.0603 | 0.0602 | 0.0605 | 0.0001 | 0.6450 | n/a | 6.6276 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0132 | 0.0133 | 0.0132 | 0.0135 | 0.0001 | 0.2781 | n/a | 1.8009 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0038 | 0.0001 | 0.2781 | n/a | 1.8009 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | PyTorch CPU | 0.0238 | 0.0239 | 0.0237 | 0.0241 | 0.0001 | 0.2781 | n/a | 1.8009 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0081 | 0.0081 | 0.0079 | 0.0084 | 0.0002 | 0.3199 | n/a | 2.8567 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0001 | 0.3199 | n/a | 2.8567 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | PyTorch CPU | 0.0230 | 0.0234 | 0.0226 | 0.0247 | 0.0008 | 0.3199 | n/a | 2.8567 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0085 | 0.0085 | 0.0084 | 0.0087 | 0.0001 | 0.3403 | n/a | 2.7102 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0029 | 0.0030 | 0.0000 | 0.3403 | n/a | 2.7102 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1024,)` | PyTorch CPU | 0.0230 | 0.0232 | 0.0224 | 0.0246 | 0.0008 | 0.3403 | n/a | 2.7102 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0045 | 0.0045 | 0.0045 | 0.0047 | 0.0001 | 0.7038 | n/a | 5.0845 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0031 | 0.0033 | 0.0030 | 0.0040 | 0.0004 | 0.7038 | n/a | 5.0845 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | PyTorch CPU | 0.0228 | 0.0275 | 0.0223 | 0.0442 | 0.0084 | 0.7038 | n/a | 5.0845 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0316 | 0.0316 | 0.0315 | 0.0319 | 0.0002 | 0.3404 | n/a | 3.4546 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0108 | 0.0108 | 0.0106 | 0.0112 | 0.0002 | 0.3404 | n/a | 3.4546 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | PyTorch CPU | 0.1093 | 0.1053 | 0.0928 | 0.1154 | 0.0091 | 0.3404 | n/a | 3.4546 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0703 | 0.0788 | 0.0644 | 0.1004 | 0.0149 | 0.1516 | n/a | 0.5825 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0107 | 0.0113 | 0.0105 | 0.0140 | 0.0014 | 0.1516 | n/a | 0.5825 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | PyTorch CPU | 0.0410 | 0.0438 | 0.0403 | 0.0488 | 0.0039 | 0.1516 | n/a | 0.5825 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0483 | 0.0453 | 0.0320 | 0.0623 | 0.0112 | 0.2530 | n/a | 0.7279 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0122 | 0.0122 | 0.0114 | 0.0129 | 0.0005 | 0.2530 | n/a | 0.7279 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | PyTorch CPU | 0.0352 | 0.0343 | 0.0264 | 0.0439 | 0.0061 | 0.2530 | n/a | 0.7279 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0330 | 0.0345 | 0.0311 | 0.0414 | 0.0037 | 0.2534 | n/a | 1.1805 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0084 | 0.0089 | 0.0074 | 0.0113 | 0.0015 | 0.2534 | n/a | 1.1805 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(4096,)` | PyTorch CPU | 0.0389 | 0.0390 | 0.0301 | 0.0454 | 0.0056 | 0.2534 | n/a | 1.1805 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0222 | 0.0224 | 0.0217 | 0.0232 | 0.0007 | 0.9858 | n/a | 1.4715 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0219 | 0.0209 | 0.0158 | 0.0232 | 0.0027 | 0.9858 | n/a | 1.4715 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | PyTorch CPU | 0.0326 | 0.0338 | 0.0288 | 0.0424 | 0.0046 | 0.9858 | n/a | 1.4715 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.2833 | 0.2495 | 0.1226 | 0.2958 | 0.0646 | 0.2064 | n/a | 0.4675 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0585 | 0.0589 | 0.0558 | 0.0645 | 0.0030 | 0.2064 | n/a | 0.4675 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | PyTorch CPU | 0.1324 | 0.1225 | 0.0949 | 0.1380 | 0.0161 | 0.2064 | n/a | 0.4675 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3321 | 0.3182 | 0.2718 | 0.3458 | 0.0278 | 0.1362 | n/a | 0.1689 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0452 | 0.0534 | 0.0429 | 0.0749 | 0.0126 | 0.1362 | n/a | 0.1689 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | PyTorch CPU | 0.0561 | 0.0549 | 0.0380 | 0.0725 | 0.0111 | 0.1362 | n/a | 0.1689 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1194 | 0.1360 | 0.1168 | 0.1841 | 0.0257 | 0.2000 | n/a | 0.2547 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0239 | 0.0245 | 0.0205 | 0.0330 | 0.0045 | 0.2000 | n/a | 0.2547 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | PyTorch CPU | 0.0304 | 0.0327 | 0.0289 | 0.0444 | 0.0059 | 0.2000 | n/a | 0.2547 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1224 | 0.1302 | 0.1170 | 0.1673 | 0.0186 | 0.2285 | n/a | 0.2760 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0280 | 0.0293 | 0.0259 | 0.0335 | 0.0031 | 0.2285 | n/a | 0.2760 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(16384,)` | PyTorch CPU | 0.0338 | 0.0327 | 0.0275 | 0.0358 | 0.0029 | 0.2285 | n/a | 0.2760 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0019 | 0.0020 | 0.0018 | 0.0023 | 0.0002 | 1.9940 | n/a | 3.3958 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0038 | 0.0033 | 0.0022 | 0.0044 | 0.0009 | 1.9940 | n/a | 3.3958 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0064 | 0.0064 | 0.0058 | 0.0071 | 0.0005 | 1.9940 | n/a | 3.3958 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0019 | 0.0033 | 0.0005 | 3.9130 | n/a | 5.9338 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0084 | 0.0104 | 0.0078 | 0.0156 | 0.0030 | 3.9130 | n/a | 5.9338 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0128 | 0.0130 | 0.0124 | 0.0141 | 0.0006 | 3.9130 | n/a | 5.9338 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 1.2346 | n/a | 2.9494 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0000 | 1.2346 | n/a | 2.9494 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0052 | 0.0053 | 0.0051 | 0.0057 | 0.0002 | 1.2346 | n/a | 2.9494 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0022 | 0.0002 | 4.4939 | n/a | 7.8667 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0079 | 0.0079 | 0.0078 | 0.0081 | 0.0001 | 4.4939 | n/a | 7.8667 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0139 | 0.0137 | 0.0127 | 0.0150 | 0.0008 | 4.4939 | n/a | 7.8667 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0020 | 0.0020 | 0.0019 | 0.0022 | 0.0001 | 1.0562 | n/a | 2.5412 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0021 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 1.0562 | n/a | 2.5412 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0052 | 0.0052 | 0.0051 | 0.0054 | 0.0001 | 1.0562 | n/a | 2.5412 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 4.1340 | n/a | 6.7869 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0078 | 0.0078 | 0.0077 | 0.0080 | 0.0001 | 4.1340 | n/a | 6.7869 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0128 | 0.0128 | 0.0126 | 0.0131 | 0.0002 | 4.1340 | n/a | 6.7869 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0025 | 0.0000 | 0.8818 | n/a | 2.0988 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0022 | 0.0000 | 0.8818 | n/a | 2.0988 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0052 | 0.0052 | 0.0051 | 0.0053 | 0.0001 | 0.8818 | n/a | 2.0988 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0027 | 0.0001 | 3.2272 | n/a | 5.1403 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0079 | 0.0080 | 0.0079 | 0.0081 | 0.0001 | 3.2272 | n/a | 5.1403 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0127 | 0.0127 | 0.0125 | 0.0129 | 0.0001 | 3.2272 | n/a | 5.1403 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0074 | 0.0076 | 0.0074 | 0.0081 | 0.0003 | 0.3537 | n/a | 0.7307 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0026 | 0.0028 | 0.0025 | 0.0036 | 0.0004 | 0.3537 | n/a | 0.7307 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0054 | 0.0053 | 0.0052 | 0.0055 | 0.0001 | 0.3537 | n/a | 0.7307 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0074 | 0.0075 | 0.0074 | 0.0078 | 0.0001 | 1.1169 | n/a | 1.8484 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0083 | 0.0084 | 0.0083 | 0.0087 | 0.0001 | 1.1169 | n/a | 1.8484 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0138 | 0.0139 | 0.0125 | 0.0167 | 0.0015 | 1.1169 | n/a | 1.8484 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0248 | 0.0248 | 0.0247 | 0.0249 | 0.0001 | 0.1364 | n/a | 0.2373 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0034 | 0.0035 | 0.0034 | 0.0037 | 0.0001 | 0.1364 | n/a | 0.2373 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0059 | 0.0063 | 0.0059 | 0.0074 | 0.0006 | 0.1364 | n/a | 0.2373 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0250 | 0.0249 | 0.0245 | 0.0250 | 0.0002 | 0.3899 | n/a | 0.5792 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0097 | 0.0098 | 0.0096 | 0.0105 | 0.0003 | 0.3899 | n/a | 0.5792 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0145 | 0.0155 | 0.0139 | 0.0179 | 0.0017 | 0.3899 | n/a | 0.5792 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0985 | 0.0983 | 0.0949 | 0.1025 | 0.0028 | 0.0686 | n/a | 0.0819 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0068 | 0.0073 | 0.0067 | 0.0082 | 0.0007 | 0.0686 | n/a | 0.0819 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0081 | 0.0086 | 0.0078 | 0.0099 | 0.0009 | 0.0686 | n/a | 0.0819 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0949 | 0.0960 | 0.0939 | 0.1003 | 0.0023 | 0.1375 | n/a | 0.1641 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0131 | 0.0133 | 0.0129 | 0.0139 | 0.0004 | 0.1375 | n/a | 0.1641 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0156 | 0.0157 | 0.0155 | 0.0164 | 0.0003 | 0.1375 | n/a | 0.1641 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0021 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 1.1545 | n/a | 3.1404 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0025 | 0.0025 | 0.0025 | 0.0025 | 0.0000 | 1.1545 | n/a | 3.1404 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | PyTorch CPU | 0.0067 | 0.0071 | 0.0067 | 0.0085 | 0.0007 | 1.1545 | n/a | 3.1404 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0045 | 0.0046 | 0.0044 | 0.0047 | 0.0001 | 1.7562 | n/a | 3.2618 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0080 | 0.0080 | 0.0079 | 0.0083 | 0.0002 | 1.7562 | n/a | 3.2618 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | PyTorch CPU | 0.0148 | 0.0149 | 0.0143 | 0.0158 | 0.0005 | 1.7562 | n/a | 3.2618 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0073 | 0.0074 | 0.0072 | 0.0076 | 0.0001 | 0.5476 | n/a | 1.2201 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0040 | 0.0040 | 0.0040 | 0.0042 | 0.0001 | 0.5476 | n/a | 1.2201 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | PyTorch CPU | 0.0089 | 0.0089 | 0.0088 | 0.0090 | 0.0001 | 0.5476 | n/a | 1.2201 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0057 | 0.0056 | 0.0055 | 0.0058 | 0.0001 | 1.7454 | n/a | 2.7956 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0099 | 0.0100 | 0.0098 | 0.0104 | 0.0002 | 1.7454 | n/a | 2.7956 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | PyTorch CPU | 0.0159 | 0.0160 | 0.0152 | 0.0165 | 0.0005 | 1.7454 | n/a | 2.7956 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0179 | 0.0181 | 0.0178 | 0.0185 | 0.0003 | 0.4112 | n/a | 0.7227 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0074 | 0.0075 | 0.0073 | 0.0082 | 0.0004 | 0.4112 | n/a | 0.7227 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | PyTorch CPU | 0.0129 | 0.0130 | 0.0129 | 0.0135 | 0.0003 | 0.4112 | n/a | 0.7227 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0084 | 0.0086 | 0.0084 | 0.0092 | 0.0003 | 1.5700 | n/a | 2.1348 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0132 | 0.0133 | 0.0131 | 0.0137 | 0.0002 | 1.5700 | n/a | 2.1348 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | PyTorch CPU | 0.0180 | 0.0180 | 0.0177 | 0.0183 | 0.0002 | 1.5700 | n/a | 2.1348 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0663 | 0.0665 | 0.0662 | 0.0670 | 0.0003 | 0.3141 | n/a | 0.3489 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0208 | 0.0210 | 0.0208 | 0.0217 | 0.0003 | 0.3141 | n/a | 0.3489 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | PyTorch CPU | 0.0231 | 0.0221 | 0.0168 | 0.0256 | 0.0029 | 0.3141 | n/a | 0.3489 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0360 | 0.0356 | 0.0332 | 0.0384 | 0.0020 | 1.1256 | n/a | 1.0715 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0405 | 0.0401 | 0.0385 | 0.0416 | 0.0014 | 1.1256 | n/a | 1.0715 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | PyTorch CPU | 0.0386 | 0.0392 | 0.0351 | 0.0461 | 0.0039 | 1.1256 | n/a | 1.0715 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0064 | 0.0064 | 0.0060 | 0.0067 | 0.0003 | 0.3031 | n/a | 0.9062 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0023 | 0.0019 | 0.0037 | 0.0007 | 0.3031 | n/a | 0.9062 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | PyTorch CPU | 0.0058 | 0.0070 | 0.0048 | 0.0099 | 0.0022 | 0.3031 | n/a | 0.9062 | n/a | no | n/a | no | n/a | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0416 | 0.0432 | 0.0377 | 0.0480 | 0.0039 | 0.2967 | n/a | 0.1989 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0123 | 0.0124 | 0.0123 | 0.0127 | 0.0002 | 0.2967 | n/a | 0.1989 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(64, 64)` | PyTorch CPU | 0.0083 | 0.0084 | 0.0082 | 0.0087 | 0.0002 | 0.2967 | n/a | 0.1989 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3024 | 0.3116 | 0.2821 | 0.3790 | 0.0347 | 0.5085 | n/a | 0.0842 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1538 | 0.1604 | 0.1317 | 0.1856 | 0.0200 | 0.5085 | n/a | 0.0842 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(128, 128)` | PyTorch CPU | 0.0255 | 0.0262 | 0.0252 | 0.0294 | 0.0016 | 0.5085 | n/a | 0.0842 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 4.0653 | 4.0920 | 3.9528 | 4.2915 | 0.1335 | 0.0977 | n/a | 0.0229 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.3973 | 0.4246 | 0.3314 | 0.5774 | 0.0823 | 0.0977 | n/a | 0.0229 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(256, 256)` | PyTorch CPU | 0.0929 | 0.1095 | 0.0904 | 0.1442 | 0.0225 | 0.0977 | n/a | 0.0229 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2040 | 0.2090 | 0.1656 | 0.2451 | 0.0301 | 6.3118 | n/a | 0.0892 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2879 | 1.4865 | 1.2269 | 2.0426 | 0.3190 | 6.3118 | n/a | 0.0892 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | PyTorch CPU | 0.0182 | 0.0170 | 0.0132 | 0.0199 | 0.0024 | 6.3118 | n/a | 0.0892 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.2872 | 8.3209 | 7.2674 | 10.9909 | 1.4549 | 8.5240 | n/a | 0.0156 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 62.1162 | 62.8240 | 60.0867 | 68.9074 | 3.1963 | 8.5240 | n/a | 0.0156 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | PyTorch CPU | 0.1133 | 0.1221 | 0.0973 | 0.1751 | 0.0273 | 8.5240 | n/a | 0.0156 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0220 | 0.0214 | 0.0141 | 0.0265 | 0.0042 | 12.9704 | n/a | 0.2517 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.2847 | 0.2485 | 0.1592 | 0.3368 | 0.0723 | 12.9704 | n/a | 0.2517 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | PyTorch CPU | 0.0055 | 0.0056 | 0.0055 | 0.0062 | 0.0003 | 12.9704 | n/a | 0.2517 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0144 | 0.0144 | 0.0139 | 0.0153 | 0.0005 | 37.8173 | n/a | 0.4778 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5428 | 0.5486 | 0.5375 | 0.5786 | 0.0154 | 37.8173 | n/a | 0.4778 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | PyTorch CPU | 0.0069 | 0.0067 | 0.0050 | 0.0081 | 0.0010 | 37.8173 | n/a | 0.4778 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.8546 | 0.8058 | 0.6739 | 0.8839 | 0.0840 | 8.9986 | n/a | 0.0430 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.6899 | 7.7274 | 7.6070 | 7.9188 | 0.1063 | 8.9986 | n/a | 0.0430 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | PyTorch CPU | 0.0368 | 0.0373 | 0.0289 | 0.0445 | 0.0052 | 8.9986 | n/a | 0.0430 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7904 | 0.7941 | 0.6993 | 0.8557 | 0.0573 | 32.7840 | n/a | 0.0116 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 25.9122 | 25.8972 | 25.5252 | 26.3353 | 0.2620 | 32.7840 | n/a | 0.0116 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | PyTorch CPU | 0.0091 | 0.0102 | 0.0084 | 0.0136 | 0.0019 | 32.7840 | n/a | 0.0116 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0276 | 0.0267 | 0.0203 | 0.0304 | 0.0035 | n/a | n/a | 6.2110 | n/a | n/a | n/a | yes | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1,)` | PyTorch CPU | 0.1715 | 0.1617 | 0.1305 | 0.1814 | 0.0195 | n/a | n/a | 6.2110 | n/a | n/a | n/a | yes | n/a | TensorStudio | reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0645 | 0.0629 | 0.0580 | 0.0657 | 0.0030 | n/a | n/a | 3.2104 | n/a | n/a | n/a | yes | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | PyTorch CPU | 0.2070 | 0.2419 | 0.1908 | 0.3435 | 0.0584 | n/a | n/a | 3.2104 | n/a | n/a | n/a | yes | n/a | TensorStudio | reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1398 | 0.1474 | 0.1392 | 0.1611 | 0.0098 | n/a | n/a | 0.7448 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | scalar_backward | `(128,)` | PyTorch CPU | 0.1041 | 0.1093 | 0.1014 | 0.1294 | 0.0103 | n/a | n/a | 0.7448 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3304 | 0.3288 | 0.3216 | 0.3332 | 0.0040 | n/a | n/a | 0.5756 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | PyTorch CPU | 0.1902 | 0.1925 | 0.1885 | 0.1998 | 0.0045 | n/a | n/a | 0.5756 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1458 | 1.1837 | 1.0484 | 1.4499 | 0.1452 | n/a | n/a | 0.0950 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | PyTorch CPU | 0.1088 | 0.1144 | 0.1046 | 0.1297 | 0.0095 | n/a | n/a | 0.0950 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.4216 | 2.4803 | 2.3992 | 2.6474 | 0.0946 | n/a | n/a | 0.0788 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | PyTorch CPU | 0.1908 | 0.1917 | 0.1897 | 0.1946 | 0.0020 | n/a | n/a | 0.0788 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.5116 | 1.6913 | 1.4933 | 2.4237 | 0.3663 | 0.0894 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1352 | 0.1378 | 0.1350 | 0.1441 | 0.0036 | 0.0894 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
