# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.2.0`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: available (2.12.1+cpu)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `26`
- TensorStudio losses versus NumPy: `77`
- TensorStudio wins versus PyTorch CPU: `75`
- TensorStudio losses versus PyTorch CPU: `33`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0000 | 0.2788 | n/a | 1.4769 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2788 | n/a | 1.4769 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | PyTorch CPU | 0.0035 | 0.0038 | 0.0034 | 0.0044 | 0.0004 | 0.2788 | n/a | 1.4769 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0025 | 0.0025 | 0.0023 | 0.0026 | 0.0001 | 0.2879 | n/a | 1.4623 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2879 | n/a | 1.4623 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | PyTorch CPU | 0.0037 | 0.0042 | 0.0035 | 0.0059 | 0.0009 | 0.2879 | n/a | 1.4623 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0026 | 0.0001 | 0.3292 | n/a | 1.5234 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3292 | n/a | 1.5234 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0035 | 0.0040 | 0.0002 | 0.3292 | n/a | 1.5234 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.3418 | n/a | 1.4707 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0009 | 0.0007 | 0.0012 | 0.0002 | 0.3418 | n/a | 1.4707 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0034 | 0.0036 | 0.0001 | 0.3418 | n/a | 1.4707 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0089 | 0.0089 | 0.0087 | 0.0090 | 0.0001 | 0.4601 | n/a | 8.1568 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0041 | 0.0041 | 0.0039 | 0.0044 | 0.0002 | 0.4601 | n/a | 8.1568 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | PyTorch CPU | 0.0723 | 0.0702 | 0.0590 | 0.0807 | 0.0086 | 0.4601 | n/a | 8.1568 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0037 | 0.0036 | 0.0028 | 0.0040 | 0.0005 | 0.4094 | n/a | 1.4091 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | 0.4094 | n/a | 1.4091 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | PyTorch CPU | 0.0053 | 0.0058 | 0.0050 | 0.0081 | 0.0012 | 0.4094 | n/a | 1.4091 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0023 | 0.0027 | 0.0023 | 0.0039 | 0.0006 | 0.4661 | n/a | 2.2341 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0011 | 0.0011 | 0.0007 | 0.0015 | 0.0003 | 0.4661 | n/a | 2.2341 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | PyTorch CPU | 0.0052 | 0.0048 | 0.0037 | 0.0054 | 0.0007 | 0.4661 | n/a | 2.2341 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0023 | 0.0026 | 0.0023 | 0.0036 | 0.0005 | 0.3046 | n/a | 1.6425 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.3046 | n/a | 1.6425 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | PyTorch CPU | 0.0038 | 0.0045 | 0.0035 | 0.0074 | 0.0015 | 0.3046 | n/a | 1.6425 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0025 | 0.0026 | 0.0023 | 0.0032 | 0.0003 | 0.3015 | n/a | 1.6368 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0008 | 0.0010 | 0.0007 | 0.0015 | 0.0004 | 0.3015 | n/a | 1.6368 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | PyTorch CPU | 0.0041 | 0.0048 | 0.0034 | 0.0074 | 0.0015 | 0.3015 | n/a | 1.6368 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0106 | 0.0115 | 0.0093 | 0.0146 | 0.0019 | 0.8014 | n/a | 6.8788 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0085 | 0.0084 | 0.0074 | 0.0091 | 0.0006 | 0.8014 | n/a | 6.8788 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | PyTorch CPU | 0.0728 | 0.0740 | 0.0713 | 0.0791 | 0.0028 | 0.8014 | n/a | 6.8788 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0029 | 0.0029 | 0.0024 | 0.0033 | 0.0003 | 0.2350 | n/a | 1.5764 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0011 | 0.0002 | 0.2350 | n/a | 1.5764 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | PyTorch CPU | 0.0045 | 0.0053 | 0.0035 | 0.0093 | 0.0020 | 0.2350 | n/a | 1.5764 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0025 | 0.0028 | 0.0025 | 0.0034 | 0.0004 | 0.2888 | n/a | 2.0013 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.2888 | n/a | 2.0013 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | PyTorch CPU | 0.0051 | 0.0050 | 0.0035 | 0.0072 | 0.0014 | 0.2888 | n/a | 2.0013 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0030 | 0.0034 | 0.0023 | 0.0051 | 0.0011 | 0.3532 | n/a | 1.2188 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0011 | 0.0010 | 0.0007 | 0.0015 | 0.0003 | 0.3532 | n/a | 1.2188 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | PyTorch CPU | 0.0037 | 0.0040 | 0.0036 | 0.0053 | 0.0006 | 0.3532 | n/a | 1.2188 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0028 | 0.0029 | 0.0023 | 0.0038 | 0.0005 | 0.2714 | n/a | 1.4621 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2714 | n/a | 1.4621 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | PyTorch CPU | 0.0040 | 0.0045 | 0.0035 | 0.0058 | 0.0008 | 0.2714 | n/a | 1.4621 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0105 | 0.0102 | 0.0094 | 0.0108 | 0.0005 | 0.9145 | n/a | 7.4136 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0096 | 0.0089 | 0.0043 | 0.0126 | 0.0027 | 0.9145 | n/a | 7.4136 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | PyTorch CPU | 0.0778 | 0.0772 | 0.0742 | 0.0807 | 0.0026 | 0.9145 | n/a | 7.4136 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0026 | 0.0027 | 0.0024 | 0.0036 | 0.0004 | 0.2861 | n/a | 1.5555 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2861 | n/a | 1.5555 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | PyTorch CPU | 0.0041 | 0.0060 | 0.0038 | 0.0093 | 0.0025 | 0.2861 | n/a | 1.5555 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0063 | 0.0052 | 0.0025 | 0.0072 | 0.0021 | 0.1543 | n/a | 0.9674 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0010 | 0.0014 | 0.0008 | 0.0023 | 0.0006 | 0.1543 | n/a | 0.9674 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | PyTorch CPU | 0.0061 | 0.0059 | 0.0046 | 0.0071 | 0.0008 | 0.1543 | n/a | 0.9674 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0041 | 0.0049 | 0.0033 | 0.0070 | 0.0017 | 0.2109 | n/a | 1.8445 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0009 | 0.0010 | 0.0008 | 0.0016 | 0.0003 | 0.2109 | n/a | 1.8445 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | PyTorch CPU | 0.0077 | 0.0076 | 0.0056 | 0.0093 | 0.0012 | 0.2109 | n/a | 1.8445 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0047 | 0.0042 | 0.0029 | 0.0049 | 0.0008 | 0.2827 | n/a | 0.9863 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0013 | 0.0013 | 0.0008 | 0.0018 | 0.0004 | 0.2827 | n/a | 0.9863 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | PyTorch CPU | 0.0046 | 0.0044 | 0.0035 | 0.0049 | 0.0005 | 0.2827 | n/a | 0.9863 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0112 | 0.0115 | 0.0104 | 0.0127 | 0.0010 | 0.4575 | n/a | 6.4533 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0051 | 0.0052 | 0.0044 | 0.0065 | 0.0007 | 0.4575 | n/a | 6.4533 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | PyTorch CPU | 0.0726 | 0.0732 | 0.0686 | 0.0802 | 0.0040 | 0.4575 | n/a | 6.4533 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0032 | 0.0033 | 0.0029 | 0.0039 | 0.0003 | 0.3078 | n/a | 2.0046 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0012 | 0.0010 | 0.0020 | 0.0004 | 0.3078 | n/a | 2.0046 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | PyTorch CPU | 0.0064 | 0.0064 | 0.0040 | 0.0093 | 0.0017 | 0.3078 | n/a | 2.0046 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0033 | 0.0035 | 0.0027 | 0.0046 | 0.0007 | 0.2913 | n/a | 1.8331 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.2913 | n/a | 1.8331 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | PyTorch CPU | 0.0061 | 0.0062 | 0.0040 | 0.0092 | 0.0021 | 0.2913 | n/a | 1.8331 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0027 | 0.0028 | 0.0027 | 0.0033 | 0.0002 | 0.7323 | n/a | 1.3862 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0020 | 0.0020 | 0.0012 | 0.0024 | 0.0004 | 0.7323 | n/a | 1.3862 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | PyTorch CPU | 0.0037 | 0.0048 | 0.0035 | 0.0085 | 0.0019 | 0.7323 | n/a | 1.3862 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0032 | 0.0034 | 0.0031 | 0.0043 | 0.0005 | 0.7888 | n/a | 2.6246 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0025 | 0.0022 | 0.0012 | 0.0026 | 0.0005 | 0.7888 | n/a | 2.6246 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | PyTorch CPU | 0.0083 | 0.0071 | 0.0044 | 0.0092 | 0.0021 | 0.7888 | n/a | 2.6246 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0180 | 0.0165 | 0.0123 | 0.0198 | 0.0028 | 0.5178 | n/a | 3.8453 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0093 | 0.0082 | 0.0053 | 0.0102 | 0.0020 | 0.5178 | n/a | 3.8453 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | PyTorch CPU | 0.0694 | 0.0715 | 0.0629 | 0.0905 | 0.0099 | 0.5178 | n/a | 3.8453 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0058 | 0.0061 | 0.0046 | 0.0077 | 0.0013 | 0.2792 | n/a | 1.3122 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0016 | 0.0017 | 0.0015 | 0.0021 | 0.0002 | 0.2792 | n/a | 1.3122 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | PyTorch CPU | 0.0076 | 0.0069 | 0.0044 | 0.0083 | 0.0015 | 0.2792 | n/a | 1.3122 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0064 | 0.0062 | 0.0054 | 0.0068 | 0.0005 | 0.2215 | n/a | 0.7206 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.2215 | n/a | 0.7206 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | PyTorch CPU | 0.0046 | 0.0048 | 0.0044 | 0.0057 | 0.0005 | 0.2215 | n/a | 0.7206 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0080 | 0.0074 | 0.0049 | 0.0088 | 0.0013 | 0.1841 | n/a | 0.5530 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.1841 | n/a | 0.5530 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | PyTorch CPU | 0.0044 | 0.0047 | 0.0043 | 0.0055 | 0.0005 | 0.1841 | n/a | 0.5530 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0053 | 0.0055 | 0.0051 | 0.0062 | 0.0004 | 0.3078 | n/a | 1.2726 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.3078 | n/a | 1.2726 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | PyTorch CPU | 0.0067 | 0.0076 | 0.0043 | 0.0117 | 0.0032 | 0.3078 | n/a | 1.2726 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0230 | 0.0228 | 0.0189 | 0.0263 | 0.0029 | 0.3723 | n/a | 3.3247 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0086 | 0.0087 | 0.0080 | 0.0104 | 0.0009 | 0.3723 | n/a | 3.3247 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | PyTorch CPU | 0.0766 | 0.0764 | 0.0720 | 0.0798 | 0.0025 | 0.3723 | n/a | 3.3247 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0109 | 0.0112 | 0.0106 | 0.0128 | 0.0008 | 0.3281 | n/a | 1.1351 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0036 | 0.0000 | 0.3281 | n/a | 1.1351 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | PyTorch CPU | 0.0124 | 0.0128 | 0.0076 | 0.0166 | 0.0035 | 0.3281 | n/a | 1.1351 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0142 | 0.0140 | 0.0112 | 0.0185 | 0.0027 | 0.2530 | n/a | 0.7944 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0037 | 0.0001 | 0.2530 | n/a | 0.7944 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | PyTorch CPU | 0.0113 | 0.0122 | 0.0076 | 0.0170 | 0.0039 | 0.2530 | n/a | 0.7944 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0173 | 0.0169 | 0.0150 | 0.0177 | 0.0010 | 0.2587 | n/a | 0.4207 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0045 | 0.0045 | 0.0038 | 0.0050 | 0.0004 | 0.2587 | n/a | 0.4207 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | PyTorch CPU | 0.0073 | 0.0073 | 0.0072 | 0.0074 | 0.0001 | 0.2587 | n/a | 0.4207 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0190 | 0.0170 | 0.0132 | 0.0194 | 0.0026 | 0.2221 | n/a | 0.3932 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0045 | 0.0042 | 0.0059 | 0.0007 | 0.2221 | n/a | 0.3932 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | PyTorch CPU | 0.0075 | 0.0103 | 0.0074 | 0.0152 | 0.0035 | 0.2221 | n/a | 0.3932 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0536 | 0.0538 | 0.0462 | 0.0609 | 0.0049 | 0.4724 | n/a | 1.4109 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0253 | 0.0246 | 0.0195 | 0.0298 | 0.0043 | 0.4724 | n/a | 1.4109 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | PyTorch CPU | 0.0756 | 0.0761 | 0.0716 | 0.0837 | 0.0044 | 0.4724 | n/a | 1.4109 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0021 | 0.0022 | 0.0015 | 0.0033 | 0.0006 | 0.7383 | n/a | 13.4465 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0015 | 0.0016 | 0.0014 | 0.0021 | 0.0003 | 0.7383 | n/a | 13.4465 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1,)` | PyTorch CPU | 0.0280 | 0.0280 | 0.0235 | 0.0312 | 0.0026 | 0.7383 | n/a | 13.4465 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0024 | 0.0022 | 0.0015 | 0.0029 | 0.0005 | 1.7762 | n/a | 31.6097 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0042 | 0.0044 | 0.0036 | 0.0055 | 0.0007 | 1.7762 | n/a | 31.6097 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | PyTorch CPU | 0.0746 | 0.0739 | 0.0668 | 0.0842 | 0.0064 | 1.7762 | n/a | 31.6097 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0015 | 0.0020 | 0.0002 | 0.6517 | n/a | 14.4244 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.6517 | n/a | 14.4244 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | PyTorch CPU | 0.0266 | 0.0275 | 0.0238 | 0.0364 | 0.0046 | 0.6517 | n/a | 14.4244 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0016 | 0.0017 | 0.0015 | 0.0019 | 0.0001 | 0.9191 | n/a | 14.8956 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0014 | 0.0016 | 0.0012 | 0.0028 | 0.0006 | 0.9191 | n/a | 14.8956 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1,)` | PyTorch CPU | 0.0234 | 0.0234 | 0.0227 | 0.0244 | 0.0006 | 0.9191 | n/a | 14.8956 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0016 | 0.0022 | 0.0002 | 0.6429 | n/a | 12.5829 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0012 | 0.0014 | 0.0012 | 0.0019 | 0.0003 | 0.6429 | n/a | 12.5829 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1,)` | PyTorch CPU | 0.0233 | 0.0232 | 0.0224 | 0.0237 | 0.0004 | 0.6429 | n/a | 12.5829 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | 0.9530 | n/a | 15.2188 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0016 | 0.0001 | 0.9530 | n/a | 15.2188 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(8,)` | PyTorch CPU | 0.0227 | 0.0229 | 0.0226 | 0.0234 | 0.0004 | 0.9530 | n/a | 15.2188 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0015 | 0.0019 | 0.0001 | 2.3215 | n/a | 38.7085 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0037 | 0.0037 | 0.0035 | 0.0041 | 0.0002 | 2.3215 | n/a | 38.7085 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | PyTorch CPU | 0.0615 | 0.0618 | 0.0607 | 0.0635 | 0.0010 | 2.3215 | n/a | 38.7085 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0020 | 0.0002 | 0.7863 | n/a | 14.3019 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.7863 | n/a | 14.3019 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | PyTorch CPU | 0.0230 | 0.0230 | 0.0230 | 0.0231 | 0.0001 | 0.7863 | n/a | 14.3019 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0018 | 0.0018 | 0.0016 | 0.0019 | 0.0001 | 0.6600 | n/a | 13.1353 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0011 | 0.0015 | 0.0001 | 0.6600 | n/a | 13.1353 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(8,)` | PyTorch CPU | 0.0231 | 0.0232 | 0.0228 | 0.0238 | 0.0003 | 0.6600 | n/a | 13.1353 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0016 | 0.0018 | 0.0016 | 0.0023 | 0.0003 | 0.7422 | n/a | 14.1766 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0002 | 0.7422 | n/a | 14.1766 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(8,)` | PyTorch CPU | 0.0233 | 0.0235 | 0.0230 | 0.0247 | 0.0006 | 0.7422 | n/a | 14.1766 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0015 | 0.0017 | 0.0014 | 0.0021 | 0.0003 | 1.1657 | n/a | 15.5757 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0014 | 0.0029 | 0.0005 | 1.1657 | n/a | 15.5757 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | relu | `(32,)` | PyTorch CPU | 0.0228 | 0.0230 | 0.0226 | 0.0241 | 0.0005 | 1.1657 | n/a | 15.5757 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0017 | 0.0019 | 0.0017 | 0.0024 | 0.0003 | 2.1691 | n/a | 35.3154 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0038 | 0.0039 | 0.0034 | 0.0046 | 0.0004 | 2.1691 | n/a | 35.3154 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | PyTorch CPU | 0.0614 | 0.0622 | 0.0603 | 0.0660 | 0.0020 | 2.1691 | n/a | 35.3154 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0019 | 0.0020 | 0.0018 | 0.0023 | 0.0002 | 0.6651 | n/a | 14.7969 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.6651 | n/a | 14.7969 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | PyTorch CPU | 0.0282 | 0.0275 | 0.0236 | 0.0313 | 0.0029 | 0.6651 | n/a | 14.7969 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0020 | 0.0021 | 0.0018 | 0.0026 | 0.0003 | 1.0339 | n/a | 13.2069 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0021 | 0.0020 | 0.0012 | 0.0030 | 0.0007 | 1.0339 | n/a | 13.2069 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | exp | `(32,)` | PyTorch CPU | 0.0269 | 0.0269 | 0.0257 | 0.0281 | 0.0008 | 1.0339 | n/a | 13.2069 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | log | `(32,)` | TensorStudio | 0.0020 | 0.0027 | 0.0018 | 0.0041 | 0.0010 | 0.6777 | n/a | 14.0673 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0016 | 0.0013 | 0.0023 | 0.0004 | 0.6777 | n/a | 14.0673 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(32,)` | PyTorch CPU | 0.0280 | 0.0291 | 0.0272 | 0.0340 | 0.0025 | 0.6777 | n/a | 14.0673 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0017 | 0.0018 | 0.0016 | 0.0021 | 0.0002 | 0.9934 | n/a | 17.9408 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0016 | 0.0020 | 0.0015 | 0.0032 | 0.0007 | 0.9934 | n/a | 17.9408 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(128,)` | PyTorch CPU | 0.0296 | 0.0314 | 0.0256 | 0.0417 | 0.0054 | 0.9934 | n/a | 17.9408 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0046 | 0.0046 | 0.0036 | 0.0055 | 0.0006 | 1.1985 | n/a | 16.6852 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0056 | 0.0063 | 0.0048 | 0.0089 | 0.0015 | 1.1985 | n/a | 16.6852 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | PyTorch CPU | 0.0774 | 0.0773 | 0.0717 | 0.0815 | 0.0038 | 1.1985 | n/a | 16.6852 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0041 | 0.0042 | 0.0027 | 0.0059 | 0.0013 | 0.4285 | n/a | 6.9716 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0017 | 0.0022 | 0.0016 | 0.0032 | 0.0008 | 0.4285 | n/a | 6.9716 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | PyTorch CPU | 0.0284 | 0.0286 | 0.0258 | 0.0308 | 0.0018 | 0.4285 | n/a | 6.9716 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0042 | 0.0044 | 0.0040 | 0.0052 | 0.0004 | 0.4707 | n/a | 7.4762 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0020 | 0.0018 | 0.0013 | 0.0024 | 0.0004 | 0.4707 | n/a | 7.4762 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(128,)` | PyTorch CPU | 0.0314 | 0.0315 | 0.0293 | 0.0333 | 0.0016 | 0.4707 | n/a | 7.4762 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0035 | 0.0036 | 0.0026 | 0.0045 | 0.0008 | 0.8134 | n/a | 9.0094 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0028 | 0.0026 | 0.0018 | 0.0030 | 0.0005 | 0.8134 | n/a | 9.0094 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(128,)` | PyTorch CPU | 0.0316 | 0.0301 | 0.0265 | 0.0330 | 0.0028 | 0.8134 | n/a | 9.0094 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0027 | 0.0029 | 0.0022 | 0.0043 | 0.0007 | 1.0032 | n/a | 12.0592 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0027 | 0.0029 | 0.0020 | 0.0042 | 0.0009 | 1.0032 | n/a | 12.0592 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | relu | `(1024,)` | PyTorch CPU | 0.0321 | 0.0315 | 0.0287 | 0.0348 | 0.0025 | 1.0032 | n/a | 12.0592 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0110 | 0.0128 | 0.0092 | 0.0194 | 0.0039 | 0.5496 | n/a | 6.4952 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0060 | 0.0071 | 0.0058 | 0.0106 | 0.0018 | 0.5496 | n/a | 6.4952 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | PyTorch CPU | 0.0714 | 0.0736 | 0.0708 | 0.0804 | 0.0037 | 0.5496 | n/a | 6.4952 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0172 | 0.0194 | 0.0147 | 0.0274 | 0.0045 | 0.2899 | n/a | 1.6216 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0050 | 0.0051 | 0.0044 | 0.0063 | 0.0007 | 0.2899 | n/a | 1.6216 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | PyTorch CPU | 0.0279 | 0.0299 | 0.0257 | 0.0407 | 0.0056 | 0.2899 | n/a | 1.6216 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0102 | 0.0111 | 0.0081 | 0.0147 | 0.0027 | 0.4889 | n/a | 3.0029 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0050 | 0.0042 | 0.0026 | 0.0054 | 0.0013 | 0.4889 | n/a | 3.0029 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | PyTorch CPU | 0.0306 | 0.0297 | 0.0244 | 0.0319 | 0.0027 | 0.4889 | n/a | 3.0029 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0122 | 0.0124 | 0.0102 | 0.0143 | 0.0014 | 0.2756 | n/a | 2.2089 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0034 | 0.0038 | 0.0030 | 0.0055 | 0.0009 | 0.2756 | n/a | 2.2089 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1024,)` | PyTorch CPU | 0.0269 | 0.0276 | 0.0241 | 0.0323 | 0.0031 | 0.2756 | n/a | 2.2089 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0063 | 0.0058 | 0.0046 | 0.0072 | 0.0010 | 0.5161 | n/a | 4.8882 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0032 | 0.0033 | 0.0032 | 0.0033 | 0.0000 | 0.5161 | n/a | 4.8882 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | PyTorch CPU | 0.0308 | 0.0304 | 0.0248 | 0.0357 | 0.0037 | 0.5161 | n/a | 4.8882 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0453 | 0.0466 | 0.0412 | 0.0530 | 0.0049 | 0.3110 | n/a | 2.5477 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0141 | 0.0139 | 0.0114 | 0.0156 | 0.0014 | 0.3110 | n/a | 2.5477 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | PyTorch CPU | 0.1154 | 0.1128 | 0.1040 | 0.1172 | 0.0049 | 0.3110 | n/a | 2.5477 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0926 | 0.0904 | 0.0784 | 0.0984 | 0.0075 | 0.1593 | n/a | 0.4555 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0147 | 0.0153 | 0.0124 | 0.0191 | 0.0022 | 0.1593 | n/a | 0.4555 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | PyTorch CPU | 0.0422 | 0.0459 | 0.0408 | 0.0554 | 0.0057 | 0.1593 | n/a | 0.4555 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0432 | 0.0452 | 0.0366 | 0.0562 | 0.0071 | 0.2853 | n/a | 0.9561 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0123 | 0.0112 | 0.0076 | 0.0134 | 0.0021 | 0.2853 | n/a | 0.9561 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | PyTorch CPU | 0.0413 | 0.0400 | 0.0348 | 0.0445 | 0.0038 | 0.2853 | n/a | 0.9561 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0608 | 0.0629 | 0.0561 | 0.0745 | 0.0069 | 0.2505 | n/a | 0.8053 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0152 | 0.0151 | 0.0141 | 0.0159 | 0.0008 | 0.2505 | n/a | 0.8053 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(4096,)` | PyTorch CPU | 0.0489 | 0.0475 | 0.0403 | 0.0503 | 0.0037 | 0.2505 | n/a | 0.8053 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0172 | 0.0176 | 0.0156 | 0.0197 | 0.0017 | 0.5999 | n/a | 1.8673 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0103 | 0.0103 | 0.0084 | 0.0133 | 0.0017 | 0.5999 | n/a | 1.8673 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | PyTorch CPU | 0.0322 | 0.0327 | 0.0285 | 0.0387 | 0.0033 | 0.5999 | n/a | 1.8673 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.2675 | 0.2584 | 0.1872 | 0.3098 | 0.0474 | 0.2195 | n/a | 0.4480 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0587 | 0.0573 | 0.0446 | 0.0751 | 0.0113 | 0.2195 | n/a | 0.4480 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | PyTorch CPU | 0.1198 | 0.1191 | 0.0995 | 0.1487 | 0.0185 | 0.2195 | n/a | 0.4480 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.4181 | 0.4180 | 0.3642 | 0.4648 | 0.0408 | 0.0989 | n/a | 0.1403 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0413 | 0.0427 | 0.0401 | 0.0473 | 0.0026 | 0.0989 | n/a | 0.1403 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | PyTorch CPU | 0.0587 | 0.0595 | 0.0567 | 0.0646 | 0.0027 | 0.0989 | n/a | 0.1403 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.2302 | 0.2236 | 0.1749 | 0.2639 | 0.0357 | 0.1993 | n/a | 0.2411 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0459 | 0.0498 | 0.0452 | 0.0568 | 0.0054 | 0.1993 | n/a | 0.2411 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | PyTorch CPU | 0.0555 | 0.0566 | 0.0504 | 0.0655 | 0.0050 | 0.1993 | n/a | 0.2411 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.2682 | 0.2323 | 0.1478 | 0.2703 | 0.0491 | 0.2108 | n/a | 0.1343 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0565 | 0.0486 | 0.0306 | 0.0619 | 0.0123 | 0.2108 | n/a | 0.1343 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| activations | log | `(16384,)` | PyTorch CPU | 0.0360 | 0.0360 | 0.0324 | 0.0386 | 0.0024 | 0.2108 | n/a | 0.1343 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0034 | 0.0035 | 0.0030 | 0.0039 | 0.0004 | 1.1425 | n/a | 3.3963 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0039 | 0.0039 | 0.0026 | 0.0051 | 0.0010 | 1.1425 | n/a | 3.3963 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0115 | 0.0118 | 0.0107 | 0.0136 | 0.0010 | 1.1425 | n/a | 3.3963 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0030 | 0.0034 | 0.0028 | 0.0042 | 0.0006 | 4.3390 | n/a | 6.7326 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0132 | 0.0125 | 0.0100 | 0.0144 | 0.0017 | 4.3390 | n/a | 6.7326 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0205 | 0.0192 | 0.0165 | 0.0210 | 0.0019 | 4.3390 | n/a | 6.7326 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0024 | 0.0024 | 0.0022 | 0.0027 | 0.0002 | 1.6530 | n/a | 3.2005 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0040 | 0.0039 | 0.0024 | 0.0048 | 0.0009 | 1.6530 | n/a | 3.2005 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0077 | 0.0075 | 0.0055 | 0.0098 | 0.0015 | 1.6530 | n/a | 3.2005 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 5.8517 | n/a | 9.4566 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0107 | 0.0106 | 0.0090 | 0.0124 | 0.0011 | 5.8517 | n/a | 9.4566 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0172 | 0.0171 | 0.0161 | 0.0179 | 0.0006 | 5.8517 | n/a | 9.4566 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0021 | 0.0023 | 0.0019 | 0.0035 | 0.0006 | 1.1567 | n/a | 3.6592 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0024 | 0.0025 | 0.0023 | 0.0028 | 0.0002 | 1.1567 | n/a | 3.6592 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0076 | 0.0076 | 0.0064 | 0.0090 | 0.0009 | 1.1567 | n/a | 3.6592 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0029 | 0.0031 | 0.0021 | 0.0044 | 0.0009 | 4.5638 | n/a | 6.4709 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0131 | 0.0122 | 0.0097 | 0.0149 | 0.0020 | 4.5638 | n/a | 6.4709 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0186 | 0.0189 | 0.0162 | 0.0221 | 0.0019 | 4.5638 | n/a | 6.4709 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0029 | 0.0033 | 0.0028 | 0.0040 | 0.0005 | 1.0515 | n/a | 3.0085 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0031 | 0.0031 | 0.0026 | 0.0037 | 0.0004 | 1.0515 | n/a | 3.0085 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0088 | 0.0083 | 0.0067 | 0.0100 | 0.0013 | 1.0515 | n/a | 3.0085 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0026 | 0.0029 | 0.0025 | 0.0036 | 0.0005 | 3.8825 | n/a | 6.1622 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0102 | 0.0107 | 0.0097 | 0.0121 | 0.0010 | 3.8825 | n/a | 6.1622 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0161 | 0.0179 | 0.0151 | 0.0241 | 0.0033 | 3.8825 | n/a | 6.1622 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0133 | 0.0128 | 0.0111 | 0.0142 | 0.0012 | 0.2067 | n/a | 0.4386 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0028 | 0.0031 | 0.0027 | 0.0044 | 0.0007 | 0.2067 | n/a | 0.4386 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0058 | 0.0068 | 0.0053 | 0.0101 | 0.0018 | 0.2067 | n/a | 0.4386 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0097 | 0.0100 | 0.0078 | 0.0123 | 0.0019 | 1.0427 | n/a | 2.1376 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0101 | 0.0105 | 0.0090 | 0.0118 | 0.0010 | 1.0427 | n/a | 2.1376 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0207 | 0.0214 | 0.0170 | 0.0265 | 0.0036 | 1.0427 | n/a | 2.1376 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0276 | 0.0286 | 0.0264 | 0.0328 | 0.0023 | 0.1292 | n/a | 0.2160 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0037 | 0.0001 | 0.1292 | n/a | 0.2160 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0060 | 0.0071 | 0.0059 | 0.0115 | 0.0022 | 0.1292 | n/a | 0.2160 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0337 | 0.0341 | 0.0320 | 0.0366 | 0.0015 | 0.3938 | n/a | 0.5081 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0133 | 0.0150 | 0.0096 | 0.0242 | 0.0055 | 0.3938 | n/a | 0.5081 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0171 | 0.0229 | 0.0143 | 0.0349 | 0.0087 | 0.3938 | n/a | 0.5081 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1070 | 0.1099 | 0.0963 | 0.1348 | 0.0132 | 0.0789 | n/a | 0.0775 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0084 | 0.0088 | 0.0083 | 0.0098 | 0.0006 | 0.0789 | n/a | 0.0775 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0083 | 0.0085 | 0.0082 | 0.0094 | 0.0004 | 0.0789 | n/a | 0.0775 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1403 | 0.1518 | 0.1105 | 0.2305 | 0.0432 | 0.0992 | n/a | 0.1799 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0139 | 0.0141 | 0.0134 | 0.0150 | 0.0006 | 0.0992 | n/a | 0.1799 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0252 | 0.0254 | 0.0155 | 0.0376 | 0.0089 | 0.0992 | n/a | 0.1799 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0027 | 0.0034 | 0.0021 | 0.0053 | 0.0013 | 1.0894 | n/a | 3.1408 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0029 | 0.0036 | 0.0027 | 0.0064 | 0.0014 | 1.0894 | n/a | 3.1408 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | PyTorch CPU | 0.0085 | 0.0103 | 0.0066 | 0.0200 | 0.0050 | 1.0894 | n/a | 3.1408 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0035 | 0.0036 | 0.0023 | 0.0052 | 0.0011 | 2.7795 | n/a | 5.5039 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0096 | 0.0112 | 0.0084 | 0.0173 | 0.0034 | 2.7795 | n/a | 5.5039 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | PyTorch CPU | 0.0190 | 0.0224 | 0.0152 | 0.0348 | 0.0075 | 2.7795 | n/a | 5.5039 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0099 | 0.0087 | 0.0058 | 0.0104 | 0.0018 | 0.5864 | n/a | 1.0375 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0058 | 0.0070 | 0.0051 | 0.0109 | 0.0021 | 0.5864 | n/a | 1.0375 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | PyTorch CPU | 0.0102 | 0.0099 | 0.0088 | 0.0111 | 0.0009 | 0.5864 | n/a | 1.0375 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0035 | 0.0036 | 0.0031 | 0.0041 | 0.0004 | 4.2275 | n/a | 6.1780 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0148 | 0.0133 | 0.0097 | 0.0167 | 0.0029 | 4.2275 | n/a | 6.1780 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | PyTorch CPU | 0.0216 | 0.0228 | 0.0191 | 0.0306 | 0.0040 | 4.2275 | n/a | 6.1780 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0161 | 0.0167 | 0.0156 | 0.0182 | 0.0011 | 0.4604 | n/a | 1.8466 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0074 | 0.0075 | 0.0074 | 0.0078 | 0.0001 | 0.4604 | n/a | 1.8466 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | PyTorch CPU | 0.0298 | 0.0294 | 0.0269 | 0.0306 | 0.0013 | 0.4604 | n/a | 1.8466 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0061 | 0.0064 | 0.0060 | 0.0071 | 0.0004 | 2.3642 | n/a | 5.6716 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0143 | 0.0155 | 0.0133 | 0.0222 | 0.0034 | 2.3642 | n/a | 5.6716 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | PyTorch CPU | 0.0344 | 0.0319 | 0.0194 | 0.0424 | 0.0094 | 2.3642 | n/a | 5.6716 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0781 | 0.0762 | 0.0712 | 0.0788 | 0.0030 | 0.2718 | n/a | 0.3104 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0212 | 0.0235 | 0.0207 | 0.0290 | 0.0034 | 0.2718 | n/a | 0.3104 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | PyTorch CPU | 0.0242 | 0.0248 | 0.0232 | 0.0271 | 0.0015 | 0.2718 | n/a | 0.3104 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0317 | 0.0320 | 0.0310 | 0.0339 | 0.0010 | 0.8060 | n/a | 1.0726 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0256 | 0.0257 | 0.0246 | 0.0266 | 0.0007 | 0.8060 | n/a | 1.0726 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | PyTorch CPU | 0.0340 | 0.0332 | 0.0264 | 0.0379 | 0.0043 | 0.8060 | n/a | 1.0726 | n/a | no | n/a | yes | n/a | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0076 | 0.0076 | 0.0070 | 0.0084 | 0.0005 | 0.5589 | n/a | 1.0908 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0043 | 0.0042 | 0.0039 | 0.0047 | 0.0003 | 0.5589 | n/a | 1.0908 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | PyTorch CPU | 0.0083 | 0.0087 | 0.0080 | 0.0098 | 0.0008 | 0.5589 | n/a | 1.0908 | n/a | no | n/a | yes | n/a | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0791 | 0.0778 | 0.0686 | 0.0838 | 0.0051 | 0.2819 | n/a | 0.1837 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0223 | 0.0222 | 0.0211 | 0.0233 | 0.0009 | 0.2819 | n/a | 0.1837 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(64, 64)` | PyTorch CPU | 0.0145 | 0.0148 | 0.0145 | 0.0153 | 0.0003 | 0.2819 | n/a | 0.1837 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.5607 | 0.5713 | 0.5291 | 0.6508 | 0.0443 | 0.2627 | n/a | 0.0658 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1473 | 0.1549 | 0.1383 | 0.1855 | 0.0163 | 0.2627 | n/a | 0.0658 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(128, 128)` | PyTorch CPU | 0.0369 | 0.0383 | 0.0364 | 0.0435 | 0.0027 | 0.2627 | n/a | 0.0658 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 4.4121 | 3.9329 | 2.5046 | 4.6563 | 0.7863 | 0.0856 | n/a | 0.0378 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.3779 | 0.4277 | 0.3136 | 0.6200 | 0.1060 | 0.0856 | n/a | 0.0378 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(256, 256)` | PyTorch CPU | 0.1669 | 0.1870 | 0.1462 | 0.2627 | 0.0424 | 0.0856 | n/a | 0.0378 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2374 | 0.2377 | 0.2069 | 0.2803 | 0.0275 | 7.2091 | n/a | 0.0784 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.7112 | 1.8947 | 1.5644 | 2.5174 | 0.3543 | 7.2091 | n/a | 0.0784 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | PyTorch CPU | 0.0186 | 0.0207 | 0.0151 | 0.0267 | 0.0049 | 7.2091 | n/a | 0.0784 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 9.5098 | 9.2718 | 8.3524 | 10.2324 | 0.7285 | 8.8563 | n/a | 0.0152 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 84.2210 | 86.4637 | 79.3654 | 95.9756 | 5.6737 | 8.8563 | n/a | 0.0152 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | PyTorch CPU | 0.1446 | 0.1372 | 0.1015 | 0.1724 | 0.0280 | 8.8563 | n/a | 0.0152 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0237 | 0.0230 | 0.0171 | 0.0276 | 0.0035 | 11.6685 | n/a | 0.3462 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.2764 | 0.2871 | 0.2404 | 0.3729 | 0.0478 | 11.6685 | n/a | 0.3462 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | PyTorch CPU | 0.0082 | 0.0092 | 0.0057 | 0.0137 | 0.0027 | 11.6685 | n/a | 0.3462 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0198 | 0.0191 | 0.0152 | 0.0221 | 0.0027 | 37.4930 | n/a | 0.3662 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.7431 | 0.7336 | 0.6641 | 0.8038 | 0.0530 | 37.4930 | n/a | 0.3662 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | PyTorch CPU | 0.0073 | 0.0074 | 0.0070 | 0.0081 | 0.0004 | 37.4930 | n/a | 0.3662 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.8440 | 0.8550 | 0.7786 | 0.9519 | 0.0560 | 12.4891 | n/a | 0.0452 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 10.5410 | 10.8089 | 10.1112 | 11.6085 | 0.5726 | 12.4891 | n/a | 0.0452 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | PyTorch CPU | 0.0381 | 0.0386 | 0.0360 | 0.0423 | 0.0022 | 12.4891 | n/a | 0.0452 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7984 | 0.8111 | 0.7000 | 0.9458 | 0.0816 | 45.4394 | n/a | 0.0204 | n/a | yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 36.2770 | 35.3866 | 29.3831 | 38.0896 | 3.1126 | 45.4394 | n/a | 0.0204 | n/a | yes | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | PyTorch CPU | 0.0163 | 0.0168 | 0.0158 | 0.0191 | 0.0012 | 45.4394 | n/a | 0.0204 | n/a | yes | n/a | no | n/a | PyTorch CPU | reference |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0264 | 0.0278 | 0.0257 | 0.0315 | 0.0022 | n/a | n/a | 6.5465 | n/a | n/a | n/a | yes | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1,)` | PyTorch CPU | 0.1731 | 0.1635 | 0.1259 | 0.1745 | 0.0189 | n/a | n/a | 6.5465 | n/a | n/a | n/a | yes | n/a | TensorStudio | reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0532 | 0.0498 | 0.0337 | 0.0653 | 0.0121 | n/a | n/a | 5.2589 | n/a | n/a | n/a | yes | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | PyTorch CPU | 0.2800 | 0.2674 | 0.2285 | 0.2991 | 0.0271 | n/a | n/a | 5.2589 | n/a | n/a | n/a | yes | n/a | TensorStudio | reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1768 | 0.1869 | 0.1677 | 0.2144 | 0.0200 | n/a | n/a | 0.6270 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | scalar_backward | `(128,)` | PyTorch CPU | 0.1109 | 0.1111 | 0.1048 | 0.1211 | 0.0057 | n/a | n/a | 0.6270 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.4277 | 0.4356 | 0.3972 | 0.4857 | 0.0288 | n/a | n/a | 0.5800 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | PyTorch CPU | 0.2481 | 0.2484 | 0.2150 | 0.2924 | 0.0255 | n/a | n/a | 0.5800 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.3767 | 1.3434 | 1.2566 | 1.4343 | 0.0717 | n/a | n/a | 0.1042 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | PyTorch CPU | 0.1434 | 0.1373 | 0.1068 | 0.1645 | 0.0204 | n/a | n/a | 0.1042 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 3.0196 | 3.0166 | 2.7784 | 3.2054 | 0.1510 | n/a | n/a | 0.0746 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | PyTorch CPU | 0.2254 | 0.2177 | 0.1864 | 0.2337 | 0.0178 | n/a | n/a | 0.0746 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 2.1022 | 2.0900 | 1.8655 | 2.4566 | 0.2068 | 0.0681 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1432 | 0.1436 | 0.1354 | 0.1562 | 0.0075 | 0.0681 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
