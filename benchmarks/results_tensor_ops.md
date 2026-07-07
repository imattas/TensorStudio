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

- TensorStudio wins versus NumPy: `12`
- TensorStudio losses versus NumPy: `72`
- TensorStudio wins versus PyTorch CPU: `64`
- TensorStudio losses versus PyTorch CPU: `20`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0025 | 0.0026 | 0.0023 | 0.0029 | 0.0002 | 0.2656 | n/a | 1.6595 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2656 | n/a | 1.6595 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | PyTorch CPU | 0.0042 | 0.0051 | 0.0039 | 0.0084 | 0.0017 | 0.2656 | n/a | 1.6595 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0034 | 0.0035 | 0.0026 | 0.0051 | 0.0009 | 0.4567 | n/a | 2.3163 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0016 | 0.0001 | 0.4567 | n/a | 2.3163 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | PyTorch CPU | 0.0078 | 0.0077 | 0.0052 | 0.0105 | 0.0018 | 0.4567 | n/a | 2.3163 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0030 | 0.0031 | 0.0027 | 0.0039 | 0.0004 | 0.5668 | n/a | 1.7806 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0017 | 0.0016 | 0.0013 | 0.0018 | 0.0002 | 0.5668 | n/a | 1.7806 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | PyTorch CPU | 0.0053 | 0.0056 | 0.0041 | 0.0075 | 0.0013 | 0.5668 | n/a | 1.7806 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0042 | 0.0042 | 0.0025 | 0.0058 | 0.0013 | 0.1719 | n/a | 0.9060 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.1719 | n/a | 0.9060 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | PyTorch CPU | 0.0038 | 0.0041 | 0.0035 | 0.0055 | 0.0007 | 0.1719 | n/a | 0.9060 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0156 | 0.0136 | 0.0093 | 0.0175 | 0.0035 | 0.2608 | n/a | 4.0877 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0041 | 0.0046 | 0.0040 | 0.0060 | 0.0008 | 0.2608 | n/a | 4.0877 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | PyTorch CPU | 0.0637 | 0.0660 | 0.0581 | 0.0798 | 0.0074 | 0.2608 | n/a | 4.0877 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0045 | 0.0044 | 0.0044 | 0.0045 | 0.0000 | 0.2059 | n/a | 0.9958 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0009 | 0.0010 | 0.0007 | 0.0013 | 0.0003 | 0.2059 | n/a | 0.9958 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | PyTorch CPU | 0.0044 | 0.0049 | 0.0035 | 0.0074 | 0.0015 | 0.2059 | n/a | 0.9958 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0023 | 0.0025 | 0.0023 | 0.0032 | 0.0003 | 0.5398 | n/a | 1.5579 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.5398 | n/a | 1.5579 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | PyTorch CPU | 0.0036 | 0.0042 | 0.0035 | 0.0069 | 0.0013 | 0.5398 | n/a | 1.5579 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3119 | n/a | 1.5627 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3119 | n/a | 1.5627 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | PyTorch CPU | 0.0035 | 0.0045 | 0.0034 | 0.0072 | 0.0015 | 0.3119 | n/a | 1.5627 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3213 | n/a | 2.7885 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.3213 | n/a | 2.7885 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | PyTorch CPU | 0.0063 | 0.0066 | 0.0059 | 0.0072 | 0.0005 | 0.3213 | n/a | 2.7885 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0154 | 0.0144 | 0.0098 | 0.0159 | 0.0023 | 0.2499 | n/a | 4.9341 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0043 | 0.0002 | 0.2499 | n/a | 4.9341 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | PyTorch CPU | 0.0762 | 0.0741 | 0.0682 | 0.0805 | 0.0049 | 0.2499 | n/a | 4.9341 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0024 | 0.0026 | 0.0023 | 0.0033 | 0.0004 | 0.2751 | n/a | 1.5553 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2751 | n/a | 1.5553 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | PyTorch CPU | 0.0037 | 0.0038 | 0.0034 | 0.0041 | 0.0003 | 0.2751 | n/a | 1.5553 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.2996 | n/a | 1.7017 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2996 | n/a | 1.7017 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | PyTorch CPU | 0.0039 | 0.0038 | 0.0035 | 0.0040 | 0.0002 | 0.2996 | n/a | 1.7017 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3048 | n/a | 1.7002 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3048 | n/a | 1.7002 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | PyTorch CPU | 0.0040 | 0.0044 | 0.0033 | 0.0066 | 0.0012 | 0.3048 | n/a | 1.7002 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0028 | 0.0002 | 0.3102 | n/a | 1.5735 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3102 | n/a | 1.5735 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | PyTorch CPU | 0.0036 | 0.0037 | 0.0034 | 0.0042 | 0.0003 | 0.3102 | n/a | 1.5735 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0086 | 0.0095 | 0.0084 | 0.0132 | 0.0018 | 0.4716 | n/a | 6.5351 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0041 | 0.0041 | 0.0038 | 0.0047 | 0.0003 | 0.4716 | n/a | 6.5351 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | PyTorch CPU | 0.0564 | 0.0594 | 0.0562 | 0.0645 | 0.0038 | 0.4716 | n/a | 6.5351 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0024 | 0.0026 | 0.0023 | 0.0030 | 0.0002 | 0.2784 | n/a | 1.5073 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2784 | n/a | 1.5073 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | PyTorch CPU | 0.0036 | 0.0037 | 0.0036 | 0.0039 | 0.0001 | 0.2784 | n/a | 1.5073 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0024 | 0.0025 | 0.0024 | 0.0030 | 0.0002 | 0.2846 | n/a | 1.5555 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2846 | n/a | 1.5555 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | PyTorch CPU | 0.0038 | 0.0041 | 0.0034 | 0.0058 | 0.0009 | 0.2846 | n/a | 1.5555 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0028 | 0.0028 | 0.0025 | 0.0034 | 0.0003 | 0.2519 | n/a | 1.2633 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2519 | n/a | 1.2633 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0034 | 0.0038 | 0.0002 | 0.2519 | n/a | 1.2633 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0023 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 0.3071 | n/a | 1.4985 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.3071 | n/a | 1.4985 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0039 | 0.0002 | 0.3071 | n/a | 1.4985 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0085 | 0.0085 | 0.0084 | 0.0086 | 0.0001 | 0.4532 | n/a | 6.7102 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0038 | 0.0038 | 0.0038 | 0.0039 | 0.0000 | 0.4532 | n/a | 6.7102 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | PyTorch CPU | 0.0567 | 0.0591 | 0.0565 | 0.0646 | 0.0033 | 0.4532 | n/a | 6.7102 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0027 | 0.0031 | 0.0001 | 0.4173 | n/a | 1.2542 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0012 | 0.0012 | 0.0009 | 0.0014 | 0.0002 | 0.4173 | n/a | 1.2542 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | PyTorch CPU | 0.0037 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 0.4173 | n/a | 1.2542 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0028 | 0.0033 | 0.0002 | 0.3110 | n/a | 1.5700 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0009 | 0.0010 | 0.0009 | 0.0012 | 0.0001 | 0.3110 | n/a | 1.5700 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | PyTorch CPU | 0.0047 | 0.0044 | 0.0038 | 0.0049 | 0.0005 | 0.3110 | n/a | 1.5700 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0027 | 0.0029 | 0.0026 | 0.0033 | 0.0003 | 0.3697 | n/a | 3.6969 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3697 | n/a | 3.6969 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | PyTorch CPU | 0.0099 | 0.0096 | 0.0047 | 0.0127 | 0.0027 | 0.3697 | n/a | 3.6969 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0070 | 0.0066 | 0.0056 | 0.0072 | 0.0007 | 0.3125 | n/a | 1.3107 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.3125 | n/a | 1.3107 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | PyTorch CPU | 0.0092 | 0.0073 | 0.0037 | 0.0093 | 0.0025 | 0.3125 | n/a | 1.3107 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0114 | 0.0139 | 0.0102 | 0.0210 | 0.0041 | 0.5026 | n/a | 5.2928 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0057 | 0.0057 | 0.0056 | 0.0060 | 0.0001 | 0.5026 | n/a | 5.2928 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | PyTorch CPU | 0.0603 | 0.0625 | 0.0584 | 0.0695 | 0.0042 | 0.5026 | n/a | 5.2928 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0044 | 0.0044 | 0.0044 | 0.0045 | 0.0000 | 0.3250 | n/a | 1.0369 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0014 | 0.0019 | 0.0014 | 0.0030 | 0.0006 | 0.3250 | n/a | 1.0369 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | PyTorch CPU | 0.0046 | 0.0049 | 0.0044 | 0.0059 | 0.0006 | 0.3250 | n/a | 1.0369 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0059 | 0.0065 | 0.0044 | 0.0103 | 0.0020 | 0.2662 | n/a | 0.7771 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0014 | 0.0018 | 0.0001 | 0.2662 | n/a | 0.7771 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | PyTorch CPU | 0.0046 | 0.0051 | 0.0046 | 0.0071 | 0.0010 | 0.2662 | n/a | 0.7771 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0045 | 0.0048 | 0.0043 | 0.0061 | 0.0007 | 0.3233 | n/a | 1.0137 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.3233 | n/a | 1.0137 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | PyTorch CPU | 0.0045 | 0.0046 | 0.0044 | 0.0051 | 0.0003 | 0.3233 | n/a | 1.0137 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0053 | 0.0053 | 0.0052 | 0.0054 | 0.0001 | 0.2960 | n/a | 0.9611 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.2960 | n/a | 0.9611 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | PyTorch CPU | 0.0051 | 0.0050 | 0.0045 | 0.0056 | 0.0004 | 0.2960 | n/a | 0.9611 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0182 | 0.0183 | 0.0175 | 0.0193 | 0.0006 | 0.4533 | n/a | 3.5422 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0083 | 0.0084 | 0.0081 | 0.0090 | 0.0004 | 0.4533 | n/a | 3.5422 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | PyTorch CPU | 0.0646 | 0.0650 | 0.0621 | 0.0693 | 0.0028 | 0.4533 | n/a | 3.5422 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0128 | 0.0141 | 0.0124 | 0.0196 | 0.0028 | 0.2853 | n/a | 0.7155 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 0.2853 | n/a | 0.7155 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | PyTorch CPU | 0.0091 | 0.0104 | 0.0073 | 0.0176 | 0.0038 | 0.2853 | n/a | 0.7155 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0093 | 0.0094 | 0.0092 | 0.0099 | 0.0003 | 0.4250 | n/a | 0.8157 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0040 | 0.0000 | 0.4250 | n/a | 0.8157 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | PyTorch CPU | 0.0076 | 0.0076 | 0.0076 | 0.0076 | 0.0000 | 0.4250 | n/a | 0.8157 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0102 | 0.0104 | 0.0098 | 0.0110 | 0.0005 | 0.3580 | n/a | 0.7193 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0036 | 0.0038 | 0.0036 | 0.0044 | 0.0003 | 0.3580 | n/a | 0.7193 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | PyTorch CPU | 0.0073 | 0.0074 | 0.0073 | 0.0075 | 0.0001 | 0.3580 | n/a | 0.7193 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0129 | 0.0129 | 0.0128 | 0.0131 | 0.0001 | 0.3285 | n/a | 0.5772 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0042 | 0.0042 | 0.0043 | 0.0000 | 0.3285 | n/a | 0.5772 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | PyTorch CPU | 0.0074 | 0.0077 | 0.0074 | 0.0083 | 0.0004 | 0.3285 | n/a | 0.5772 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0455 | 0.0457 | 0.0454 | 0.0463 | 0.0004 | 0.4025 | n/a | 1.5519 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0183 | 0.0185 | 0.0183 | 0.0191 | 0.0003 | 0.4025 | n/a | 1.5519 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | PyTorch CPU | 0.0705 | 0.0710 | 0.0694 | 0.0738 | 0.0015 | 0.4025 | n/a | 1.5519 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.8744 | n/a | 14.4949 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.8744 | n/a | 14.4949 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1,)` | PyTorch CPU | 0.0219 | 0.0219 | 0.0218 | 0.0220 | 0.0001 | 0.8744 | n/a | 14.4949 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 2.2228 | n/a | 38.5521 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0034 | 0.0034 | 0.0034 | 0.0034 | 0.0000 | 2.2228 | n/a | 38.5521 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | PyTorch CPU | 0.0591 | 0.0595 | 0.0585 | 0.0610 | 0.0010 | 2.2228 | n/a | 38.5521 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 0.7745 | n/a | 14.2700 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7745 | n/a | 14.2700 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | PyTorch CPU | 0.0223 | 0.0223 | 0.0222 | 0.0223 | 0.0001 | 0.7745 | n/a | 14.2700 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.7693 | n/a | 13.9809 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.7693 | n/a | 13.9809 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1,)` | PyTorch CPU | 0.0220 | 0.0221 | 0.0220 | 0.0226 | 0.0002 | 0.7693 | n/a | 13.9809 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.7794 | n/a | 13.6173 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0018 | 0.0012 | 0.0026 | 0.0006 | 0.7794 | n/a | 13.6173 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1,)` | PyTorch CPU | 0.0219 | 0.0221 | 0.0217 | 0.0229 | 0.0004 | 0.7794 | n/a | 13.6173 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | 0.9082 | n/a | 14.5329 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.9082 | n/a | 14.5329 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(8,)` | PyTorch CPU | 0.0219 | 0.0220 | 0.0219 | 0.0221 | 0.0001 | 0.9082 | n/a | 14.5329 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0018 | 0.0001 | 2.1307 | n/a | 36.0218 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0034 | 0.0034 | 0.0034 | 0.0035 | 0.0000 | 2.1307 | n/a | 36.0218 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | PyTorch CPU | 0.0581 | 0.0579 | 0.0574 | 0.0586 | 0.0004 | 2.1307 | n/a | 36.0218 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.7585 | n/a | 13.6880 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0001 | 0.7585 | n/a | 13.6880 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | PyTorch CPU | 0.0220 | 0.0221 | 0.0219 | 0.0225 | 0.0002 | 0.7585 | n/a | 13.6880 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.7573 | n/a | 13.5624 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.7573 | n/a | 13.5624 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(8,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0218 | 0.0219 | 0.0000 | 0.7573 | n/a | 13.5624 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.7523 | n/a | 13.2052 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.7523 | n/a | 13.2052 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(8,)` | PyTorch CPU | 0.0219 | 0.0219 | 0.0218 | 0.0221 | 0.0001 | 0.7523 | n/a | 13.2052 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.8856 | n/a | 14.5258 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.8856 | n/a | 14.5258 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(32,)` | PyTorch CPU | 0.0218 | 0.0219 | 0.0217 | 0.0222 | 0.0002 | 0.8856 | n/a | 14.5258 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0017 | 0.0019 | 0.0017 | 0.0023 | 0.0002 | 2.0347 | n/a | 35.5563 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0035 | 0.0035 | 0.0035 | 0.0036 | 0.0001 | 2.0347 | n/a | 35.5563 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | PyTorch CPU | 0.0615 | 0.0645 | 0.0580 | 0.0775 | 0.0070 | 2.0347 | n/a | 35.5563 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0024 | 0.0025 | 0.0021 | 0.0032 | 0.0004 | 0.6480 | n/a | 9.1404 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0016 | 0.0017 | 0.0014 | 0.0020 | 0.0002 | 0.6480 | n/a | 9.1404 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | PyTorch CPU | 0.0223 | 0.0225 | 0.0221 | 0.0233 | 0.0005 | 0.6480 | n/a | 9.1404 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0019 | 0.0001 | 0.7156 | n/a | 12.5559 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.7156 | n/a | 12.5559 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(32,)` | PyTorch CPU | 0.0221 | 0.0220 | 0.0217 | 0.0224 | 0.0003 | 0.7156 | n/a | 12.5559 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0020 | 0.0001 | 0.7361 | n/a | 12.4046 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.7361 | n/a | 12.4046 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(32,)` | PyTorch CPU | 0.0224 | 0.0243 | 0.0218 | 0.0311 | 0.0035 | 0.7361 | n/a | 12.4046 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0018 | 0.0001 | 0.8584 | n/a | 13.3579 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0016 | 0.0001 | 0.8584 | n/a | 13.3579 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(128,)` | PyTorch CPU | 0.0215 | 0.0215 | 0.0215 | 0.0216 | 0.0001 | 0.8584 | n/a | 13.3579 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0025 | 0.0025 | 0.0000 | 1.5065 | n/a | 23.4812 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0038 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 1.5065 | n/a | 23.4812 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | PyTorch CPU | 0.0587 | 0.0588 | 0.0582 | 0.0595 | 0.0005 | 1.5065 | n/a | 23.4812 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0026 | 0.0027 | 0.0026 | 0.0027 | 0.0001 | 0.5635 | n/a | 8.7047 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0018 | 0.0001 | 0.5635 | n/a | 8.7047 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | PyTorch CPU | 0.0228 | 0.0225 | 0.0221 | 0.0228 | 0.0003 | 0.5635 | n/a | 8.7047 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 0.5595 | n/a | 9.3320 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.5595 | n/a | 9.3320 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(128,)` | PyTorch CPU | 0.0224 | 0.0223 | 0.0218 | 0.0225 | 0.0002 | 0.5595 | n/a | 9.3320 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0025 | 0.0026 | 0.0025 | 0.0028 | 0.0002 | 0.5600 | n/a | 9.1071 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0017 | 0.0001 | 0.5600 | n/a | 9.1071 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(128,)` | PyTorch CPU | 0.0227 | 0.0232 | 0.0219 | 0.0265 | 0.0017 | 0.5600 | n/a | 9.1071 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0000 | 0.8944 | n/a | 10.1431 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0019 | 0.0021 | 0.0019 | 0.0025 | 0.0002 | 0.8944 | n/a | 10.1431 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | PyTorch CPU | 0.0220 | 0.0220 | 0.0218 | 0.0221 | 0.0001 | 0.8944 | n/a | 10.1431 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0090 | 0.0092 | 0.0089 | 0.0098 | 0.0003 | 0.6513 | n/a | 6.6296 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0059 | 0.0059 | 0.0058 | 0.0060 | 0.0001 | 0.6513 | n/a | 6.6296 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | PyTorch CPU | 0.0596 | 0.0595 | 0.0592 | 0.0597 | 0.0002 | 0.6513 | n/a | 6.6296 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0131 | 0.0131 | 0.0130 | 0.0132 | 0.0001 | 0.2835 | n/a | 1.8841 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0037 | 0.0038 | 0.0037 | 0.0042 | 0.0002 | 0.2835 | n/a | 1.8841 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | PyTorch CPU | 0.0246 | 0.0247 | 0.0243 | 0.0252 | 0.0003 | 0.2835 | n/a | 1.8841 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0080 | 0.0080 | 0.0080 | 0.0083 | 0.0001 | 0.3193 | n/a | 2.8389 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0025 | 0.0027 | 0.0025 | 0.0031 | 0.0002 | 0.3193 | n/a | 2.8389 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | PyTorch CPU | 0.0227 | 0.0227 | 0.0225 | 0.0228 | 0.0001 | 0.3193 | n/a | 2.8389 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0087 | 0.0088 | 0.0086 | 0.0092 | 0.0002 | 0.3374 | n/a | 2.6661 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 0.3374 | n/a | 2.6661 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1024,)` | PyTorch CPU | 0.0232 | 0.0235 | 0.0228 | 0.0250 | 0.0008 | 0.3374 | n/a | 2.6661 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0043 | 0.0044 | 0.0043 | 0.0044 | 0.0000 | 0.7207 | n/a | 5.2876 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0031 | 0.0033 | 0.0031 | 0.0035 | 0.0002 | 0.7207 | n/a | 5.2876 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | PyTorch CPU | 0.0229 | 0.0229 | 0.0226 | 0.0230 | 0.0001 | 0.7207 | n/a | 5.2876 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0327 | 0.0327 | 0.0316 | 0.0347 | 0.0011 | 0.3366 | n/a | 2.6914 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0110 | 0.0110 | 0.0107 | 0.0112 | 0.0002 | 0.3366 | n/a | 2.6914 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | PyTorch CPU | 0.0880 | 0.0951 | 0.0834 | 0.1130 | 0.0117 | 0.3366 | n/a | 2.6914 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0818 | 0.0860 | 0.0721 | 0.1004 | 0.0113 | 0.1311 | n/a | 0.5165 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0107 | 0.0107 | 0.0106 | 0.0109 | 0.0001 | 0.1311 | n/a | 0.5165 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | PyTorch CPU | 0.0422 | 0.0425 | 0.0326 | 0.0497 | 0.0057 | 0.1311 | n/a | 0.5165 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0405 | 0.0424 | 0.0359 | 0.0534 | 0.0060 | 0.1549 | n/a | 0.7109 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0063 | 0.0064 | 0.0059 | 0.0069 | 0.0004 | 0.1549 | n/a | 0.7109 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | PyTorch CPU | 0.0288 | 0.0293 | 0.0280 | 0.0315 | 0.0012 | 0.1549 | n/a | 0.7109 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0433 | 0.0444 | 0.0350 | 0.0601 | 0.0089 | 0.3429 | n/a | 0.7730 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0148 | 0.0138 | 0.0087 | 0.0156 | 0.0026 | 0.3429 | n/a | 0.7730 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(4096,)` | PyTorch CPU | 0.0334 | 0.0350 | 0.0299 | 0.0409 | 0.0037 | 0.3429 | n/a | 0.7730 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0133 | 0.0139 | 0.0132 | 0.0157 | 0.0010 | 0.7302 | n/a | 2.0934 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0097 | 0.0102 | 0.0090 | 0.0125 | 0.0012 | 0.7302 | n/a | 2.0934 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | PyTorch CPU | 0.0279 | 0.0295 | 0.0271 | 0.0353 | 0.0030 | 0.7302 | n/a | 2.0934 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.2031 | 0.1898 | 0.1472 | 0.2183 | 0.0292 | 0.2198 | n/a | 0.6618 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0446 | 0.0453 | 0.0356 | 0.0562 | 0.0081 | 0.2198 | n/a | 0.6618 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | PyTorch CPU | 0.1344 | 0.1360 | 0.1038 | 0.1674 | 0.0223 | 0.2198 | n/a | 0.6618 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3386 | 0.3424 | 0.3089 | 0.3919 | 0.0274 | 0.1839 | n/a | 0.1092 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0623 | 0.0609 | 0.0386 | 0.0783 | 0.0148 | 0.1839 | n/a | 0.1092 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| activations | tanh | `(16384,)` | PyTorch CPU | 0.0370 | 0.0386 | 0.0352 | 0.0437 | 0.0032 | 0.1839 | n/a | 0.1092 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1781 | 0.1776 | 0.1161 | 0.2378 | 0.0504 | 0.2240 | n/a | 0.1816 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0399 | 0.0407 | 0.0390 | 0.0443 | 0.0019 | 0.2240 | n/a | 0.1816 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| activations | exp | `(16384,)` | PyTorch CPU | 0.0323 | 0.0394 | 0.0290 | 0.0571 | 0.0116 | 0.2240 | n/a | 0.1816 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1712 | 0.1812 | 0.1254 | 0.2476 | 0.0489 | 0.2933 | n/a | 0.2683 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0502 | 0.0506 | 0.0482 | 0.0530 | 0.0016 | 0.2933 | n/a | 0.2683 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| activations | log | `(16384,)` | PyTorch CPU | 0.0459 | 0.0539 | 0.0370 | 0.0971 | 0.0222 | 0.2933 | n/a | 0.2683 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0021 | 0.0025 | 0.0015 | 0.0037 | 0.0009 | 1.2450 | n/a | 2.3591 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0026 | 0.0026 | 0.0018 | 0.0040 | 0.0008 | 1.2450 | n/a | 2.3591 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0049 | 0.0053 | 0.0047 | 0.0064 | 0.0007 | 1.2450 | n/a | 2.3591 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0015 | 0.0018 | 0.0014 | 0.0028 | 0.0005 | 6.9810 | n/a | 8.2745 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0105 | 0.0114 | 0.0094 | 0.0153 | 0.0022 | 6.9810 | n/a | 8.2745 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0124 | 0.0126 | 0.0115 | 0.0145 | 0.0010 | 6.9810 | n/a | 8.2745 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 1.1101 | n/a | 3.0492 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0016 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 1.1101 | n/a | 3.0492 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0001 | 1.1101 | n/a | 3.0492 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | 4.8852 | n/a | 7.7528 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0071 | 0.0071 | 0.0069 | 0.0073 | 0.0001 | 4.8852 | n/a | 7.7528 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0113 | 0.0113 | 0.0112 | 0.0116 | 0.0002 | 4.8852 | n/a | 7.7528 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 1.0404 | n/a | 2.8404 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 1.0404 | n/a | 2.8404 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0000 | 1.0404 | n/a | 2.8404 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 4.4855 | n/a | 7.1491 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0070 | 0.0070 | 0.0068 | 0.0071 | 0.0001 | 4.4855 | n/a | 7.1491 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0112 | 0.0112 | 0.0111 | 0.0113 | 0.0001 | 4.4855 | n/a | 7.1491 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0022 | 0.0001 | 0.7812 | n/a | 2.2072 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.7812 | n/a | 2.2072 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0046 | 0.0045 | 0.0043 | 0.0047 | 0.0001 | 0.7812 | n/a | 2.2072 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0021 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 3.2734 | n/a | 5.3696 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0070 | 0.0070 | 0.0069 | 0.0072 | 0.0001 | 3.2734 | n/a | 5.3696 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0114 | 0.0113 | 0.0110 | 0.0115 | 0.0002 | 3.2734 | n/a | 5.3696 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0070 | 0.0071 | 0.0070 | 0.0072 | 0.0001 | 0.2908 | n/a | 0.6507 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0020 | 0.0021 | 0.0020 | 0.0024 | 0.0001 | 0.2908 | n/a | 0.6507 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0045 | 0.0046 | 0.0045 | 0.0047 | 0.0001 | 0.2908 | n/a | 0.6507 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0074 | 0.0075 | 0.0073 | 0.0079 | 0.0002 | 1.0930 | n/a | 1.5425 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0081 | 0.0081 | 0.0076 | 0.0087 | 0.0004 | 1.0930 | n/a | 1.5425 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0114 | 0.0114 | 0.0113 | 0.0115 | 0.0001 | 1.0930 | n/a | 1.5425 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0243 | 0.0242 | 0.0239 | 0.0245 | 0.0002 | 0.1175 | n/a | 0.2069 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0029 | 0.0029 | 0.0028 | 0.0029 | 0.0000 | 0.1175 | n/a | 0.2069 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0050 | 0.0050 | 0.0049 | 0.0052 | 0.0001 | 0.1175 | n/a | 0.2069 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0242 | 0.0242 | 0.0241 | 0.0242 | 0.0000 | 0.3476 | n/a | 0.4996 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0084 | 0.0084 | 0.0082 | 0.0086 | 0.0001 | 0.3476 | n/a | 0.4996 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0121 | 0.0122 | 0.0117 | 0.0129 | 0.0004 | 0.3476 | n/a | 0.4996 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0942 | 0.0955 | 0.0921 | 0.1029 | 0.0038 | 0.0636 | n/a | 0.0726 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0060 | 0.0060 | 0.0060 | 0.0060 | 0.0000 | 0.0636 | n/a | 0.0726 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0068 | 0.0073 | 0.0067 | 0.0088 | 0.0008 | 0.0636 | n/a | 0.0726 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0929 | 0.0931 | 0.0927 | 0.0936 | 0.0004 | 0.1291 | n/a | 0.1520 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0120 | 0.0121 | 0.0119 | 0.0125 | 0.0002 | 0.1291 | n/a | 0.1520 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0141 | 0.0146 | 0.0139 | 0.0169 | 0.0012 | 0.1291 | n/a | 0.1520 | n/a | no | n/a | no | n/a | NumPy | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
