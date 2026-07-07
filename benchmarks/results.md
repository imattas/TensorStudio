# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.4.0`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `21`
- TensorStudio losses versus NumPy: `82`
- TensorStudio wins versus JAX CPU dispatch: `82`
- TensorStudio losses versus JAX CPU dispatch: `16`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0037 | 0.0037 | 0.0034 | 0.0042 | 0.0003 | 0.2530 | n/a | n/a | 4.6534 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0009 | 0.0010 | 0.0008 | 0.0011 | 0.0001 | 0.2530 | n/a | n/a | 4.6534 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0171 | 0.0181 | 0.0164 | 0.0204 | 0.0017 | 0.2530 | n/a | n/a | 4.6534 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0035 | 0.0035 | 0.0033 | 0.0038 | 0.0002 | 0.2537 | n/a | n/a | 4.4195 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0009 | 0.0009 | 0.0008 | 0.0010 | 0.0001 | 0.2537 | n/a | n/a | 4.4195 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0154 | 0.0156 | 0.0150 | 0.0166 | 0.0006 | 0.2537 | n/a | n/a | 4.4195 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0033 | 0.0035 | 0.0031 | 0.0043 | 0.0004 | 0.2510 | n/a | n/a | 4.6930 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0010 | 0.0001 | 0.2510 | n/a | n/a | 4.6930 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0157 | 0.0154 | 0.0123 | 0.0173 | 0.0019 | 0.2510 | n/a | n/a | 4.6930 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0031 | 0.0030 | 0.0027 | 0.0034 | 0.0003 | 0.2402 | n/a | n/a | 3.3549 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2402 | n/a | n/a | 3.3549 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0103 | 0.0107 | 0.0100 | 0.0121 | 0.0008 | 0.2402 | n/a | n/a | 3.3549 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0103 | 0.0102 | 0.0101 | 0.0104 | 0.0001 | 0.5893 | n/a | n/a | 9.1453 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0061 | 0.0061 | 0.0053 | 0.0071 | 0.0006 | 0.5893 | n/a | n/a | 9.1453 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0940 | 0.0959 | 0.0925 | 0.1011 | 0.0032 | 0.5893 | n/a | n/a | 9.1453 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0027 | 0.0028 | 0.0026 | 0.0032 | 0.0002 | 0.2534 | n/a | n/a | 4.8394 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0010 | 0.0001 | 0.2534 | n/a | n/a | 4.8394 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0131 | 0.0131 | 0.0123 | 0.0136 | 0.0004 | 0.2534 | n/a | n/a | 4.8394 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0030 | 0.0030 | 0.0027 | 0.0035 | 0.0003 | 0.2341 | n/a | n/a | 4.1405 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2341 | n/a | n/a | 4.1405 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0124 | 0.0124 | 0.0121 | 0.0127 | 0.0002 | 0.2341 | n/a | n/a | 4.1405 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0044 | 0.0042 | 0.0031 | 0.0050 | 0.0007 | 0.2878 | n/a | n/a | 3.7879 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0009 | 0.0017 | 0.0003 | 0.2878 | n/a | n/a | 3.7879 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0166 | 0.0172 | 0.0142 | 0.0219 | 0.0028 | 0.2878 | n/a | n/a | 3.7879 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0043 | 0.0039 | 0.0028 | 0.0047 | 0.0008 | 0.2219 | n/a | n/a | 2.8756 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0010 | 0.0012 | 0.0009 | 0.0016 | 0.0003 | 0.2219 | n/a | n/a | 2.8756 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0125 | 0.0133 | 0.0102 | 0.0169 | 0.0026 | 0.2219 | n/a | n/a | 2.8756 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0127 | 0.0126 | 0.0116 | 0.0136 | 0.0008 | 0.6136 | n/a | n/a | 8.8126 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0078 | 0.0075 | 0.0060 | 0.0087 | 0.0011 | 0.6136 | n/a | n/a | 8.8126 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1118 | 0.1151 | 0.1068 | 0.1285 | 0.0084 | 0.6136 | n/a | n/a | 8.8126 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0042 | 0.0041 | 0.0035 | 0.0048 | 0.0005 | 0.2307 | n/a | n/a | 3.6665 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0010 | 0.0010 | 0.0007 | 0.0012 | 0.0002 | 0.2307 | n/a | n/a | 3.6665 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0152 | 0.0153 | 0.0127 | 0.0182 | 0.0020 | 0.2307 | n/a | n/a | 3.6665 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0032 | 0.0031 | 0.0026 | 0.0035 | 0.0004 | 0.3866 | n/a | n/a | 5.1002 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0012 | 0.0012 | 0.0008 | 0.0015 | 0.0003 | 0.3866 | n/a | n/a | 5.1002 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0164 | 0.0165 | 0.0124 | 0.0197 | 0.0024 | 0.3866 | n/a | n/a | 5.1002 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0038 | 0.0040 | 0.0032 | 0.0052 | 0.0007 | 0.3289 | n/a | n/a | 4.1596 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0013 | 0.0011 | 0.0007 | 0.0014 | 0.0003 | 0.3289 | n/a | n/a | 4.1596 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0158 | 0.0158 | 0.0144 | 0.0171 | 0.0009 | 0.3289 | n/a | n/a | 4.1596 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0037 | 0.0037 | 0.0029 | 0.0044 | 0.0005 | 0.3093 | n/a | n/a | 4.0385 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0011 | 0.0011 | 0.0008 | 0.0016 | 0.0003 | 0.3093 | n/a | n/a | 4.0385 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0148 | 0.0143 | 0.0122 | 0.0156 | 0.0012 | 0.3093 | n/a | n/a | 4.0385 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0126 | 0.0129 | 0.0118 | 0.0148 | 0.0010 | 0.5922 | n/a | n/a | 8.9456 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0075 | 0.0072 | 0.0058 | 0.0080 | 0.0007 | 0.5922 | n/a | n/a | 8.9456 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.1131 | 0.1132 | 0.1035 | 0.1234 | 0.0066 | 0.5922 | n/a | n/a | 8.9456 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0030 | 0.0031 | 0.0028 | 0.0035 | 0.0002 | 0.2890 | n/a | n/a | 4.2054 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0009 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.2890 | n/a | n/a | 4.2054 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0126 | 0.0125 | 0.0122 | 0.0129 | 0.0003 | 0.2890 | n/a | n/a | 4.2054 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0031 | 0.0030 | 0.0027 | 0.0032 | 0.0002 | 0.2434 | n/a | n/a | 4.3311 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2434 | n/a | n/a | 4.3311 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0133 | 0.0135 | 0.0127 | 0.0149 | 0.0008 | 0.2434 | n/a | n/a | 4.3311 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0029 | 0.0029 | 0.0028 | 0.0031 | 0.0001 | 0.2828 | n/a | n/a | 4.7270 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.2828 | n/a | n/a | 4.7270 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0137 | 0.0138 | 0.0130 | 0.0147 | 0.0006 | 0.2828 | n/a | n/a | 4.7270 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0032 | 0.0031 | 0.0029 | 0.0034 | 0.0002 | 0.2336 | n/a | n/a | 3.0804 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0014 | 0.0003 | 0.2336 | n/a | n/a | 3.0804 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0098 | 0.0101 | 0.0096 | 0.0109 | 0.0006 | 0.2336 | n/a | n/a | 3.0804 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0109 | 0.0109 | 0.0105 | 0.0113 | 0.0003 | 0.5558 | n/a | n/a | 8.4986 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0061 | 0.0061 | 0.0059 | 0.0063 | 0.0002 | 0.5558 | n/a | n/a | 8.4986 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0930 | 0.0940 | 0.0914 | 0.0973 | 0.0025 | 0.5558 | n/a | n/a | 8.4986 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0033 | 0.0034 | 0.0031 | 0.0039 | 0.0003 | 0.3062 | n/a | n/a | 4.2459 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3062 | n/a | n/a | 4.2459 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0141 | 0.0143 | 0.0125 | 0.0157 | 0.0012 | 0.3062 | n/a | n/a | 4.2459 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0031 | 0.0031 | 0.0031 | 0.0031 | 0.0000 | 0.3425 | n/a | n/a | 4.3143 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0012 | 0.0010 | 0.0016 | 0.0002 | 0.3425 | n/a | n/a | 4.3143 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0132 | 0.0141 | 0.0130 | 0.0164 | 0.0013 | 0.3425 | n/a | n/a | 4.3143 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 0.3330 | n/a | n/a | 4.8399 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3330 | n/a | n/a | 4.8399 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0146 | 0.0149 | 0.0137 | 0.0162 | 0.0009 | 0.3330 | n/a | n/a | 4.8399 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0034 | 0.0035 | 0.0032 | 0.0038 | 0.0002 | 0.3238 | n/a | n/a | 3.2645 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.3238 | n/a | n/a | 3.2645 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0112 | 0.0111 | 0.0102 | 0.0117 | 0.0005 | 0.3238 | n/a | n/a | 3.2645 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0119 | 0.0121 | 0.0116 | 0.0129 | 0.0004 | 0.6478 | n/a | n/a | 9.8390 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0077 | 0.0076 | 0.0073 | 0.0078 | 0.0002 | 0.6478 | n/a | n/a | 9.8390 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1172 | 0.1179 | 0.1092 | 0.1307 | 0.0075 | 0.6478 | n/a | n/a | 9.8390 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0050 | 0.0052 | 0.0050 | 0.0055 | 0.0002 | 0.5239 | n/a | n/a | 2.8032 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0026 | 0.0025 | 0.0018 | 0.0032 | 0.0006 | 0.5239 | n/a | n/a | 2.8032 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0141 | 0.0147 | 0.0134 | 0.0167 | 0.0013 | 0.5239 | n/a | n/a | 2.8032 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0050 | 0.0049 | 0.0047 | 0.0051 | 0.0002 | 0.3018 | n/a | n/a | 3.0245 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.3018 | n/a | n/a | 3.0245 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0150 | 0.0151 | 0.0141 | 0.0159 | 0.0007 | 0.3018 | n/a | n/a | 3.0245 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0052 | 0.0053 | 0.0046 | 0.0065 | 0.0007 | 0.2811 | n/a | n/a | 2.8881 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.2811 | n/a | n/a | 2.8881 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0151 | 0.0167 | 0.0129 | 0.0206 | 0.0029 | 0.2811 | n/a | n/a | 2.8881 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0057 | 0.0057 | 0.0056 | 0.0059 | 0.0001 | 0.2934 | n/a | n/a | 2.0481 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0001 | 0.2934 | n/a | n/a | 2.0481 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0117 | 0.0115 | 0.0104 | 0.0127 | 0.0008 | 0.2934 | n/a | n/a | 2.0481 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0199 | 0.0198 | 0.0192 | 0.0204 | 0.0005 | 0.5088 | n/a | n/a | 6.0397 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0101 | 0.0106 | 0.0099 | 0.0128 | 0.0011 | 0.5088 | n/a | n/a | 6.0397 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1202 | 0.1209 | 0.1152 | 0.1287 | 0.0052 | 0.5088 | n/a | n/a | 6.0397 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0141 | 0.0143 | 0.0138 | 0.0154 | 0.0006 | 0.2772 | n/a | n/a | 0.9670 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0039 | 0.0044 | 0.0038 | 0.0062 | 0.0009 | 0.2772 | n/a | n/a | 0.9670 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0136 | 0.0144 | 0.0128 | 0.0165 | 0.0015 | 0.2772 | n/a | n/a | 0.9670 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0099 | 0.0099 | 0.0098 | 0.0099 | 0.0000 | 0.4004 | n/a | n/a | 1.3479 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0040 | 0.0039 | 0.0040 | 0.0000 | 0.4004 | n/a | n/a | 1.3479 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0133 | 0.0175 | 0.0128 | 0.0332 | 0.0079 | 0.4004 | n/a | n/a | 1.3479 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0095 | 0.0095 | 0.0095 | 0.0095 | 0.0000 | 0.3853 | n/a | n/a | 1.4744 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0036 | 0.0037 | 0.0000 | 0.3853 | n/a | n/a | 1.4744 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0140 | 0.0161 | 0.0131 | 0.0215 | 0.0034 | 0.3853 | n/a | n/a | 1.4744 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0135 | 0.0136 | 0.0134 | 0.0138 | 0.0001 | 0.4467 | n/a | n/a | 0.8842 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0060 | 0.0056 | 0.0046 | 0.0064 | 0.0008 | 0.4467 | n/a | n/a | 0.8842 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0119 | 0.0137 | 0.0112 | 0.0189 | 0.0029 | 0.4467 | n/a | n/a | 0.8842 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0451 | 0.0479 | 0.0436 | 0.0533 | 0.0044 | 0.5097 | n/a | n/a | 3.2142 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0230 | 0.0243 | 0.0228 | 0.0282 | 0.0021 | 0.5097 | n/a | n/a | 3.2142 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1449 | 0.1438 | 0.1240 | 0.1541 | 0.0109 | 0.5097 | n/a | n/a | 3.2142 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.9680 | n/a | n/a | 24.7693 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0019 | 0.0017 | 0.0021 | 0.0002 | 0.9680 | n/a | n/a | 24.7693 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0445 | 0.0444 | 0.0434 | 0.0453 | 0.0008 | 0.9680 | n/a | n/a | 24.7693 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0018 | 0.0000 | 2.6046 | n/a | n/a | 41.0256 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0046 | 0.0046 | 0.0044 | 0.0046 | 0.0001 | 2.6046 | n/a | n/a | 41.0256 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0731 | 0.0732 | 0.0718 | 0.0747 | 0.0009 | 2.6046 | n/a | n/a | 41.0256 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0018 | 0.0000 | 0.6867 | n/a | n/a | 17.2819 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.6867 | n/a | n/a | 17.2819 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0311 | 0.0305 | 0.0289 | 0.0313 | 0.0010 | 0.6867 | n/a | n/a | 17.2819 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0019 | 0.0000 | 0.6774 | n/a | n/a | 16.4073 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.6774 | n/a | n/a | 16.4073 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0306 | 0.0307 | 0.0302 | 0.0310 | 0.0003 | 0.6774 | n/a | n/a | 16.4073 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0019 | 0.0021 | 0.0018 | 0.0031 | 0.0005 | 0.6880 | n/a | n/a | 16.2846 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.6880 | n/a | n/a | 16.2846 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0308 | 0.0309 | 0.0298 | 0.0321 | 0.0007 | 0.6880 | n/a | n/a | 16.2846 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0019 | 0.0000 | 0.9511 | n/a | n/a | 25.2251 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.9511 | n/a | n/a | 25.2251 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0465 | 0.0466 | 0.0448 | 0.0482 | 0.0013 | 0.9511 | n/a | n/a | 25.2251 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0019 | 0.0000 | 2.4135 | n/a | n/a | 40.1222 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0046 | 0.0047 | 0.0045 | 0.0050 | 0.0002 | 2.4135 | n/a | n/a | 40.1222 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0757 | 0.0761 | 0.0736 | 0.0790 | 0.0019 | 2.4135 | n/a | n/a | 40.1222 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0019 | 0.0020 | 0.0019 | 0.0023 | 0.0002 | 0.7120 | n/a | n/a | 17.3624 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0013 | 0.0017 | 0.0012 | 0.0032 | 0.0008 | 0.7120 | n/a | n/a | 17.3624 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0325 | 0.0345 | 0.0299 | 0.0448 | 0.0053 | 0.7120 | n/a | n/a | 17.3624 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0022 | 0.0001 | 0.7092 | n/a | n/a | 17.5271 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0016 | 0.0002 | 0.7092 | n/a | n/a | 17.5271 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0325 | 0.0326 | 0.0305 | 0.0356 | 0.0017 | 0.7092 | n/a | n/a | 17.5271 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0019 | 0.0021 | 0.0018 | 0.0028 | 0.0004 | 0.7024 | n/a | n/a | 16.5909 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0015 | 0.0012 | 0.0020 | 0.0003 | 0.7024 | n/a | n/a | 16.5909 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0314 | 0.0316 | 0.0302 | 0.0326 | 0.0009 | 0.7024 | n/a | n/a | 16.5909 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.9615 | n/a | n/a | 24.8700 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0021 | 0.0002 | 0.9615 | n/a | n/a | 24.8700 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0449 | 0.0451 | 0.0440 | 0.0468 | 0.0009 | 0.9615 | n/a | n/a | 24.8700 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0022 | 0.0023 | 0.0021 | 0.0025 | 0.0002 | 2.0911 | n/a | n/a | 34.4554 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0047 | 0.0047 | 0.0044 | 0.0051 | 0.0003 | 2.0911 | n/a | n/a | 34.4554 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0767 | 0.0771 | 0.0738 | 0.0816 | 0.0029 | 2.0911 | n/a | n/a | 34.4554 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0021 | 0.0026 | 0.0002 | 0.5783 | n/a | n/a | 14.5695 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 0.5783 | n/a | n/a | 14.5695 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0333 | 0.0340 | 0.0303 | 0.0411 | 0.0038 | 0.5783 | n/a | n/a | 14.5695 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0021 | 0.0022 | 0.0020 | 0.0025 | 0.0002 | 0.6087 | n/a | n/a | 14.9879 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.6087 | n/a | n/a | 14.9879 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0318 | 0.0339 | 0.0302 | 0.0426 | 0.0044 | 0.6087 | n/a | n/a | 14.9879 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0023 | 0.0001 | 0.6062 | n/a | n/a | 14.4819 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0001 | 0.6062 | n/a | n/a | 14.4819 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0311 | 0.0314 | 0.0304 | 0.0325 | 0.0009 | 0.6062 | n/a | n/a | 14.4819 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0020 | 0.0020 | 0.0019 | 0.0021 | 0.0001 | 0.9175 | n/a | n/a | 23.1752 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0021 | 0.0001 | 0.9175 | n/a | n/a | 23.1752 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0453 | 0.0455 | 0.0449 | 0.0459 | 0.0004 | 0.9175 | n/a | n/a | 23.1752 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0028 | 0.0028 | 0.0028 | 0.0029 | 0.0000 | 1.6843 | n/a | n/a | 26.0389 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0048 | 0.0047 | 0.0046 | 0.0048 | 0.0001 | 1.6843 | n/a | n/a | 26.0389 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0741 | 0.0751 | 0.0732 | 0.0776 | 0.0019 | 1.6843 | n/a | n/a | 26.0389 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0030 | 0.0031 | 0.0030 | 0.0037 | 0.0003 | 0.5310 | n/a | n/a | 9.9452 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0016 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.5310 | n/a | n/a | 9.9452 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0301 | 0.0303 | 0.0295 | 0.0314 | 0.0007 | 0.5310 | n/a | n/a | 9.9452 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0028 | 0.0028 | 0.0026 | 0.0030 | 0.0001 | 0.4700 | n/a | n/a | 11.1078 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0018 | 0.0013 | 0.0028 | 0.0006 | 0.4700 | n/a | n/a | 11.1078 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0311 | 0.0316 | 0.0302 | 0.0330 | 0.0011 | 0.4700 | n/a | n/a | 11.1078 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0029 | 0.0031 | 0.0028 | 0.0034 | 0.0003 | 0.4794 | n/a | n/a | 10.4809 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0018 | 0.0002 | 0.4794 | n/a | n/a | 10.4809 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0308 | 0.0310 | 0.0300 | 0.0331 | 0.0011 | 0.4794 | n/a | n/a | 10.4809 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0025 | 0.0025 | 0.0025 | 0.0027 | 0.0001 | 1.0221 | n/a | n/a | 30.0745 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0025 | 0.0027 | 0.0024 | 0.0034 | 0.0004 | 1.0221 | n/a | n/a | 30.0745 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0742 | 0.0806 | 0.0623 | 0.1257 | 0.0232 | 1.0221 | n/a | n/a | 30.0745 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0104 | 0.0112 | 0.0095 | 0.0148 | 0.0020 | 0.7219 | n/a | n/a | 10.6132 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0075 | 0.0076 | 0.0073 | 0.0084 | 0.0004 | 0.7219 | n/a | n/a | 10.6132 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1106 | 0.1083 | 0.0949 | 0.1196 | 0.0090 | 0.7219 | n/a | n/a | 10.6132 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0146 | 0.0148 | 0.0145 | 0.0153 | 0.0003 | 0.2743 | n/a | n/a | 2.0910 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0040 | 0.0041 | 0.0040 | 0.0044 | 0.0002 | 0.2743 | n/a | n/a | 2.0910 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0306 | 0.0307 | 0.0302 | 0.0312 | 0.0004 | 0.2743 | n/a | n/a | 2.0910 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0085 | 0.0086 | 0.0083 | 0.0089 | 0.0002 | 0.3059 | n/a | n/a | 4.8684 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0000 | 0.3059 | n/a | n/a | 4.8684 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0416 | 0.0431 | 0.0304 | 0.0548 | 0.0080 | 0.3059 | n/a | n/a | 4.8684 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0091 | 0.0091 | 0.0087 | 0.0096 | 0.0003 | 0.3370 | n/a | n/a | 3.3997 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0031 | 0.0033 | 0.0030 | 0.0041 | 0.0004 | 0.3370 | n/a | n/a | 3.3997 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0310 | 0.0323 | 0.0305 | 0.0379 | 0.0028 | 0.3370 | n/a | n/a | 3.3997 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0049 | 0.0050 | 0.0048 | 0.0054 | 0.0002 | 0.7495 | n/a | n/a | 14.1286 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0037 | 0.0037 | 0.0036 | 0.0037 | 0.0000 | 0.7495 | n/a | n/a | 14.1286 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0692 | 0.0723 | 0.0627 | 0.0881 | 0.0086 | 0.7495 | n/a | n/a | 14.1286 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0355 | 0.0366 | 0.0324 | 0.0464 | 0.0051 | 0.3622 | n/a | n/a | 3.4336 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0128 | 0.0129 | 0.0124 | 0.0137 | 0.0005 | 0.3622 | n/a | n/a | 3.4336 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1217 | 0.1357 | 0.1155 | 0.2009 | 0.0328 | 0.3622 | n/a | n/a | 3.4336 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0638 | 0.0675 | 0.0605 | 0.0851 | 0.0090 | 0.1651 | n/a | n/a | 0.6191 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0105 | 0.0106 | 0.0103 | 0.0111 | 0.0003 | 0.1651 | n/a | n/a | 0.6191 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0395 | 0.0407 | 0.0350 | 0.0474 | 0.0049 | 0.1651 | n/a | n/a | 0.6191 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0287 | 0.0293 | 0.0286 | 0.0313 | 0.0010 | 0.2086 | n/a | n/a | 1.1539 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0060 | 0.0060 | 0.0059 | 0.0061 | 0.0001 | 0.2086 | n/a | n/a | 1.1539 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0331 | 0.0337 | 0.0329 | 0.0350 | 0.0009 | 0.2086 | n/a | n/a | 1.1539 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0310 | 0.0309 | 0.0303 | 0.0314 | 0.0004 | 0.2618 | n/a | n/a | 1.1345 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0081 | 0.0085 | 0.0073 | 0.0100 | 0.0011 | 0.2618 | n/a | n/a | 1.1345 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0352 | 0.0351 | 0.0340 | 0.0373 | 0.0012 | 0.2618 | n/a | n/a | 1.1345 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0140 | 0.0140 | 0.0137 | 0.0144 | 0.0002 | 0.6411 | n/a | n/a | 5.8649 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0090 | 0.0091 | 0.0089 | 0.0094 | 0.0002 | 0.6411 | n/a | n/a | 5.8649 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0822 | 0.0894 | 0.0738 | 0.1141 | 0.0150 | 0.6411 | n/a | n/a | 5.8649 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1222 | 0.1261 | 0.1197 | 0.1386 | 0.0069 | 0.2750 | n/a | n/a | 1.0749 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0336 | 0.0363 | 0.0330 | 0.0441 | 0.0042 | 0.2750 | n/a | n/a | 1.0749 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1313 | 0.1317 | 0.0955 | 0.1677 | 0.0234 | 0.2750 | n/a | n/a | 1.0749 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2878 | 0.2837 | 0.2528 | 0.3053 | 0.0184 | 0.1403 | n/a | n/a | 0.1379 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0404 | 0.0412 | 0.0391 | 0.0436 | 0.0018 | 0.1403 | n/a | n/a | 0.1379 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0397 | 0.0397 | 0.0376 | 0.0435 | 0.0021 | 0.1403 | n/a | n/a | 0.1379 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1177 | 0.1196 | 0.1099 | 0.1354 | 0.0085 | 0.1660 | n/a | n/a | 0.3495 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0195 | 0.0194 | 0.0190 | 0.0198 | 0.0003 | 0.1660 | n/a | n/a | 0.3495 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0411 | 0.0465 | 0.0399 | 0.0581 | 0.0075 | 0.1660 | n/a | n/a | 0.3495 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1221 | 0.1230 | 0.1141 | 0.1346 | 0.0077 | 0.2081 | n/a | n/a | 0.3756 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0254 | 0.0255 | 0.0248 | 0.0263 | 0.0005 | 0.2081 | n/a | n/a | 0.3756 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0459 | 0.0462 | 0.0420 | 0.0511 | 0.0035 | 0.2081 | n/a | n/a | 0.3756 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0026 | 0.0002 | 1.0357 | n/a | n/a | 5.6417 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0025 | 0.0022 | 0.0034 | 0.0005 | 1.0357 | n/a | n/a | 5.6417 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0126 | 0.0127 | 0.0123 | 0.0135 | 0.0004 | 1.0357 | n/a | n/a | 5.6417 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0024 | 0.0024 | 0.0022 | 0.0025 | 0.0001 | 3.6003 | n/a | n/a | 4.8772 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0086 | 0.0091 | 0.0081 | 0.0112 | 0.0011 | 3.6003 | n/a | n/a | 4.8772 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0116 | 0.0117 | 0.0113 | 0.0123 | 0.0003 | 3.6003 | n/a | n/a | 4.8772 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0028 | 0.0002 | 0.9897 | n/a | n/a | 5.5504 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.9897 | n/a | n/a | 5.5504 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0131 | 0.0138 | 0.0126 | 0.0166 | 0.0014 | 0.9897 | n/a | n/a | 5.5504 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0024 | 0.0024 | 0.0022 | 0.0025 | 0.0001 | 3.5618 | n/a | n/a | 5.2277 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0084 | 0.0091 | 0.0081 | 0.0116 | 0.0013 | 3.5618 | n/a | n/a | 5.2277 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0123 | 0.0127 | 0.0120 | 0.0135 | 0.0006 | 3.5618 | n/a | n/a | 5.2277 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0044 | 0.0040 | 0.0025 | 0.0052 | 0.0011 | 0.7703 | n/a | n/a | 2.9076 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0034 | 0.0033 | 0.0026 | 0.0041 | 0.0005 | 0.7703 | n/a | n/a | 2.9076 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0129 | 0.0129 | 0.0127 | 0.0130 | 0.0001 | 0.7703 | n/a | n/a | 2.9076 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0027 | 0.0001 | 3.2990 | n/a | n/a | 5.1964 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0083 | 0.0085 | 0.0081 | 0.0092 | 0.0004 | 3.2990 | n/a | n/a | 5.1964 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0131 | 0.0130 | 0.0124 | 0.0134 | 0.0004 | 3.2990 | n/a | n/a | 5.1964 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0035 | 0.0036 | 0.0034 | 0.0041 | 0.0002 | 0.8092 | n/a | n/a | 4.5721 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0029 | 0.0030 | 0.0027 | 0.0040 | 0.0005 | 0.8092 | n/a | n/a | 4.5721 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0162 | 0.0168 | 0.0156 | 0.0185 | 0.0011 | 0.8092 | n/a | n/a | 4.5721 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0034 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 2.5561 | n/a | n/a | 4.1558 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0087 | 0.0091 | 0.0086 | 0.0105 | 0.0007 | 2.5561 | n/a | n/a | 4.1558 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0141 | 0.0140 | 0.0135 | 0.0144 | 0.0003 | 2.5561 | n/a | n/a | 4.1558 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0088 | 0.0091 | 0.0085 | 0.0106 | 0.0008 | 0.3168 | n/a | n/a | 1.8078 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0028 | 0.0028 | 0.0027 | 0.0030 | 0.0001 | 0.3168 | n/a | n/a | 1.8078 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0159 | 0.0161 | 0.0152 | 0.0172 | 0.0009 | 0.3168 | n/a | n/a | 1.8078 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0091 | 0.0091 | 0.0090 | 0.0092 | 0.0001 | 0.9673 | n/a | n/a | 1.7578 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0088 | 0.0090 | 0.0087 | 0.0097 | 0.0004 | 0.9673 | n/a | n/a | 1.7578 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0160 | 0.0160 | 0.0137 | 0.0187 | 0.0020 | 0.9673 | n/a | n/a | 1.7578 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0309 | 0.0312 | 0.0288 | 0.0338 | 0.0019 | 0.1145 | n/a | n/a | 0.5730 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0035 | 0.0038 | 0.0035 | 0.0046 | 0.0004 | 0.1145 | n/a | n/a | 0.5730 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0177 | 0.0175 | 0.0142 | 0.0207 | 0.0025 | 0.1145 | n/a | n/a | 0.5730 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0293 | 0.0315 | 0.0288 | 0.0393 | 0.0040 | 0.3396 | n/a | n/a | 0.6152 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0100 | 0.0102 | 0.0096 | 0.0114 | 0.0007 | 0.3396 | n/a | n/a | 0.6152 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0180 | 0.0185 | 0.0165 | 0.0206 | 0.0015 | 0.3396 | n/a | n/a | 0.6152 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1113 | 0.1134 | 0.1099 | 0.1175 | 0.0034 | 0.0670 | n/a | n/a | 0.2643 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0075 | 0.0075 | 0.0071 | 0.0080 | 0.0003 | 0.0670 | n/a | n/a | 0.2643 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0294 | 0.0281 | 0.0213 | 0.0342 | 0.0058 | 0.0670 | n/a | n/a | 0.2643 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1112 | 0.1117 | 0.1084 | 0.1164 | 0.0026 | 0.1218 | n/a | n/a | 0.3001 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0135 | 0.0137 | 0.0134 | 0.0144 | 0.0004 | 0.1218 | n/a | n/a | 0.3001 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0334 | 0.0301 | 0.0200 | 0.0393 | 0.0081 | 0.1218 | n/a | n/a | 0.3001 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0034 | 0.0035 | 0.0028 | 0.0049 | 0.0008 | 0.8316 | n/a | n/a | 4.1658 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0028 | 0.0036 | 0.0027 | 0.0051 | 0.0011 | 0.8316 | n/a | n/a | 4.1658 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0140 | 0.0145 | 0.0132 | 0.0176 | 0.0016 | 0.8316 | n/a | n/a | 4.1658 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0029 | 0.0029 | 0.0027 | 0.0030 | 0.0001 | 3.6725 | n/a | n/a | 4.5661 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0105 | 0.0118 | 0.0101 | 0.0160 | 0.0022 | 3.6725 | n/a | n/a | 4.5661 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0130 | 0.0130 | 0.0126 | 0.0135 | 0.0003 | 3.6725 | n/a | n/a | 4.5661 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0057 | 0.0058 | 0.0055 | 0.0061 | 0.0002 | 0.7435 | n/a | n/a | 2.7077 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0043 | 0.0043 | 0.0042 | 0.0046 | 0.0002 | 0.7435 | n/a | n/a | 2.7077 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0155 | 0.0165 | 0.0146 | 0.0214 | 0.0024 | 0.7435 | n/a | n/a | 2.7077 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0041 | 0.0040 | 0.0038 | 0.0042 | 0.0002 | 2.9348 | n/a | n/a | 4.2028 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0121 | 0.0122 | 0.0119 | 0.0125 | 0.0002 | 2.9348 | n/a | n/a | 4.2028 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0173 | 0.0180 | 0.0160 | 0.0202 | 0.0017 | 2.9348 | n/a | n/a | 4.2028 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0226 | 0.0208 | 0.0166 | 0.0241 | 0.0033 | 0.3611 | n/a | n/a | 1.1116 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0082 | 0.0100 | 0.0081 | 0.0149 | 0.0027 | 0.3611 | n/a | n/a | 1.1116 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0251 | 0.0247 | 0.0189 | 0.0310 | 0.0040 | 0.3611 | n/a | n/a | 1.1116 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0070 | 0.0070 | 0.0068 | 0.0071 | 0.0001 | 2.1542 | n/a | n/a | 2.0980 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0151 | 0.0155 | 0.0150 | 0.0168 | 0.0007 | 2.1542 | n/a | n/a | 2.0980 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0147 | 0.0150 | 0.0141 | 0.0165 | 0.0008 | 2.1542 | n/a | n/a | 2.0980 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0661 | 0.0663 | 0.0651 | 0.0684 | 0.0011 | 0.3382 | n/a | n/a | 1.1770 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0224 | 0.0224 | 0.0219 | 0.0233 | 0.0005 | 0.3382 | n/a | n/a | 1.1770 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0778 | 0.0768 | 0.0714 | 0.0803 | 0.0030 | 0.3382 | n/a | n/a | 1.1770 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0187 | 0.0191 | 0.0186 | 0.0203 | 0.0007 | 1.2602 | n/a | n/a | 0.9101 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0235 | 0.0234 | 0.0224 | 0.0249 | 0.0009 | 1.2602 | n/a | n/a | 0.9101 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0170 | 0.0185 | 0.0167 | 0.0237 | 0.0027 | 1.2602 | n/a | n/a | 0.9101 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0035 | 0.0037 | 0.0035 | 0.0041 | 0.0002 | 0.5507 | n/a | n/a | 4.6039 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0020 | 0.0019 | 0.0020 | 0.0000 | 0.5507 | n/a | n/a | 4.6039 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0163 | 0.0151 | 0.0104 | 0.0186 | 0.0034 | 0.5507 | n/a | n/a | 4.6039 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0372 | 0.0382 | 0.0366 | 0.0423 | 0.0021 | 0.2886 | n/a | n/a | 0.5855 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0107 | 0.0108 | 0.0107 | 0.0108 | 0.0000 | 0.2886 | n/a | n/a | 0.5855 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0218 | 0.0237 | 0.0172 | 0.0357 | 0.0063 | 0.2886 | n/a | n/a | 0.5855 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.2734 | 0.2773 | 0.2725 | 0.2919 | 0.0074 | 1.0492 | n/a | n/a | 0.2997 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2868 | 0.3047 | 0.2833 | 0.3607 | 0.0294 | 1.0492 | n/a | n/a | 0.2997 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.0819 | 0.0803 | 0.0733 | 0.0876 | 0.0050 | 1.0492 | n/a | n/a | 0.2997 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.4787 | 2.4629 | 2.2982 | 2.6123 | 0.1160 | 0.1770 | n/a | n/a | 0.1071 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4388 | 0.4237 | 0.3726 | 0.4574 | 0.0332 | 0.1770 | n/a | n/a | 0.1071 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2653 | 0.2712 | 0.2498 | 0.2948 | 0.0166 | 0.1770 | n/a | n/a | 0.1071 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1530 | 0.1525 | 0.1490 | 0.1563 | 0.0027 | 8.7232 | n/a | n/a | 0.7925 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.3350 | 1.3297 | 1.2569 | 1.3741 | 0.0395 | 8.7232 | n/a | n/a | 0.7925 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1213 | 0.1150 | 0.0973 | 0.1320 | 0.0131 | 8.7232 | n/a | n/a | 0.7925 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.4653 | 7.4613 | 7.3839 | 7.5298 | 0.0623 | 8.5374 | n/a | n/a | 0.0466 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 63.7341 | 63.6304 | 61.9543 | 66.4063 | 1.6114 | 8.5374 | n/a | n/a | 0.0466 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3481 | 0.3450 | 0.3104 | 0.3849 | 0.0249 | 8.5374 | n/a | n/a | 0.0466 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0146 | 0.0149 | 0.0140 | 0.0166 | 0.0009 | 11.4002 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1664 | 0.1703 | 0.1614 | 0.1813 | 0.0082 | 11.4002 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0141 | 0.0144 | 0.0139 | 0.0154 | 0.0006 | 42.0542 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5914 | 0.5920 | 0.5737 | 0.6096 | 0.0122 | 42.0542 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5666 | 0.5651 | 0.5607 | 0.5702 | 0.0037 | 14.4024 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.1597 | 8.1688 | 8.0593 | 8.3401 | 0.1043 | 14.4024 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5628 | 0.5753 | 0.5457 | 0.6274 | 0.0302 | 49.9696 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 28.1207 | 27.9480 | 26.7410 | 29.1236 | 0.8853 | 49.9696 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0147 | 0.0152 | 0.0144 | 0.0165 | 0.0009 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0313 | 0.0389 | 0.0310 | 0.0658 | 0.0135 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1500 | 0.1526 | 0.1466 | 0.1659 | 0.0069 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3467 | 0.3503 | 0.3281 | 0.3691 | 0.0143 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.0346 | 1.0444 | 1.0287 | 1.0814 | 0.0194 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.4400 | 2.4641 | 2.4192 | 2.5139 | 0.0402 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.6824 | 1.6859 | 1.6085 | 1.7739 | 0.0542 | 0.0937 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1577 | 0.1820 | 0.1553 | 0.2460 | 0.0354 | 0.0937 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
