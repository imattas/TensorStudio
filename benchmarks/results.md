# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.3.7`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `26`
- TensorStudio losses versus NumPy: `77`
- TensorStudio wins versus JAX CPU dispatch: `83`
- TensorStudio losses versus JAX CPU dispatch: `15`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0029 | 0.0031 | 0.0027 | 0.0037 | 0.0004 | 0.2483 | n/a | n/a | 5.0556 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.2483 | n/a | n/a | 5.0556 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0148 | 0.0145 | 0.0127 | 0.0157 | 0.0011 | 0.2483 | n/a | n/a | 5.0556 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0035 | 0.0038 | 0.0032 | 0.0050 | 0.0007 | 0.2105 | n/a | n/a | 3.6316 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.2105 | n/a | n/a | 3.6316 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0127 | 0.0129 | 0.0124 | 0.0138 | 0.0005 | 0.2105 | n/a | n/a | 3.6316 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0032 | 0.0032 | 0.0026 | 0.0041 | 0.0005 | 0.2572 | n/a | n/a | 4.1833 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2572 | n/a | n/a | 4.1833 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0133 | 0.0135 | 0.0126 | 0.0155 | 0.0010 | 0.2572 | n/a | n/a | 4.1833 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0030 | 0.0030 | 0.0028 | 0.0033 | 0.0002 | 0.2704 | n/a | n/a | 3.5637 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0010 | 0.0001 | 0.2704 | n/a | n/a | 3.5637 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0106 | 0.0113 | 0.0102 | 0.0129 | 0.0012 | 0.2704 | n/a | n/a | 3.5637 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0106 | 0.0107 | 0.0103 | 0.0113 | 0.0004 | 0.5511 | n/a | n/a | 8.8193 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0059 | 0.0058 | 0.0054 | 0.0060 | 0.0002 | 0.5511 | n/a | n/a | 8.8193 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0937 | 0.0933 | 0.0901 | 0.0974 | 0.0028 | 0.5511 | n/a | n/a | 8.8193 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0030 | 0.0001 | 0.2400 | n/a | n/a | 4.3860 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0007 | 0.0000 | 0.2400 | n/a | n/a | 4.3860 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0118 | 0.0121 | 0.0115 | 0.0132 | 0.0006 | 0.2400 | n/a | n/a | 4.3860 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.2440 | n/a | n/a | 4.2640 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2440 | n/a | n/a | 4.2640 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0117 | 0.0118 | 0.0114 | 0.0123 | 0.0003 | 0.2440 | n/a | n/a | 4.2640 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0026 | 0.0027 | 0.0025 | 0.0031 | 0.0002 | 0.2588 | n/a | n/a | 4.5258 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.2588 | n/a | n/a | 4.5258 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0121 | 0.0118 | 0.0127 | 0.0003 | 0.2588 | n/a | n/a | 4.5258 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0028 | 0.0029 | 0.0027 | 0.0033 | 0.0002 | 0.2557 | n/a | n/a | 3.4071 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2557 | n/a | n/a | 3.4071 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0094 | 0.0099 | 0.0092 | 0.0117 | 0.0009 | 0.2557 | n/a | n/a | 3.4071 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0104 | 0.0105 | 0.0101 | 0.0109 | 0.0003 | 0.5539 | n/a | n/a | 12.0783 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0057 | 0.0058 | 0.0054 | 0.0062 | 0.0003 | 0.5539 | n/a | n/a | 12.0783 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1251 | 0.1492 | 0.0899 | 0.2546 | 0.0631 | 0.5539 | n/a | n/a | 12.0783 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0044 | 0.0047 | 0.0032 | 0.0076 | 0.0016 | 0.2083 | n/a | n/a | 3.5910 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0009 | 0.0011 | 0.0008 | 0.0019 | 0.0004 | 0.2083 | n/a | n/a | 3.5910 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0160 | 0.0180 | 0.0151 | 0.0265 | 0.0043 | 0.2083 | n/a | n/a | 3.5910 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0035 | 0.0036 | 0.0029 | 0.0045 | 0.0006 | 0.2324 | n/a | n/a | 4.8242 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0000 | 0.2324 | n/a | n/a | 4.8242 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0167 | 0.0184 | 0.0148 | 0.0266 | 0.0042 | 0.2324 | n/a | n/a | 4.8242 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0057 | 0.0055 | 0.0045 | 0.0063 | 0.0007 | 0.2381 | n/a | n/a | 4.4739 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0013 | 0.0015 | 0.0009 | 0.0020 | 0.0004 | 0.2381 | n/a | n/a | 4.4739 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0253 | 0.0269 | 0.0242 | 0.0324 | 0.0031 | 0.2381 | n/a | n/a | 4.4739 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0105 | 0.0106 | 0.0078 | 0.0140 | 0.0020 | 0.2642 | n/a | n/a | 1.8794 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0028 | 0.0027 | 0.0022 | 0.0033 | 0.0004 | 0.2642 | n/a | n/a | 1.8794 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0198 | 0.0206 | 0.0194 | 0.0223 | 0.0012 | 0.2642 | n/a | n/a | 1.8794 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0249 | 0.0234 | 0.0126 | 0.0343 | 0.0071 | 0.4267 | n/a | n/a | 3.7766 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0106 | 0.0111 | 0.0069 | 0.0166 | 0.0035 | 0.4267 | n/a | n/a | 3.7766 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0939 | 0.0964 | 0.0890 | 0.1103 | 0.0073 | 0.4267 | n/a | n/a | 3.7766 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0029 | 0.0029 | 0.0028 | 0.0030 | 0.0001 | 0.2340 | n/a | n/a | 4.1844 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.2340 | n/a | n/a | 4.1844 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0121 | 0.0123 | 0.0115 | 0.0134 | 0.0006 | 0.2340 | n/a | n/a | 4.1844 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0029 | 0.0031 | 0.0027 | 0.0036 | 0.0004 | 0.2794 | n/a | n/a | 4.2353 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0010 | 0.0001 | 0.2794 | n/a | n/a | 4.2353 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0124 | 0.0124 | 0.0116 | 0.0131 | 0.0005 | 0.2794 | n/a | n/a | 4.2353 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0029 | 0.0028 | 0.0027 | 0.0030 | 0.0001 | 0.2579 | n/a | n/a | 4.5884 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2579 | n/a | n/a | 4.5884 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0132 | 0.0132 | 0.0123 | 0.0144 | 0.0008 | 0.2579 | n/a | n/a | 4.5884 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0028 | 0.0030 | 0.0027 | 0.0032 | 0.0002 | 0.2558 | n/a | n/a | 3.4513 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.2558 | n/a | n/a | 3.4513 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0098 | 0.0096 | 0.0091 | 0.0101 | 0.0004 | 0.2558 | n/a | n/a | 3.4513 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0107 | 0.0107 | 0.0104 | 0.0109 | 0.0002 | 0.5505 | n/a | n/a | 9.8635 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0059 | 0.0058 | 0.0055 | 0.0061 | 0.0002 | 0.5505 | n/a | n/a | 9.8635 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.1060 | 0.1069 | 0.0911 | 0.1358 | 0.0160 | 0.5505 | n/a | n/a | 9.8635 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0032 | 0.0034 | 0.0032 | 0.0037 | 0.0002 | 0.3282 | n/a | n/a | 4.8340 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0011 | 0.0001 | 0.3282 | n/a | n/a | 4.8340 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0157 | 0.0162 | 0.0137 | 0.0190 | 0.0020 | 0.3282 | n/a | n/a | 4.8340 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0000 | 0.3802 | n/a | n/a | 5.2089 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0012 | 0.0012 | 0.0010 | 0.0017 | 0.0003 | 0.3802 | n/a | n/a | 5.2089 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0158 | 0.0156 | 0.0148 | 0.0164 | 0.0006 | 0.3802 | n/a | n/a | 5.2089 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0030 | 0.0036 | 0.0030 | 0.0053 | 0.0009 | 0.3637 | n/a | n/a | 4.7414 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0013 | 0.0010 | 0.0018 | 0.0003 | 0.3637 | n/a | n/a | 4.7414 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0143 | 0.0161 | 0.0136 | 0.0200 | 0.0027 | 0.3637 | n/a | n/a | 4.7414 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0035 | 0.0038 | 0.0032 | 0.0046 | 0.0006 | 0.3125 | n/a | n/a | 3.2175 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.3125 | n/a | n/a | 3.2175 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0113 | 0.0117 | 0.0106 | 0.0143 | 0.0014 | 0.3125 | n/a | n/a | 3.2175 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0126 | 0.0126 | 0.0119 | 0.0136 | 0.0006 | 0.5886 | n/a | n/a | 9.9242 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0074 | 0.0075 | 0.0071 | 0.0084 | 0.0004 | 0.5886 | n/a | n/a | 9.9242 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1246 | 0.1250 | 0.1113 | 0.1340 | 0.0079 | 0.5886 | n/a | n/a | 9.9242 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0052 | 0.0052 | 0.0049 | 0.0058 | 0.0003 | 0.5193 | n/a | n/a | 2.9399 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0027 | 0.0032 | 0.0018 | 0.0045 | 0.0010 | 0.5193 | n/a | n/a | 2.9399 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0154 | 0.0158 | 0.0126 | 0.0203 | 0.0026 | 0.5193 | n/a | n/a | 2.9399 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0049 | 0.0050 | 0.0047 | 0.0055 | 0.0003 | 0.4100 | n/a | n/a | 2.8824 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0020 | 0.0000 | 0.4100 | n/a | n/a | 2.8824 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0141 | 0.0148 | 0.0132 | 0.0166 | 0.0013 | 0.4100 | n/a | n/a | 2.8824 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0048 | 0.0049 | 0.0045 | 0.0055 | 0.0004 | 0.3099 | n/a | n/a | 3.4521 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.3099 | n/a | n/a | 3.4521 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0164 | 0.0170 | 0.0137 | 0.0200 | 0.0025 | 0.3099 | n/a | n/a | 3.4521 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0056 | 0.0064 | 0.0055 | 0.0093 | 0.0015 | 0.3108 | n/a | n/a | 2.1170 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.3108 | n/a | n/a | 2.1170 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0119 | 0.0121 | 0.0106 | 0.0140 | 0.0014 | 0.3108 | n/a | n/a | 2.1170 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0200 | 0.0202 | 0.0191 | 0.0222 | 0.0011 | 0.5173 | n/a | n/a | 6.6034 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0103 | 0.0103 | 0.0097 | 0.0112 | 0.0006 | 0.5173 | n/a | n/a | 6.6034 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1319 | 0.1314 | 0.1217 | 0.1439 | 0.0075 | 0.5173 | n/a | n/a | 6.6034 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0114 | 0.0115 | 0.0114 | 0.0117 | 0.0001 | 0.3101 | n/a | n/a | 1.1060 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0035 | 0.0036 | 0.0035 | 0.0036 | 0.0000 | 0.3101 | n/a | n/a | 1.1060 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0127 | 0.0131 | 0.0123 | 0.0149 | 0.0009 | 0.3101 | n/a | n/a | 1.1060 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0142 | 0.0138 | 0.0109 | 0.0170 | 0.0025 | 0.2420 | n/a | n/a | 0.9125 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0034 | 0.0035 | 0.0034 | 0.0035 | 0.0000 | 0.2420 | n/a | n/a | 0.9125 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0130 | 0.0132 | 0.0125 | 0.0142 | 0.0006 | 0.2420 | n/a | n/a | 0.9125 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0103 | 0.0102 | 0.0094 | 0.0112 | 0.0007 | 0.3637 | n/a | n/a | 1.6966 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0040 | 0.0037 | 0.0048 | 0.0004 | 0.3637 | n/a | n/a | 1.6966 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0174 | 0.0165 | 0.0149 | 0.0175 | 0.0011 | 0.3637 | n/a | n/a | 1.6966 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0135 | 0.0143 | 0.0134 | 0.0177 | 0.0017 | 0.3101 | n/a | n/a | 1.1570 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0043 | 0.0042 | 0.0047 | 0.0002 | 0.3101 | n/a | n/a | 1.1570 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0156 | 0.0163 | 0.0113 | 0.0266 | 0.0056 | 0.3101 | n/a | n/a | 1.1570 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0443 | 0.0464 | 0.0438 | 0.0551 | 0.0044 | 0.5219 | n/a | n/a | 3.1701 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0231 | 0.0230 | 0.0214 | 0.0243 | 0.0011 | 0.5219 | n/a | n/a | 3.1701 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1405 | 0.1379 | 0.1142 | 0.1561 | 0.0149 | 0.5219 | n/a | n/a | 3.1701 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0017 | 0.0018 | 0.0017 | 0.0022 | 0.0002 | 1.0598 | n/a | n/a | 25.6322 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 1.0598 | n/a | n/a | 25.6322 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0447 | 0.0446 | 0.0440 | 0.0450 | 0.0004 | 1.0598 | n/a | n/a | 25.6322 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 2.6471 | n/a | n/a | 40.8305 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0048 | 0.0047 | 0.0044 | 0.0051 | 0.0003 | 2.6471 | n/a | n/a | 40.8305 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0733 | 0.0764 | 0.0726 | 0.0889 | 0.0063 | 2.6471 | n/a | n/a | 40.8305 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0020 | 0.0023 | 0.0019 | 0.0030 | 0.0005 | 0.6884 | n/a | n/a | 24.4665 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.6884 | n/a | n/a | 24.4665 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0489 | 0.0461 | 0.0370 | 0.0503 | 0.0049 | 0.6884 | n/a | n/a | 24.4665 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0027 | 0.0027 | 0.0020 | 0.0038 | 0.0007 | 0.6419 | n/a | n/a | 11.9592 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0017 | 0.0018 | 0.0014 | 0.0023 | 0.0003 | 0.6419 | n/a | n/a | 11.9592 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0324 | 0.0388 | 0.0299 | 0.0590 | 0.0110 | 0.6419 | n/a | n/a | 11.9592 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0019 | 0.0021 | 0.0018 | 0.0028 | 0.0004 | 0.6621 | n/a | n/a | 15.4061 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0020 | 0.0003 | 0.6621 | n/a | n/a | 15.4061 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0299 | 0.0307 | 0.0295 | 0.0331 | 0.0013 | 0.6621 | n/a | n/a | 15.4061 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0017 | 0.0019 | 0.0017 | 0.0023 | 0.0002 | 1.0687 | n/a | n/a | 26.1632 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0019 | 0.0019 | 0.0016 | 0.0024 | 0.0003 | 1.0687 | n/a | n/a | 26.1632 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0458 | 0.0458 | 0.0445 | 0.0472 | 0.0009 | 1.0687 | n/a | n/a | 26.1632 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 2.6045 | n/a | n/a | 40.4515 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0048 | 0.0046 | 0.0043 | 0.0049 | 0.0002 | 2.6045 | n/a | n/a | 40.4515 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0743 | 0.0743 | 0.0728 | 0.0765 | 0.0013 | 2.6045 | n/a | n/a | 40.4515 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0020 | 0.0020 | 0.0018 | 0.0023 | 0.0002 | 0.6187 | n/a | n/a | 16.0871 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0001 | 0.6187 | n/a | n/a | 16.0871 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0322 | 0.0329 | 0.0301 | 0.0367 | 0.0024 | 0.6187 | n/a | n/a | 16.0871 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0021 | 0.0020 | 0.0018 | 0.0022 | 0.0002 | 0.7016 | n/a | n/a | 14.7280 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0015 | 0.0017 | 0.0012 | 0.0032 | 0.0007 | 0.7016 | n/a | n/a | 14.7280 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0312 | 0.0324 | 0.0304 | 0.0383 | 0.0030 | 0.7016 | n/a | n/a | 14.7280 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0023 | 0.0024 | 0.0019 | 0.0029 | 0.0004 | 0.6012 | n/a | n/a | 13.2687 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.6012 | n/a | n/a | 13.2687 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0310 | 0.0306 | 0.0295 | 0.0313 | 0.0007 | 0.6012 | n/a | n/a | 13.2687 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 0.9172 | n/a | n/a | 25.4022 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.9172 | n/a | n/a | 25.4022 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0472 | 0.0517 | 0.0445 | 0.0723 | 0.0104 | 0.9172 | n/a | n/a | 25.4022 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0020 | 0.0021 | 0.0019 | 0.0025 | 0.0002 | 2.3896 | n/a | n/a | 41.3862 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0048 | 0.0048 | 0.0045 | 0.0053 | 0.0003 | 2.3896 | n/a | n/a | 41.3862 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0827 | 0.0829 | 0.0797 | 0.0887 | 0.0033 | 2.3896 | n/a | n/a | 41.3862 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0023 | 0.0026 | 0.0022 | 0.0036 | 0.0006 | 0.6706 | n/a | n/a | 15.2575 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0015 | 0.0015 | 0.0012 | 0.0020 | 0.0002 | 0.6706 | n/a | n/a | 15.2575 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0346 | 0.0340 | 0.0321 | 0.0349 | 0.0011 | 0.6706 | n/a | n/a | 15.2575 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0022 | 0.0022 | 0.0020 | 0.0024 | 0.0001 | 0.6309 | n/a | n/a | 14.3512 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0012 | 0.0018 | 0.0002 | 0.6309 | n/a | n/a | 14.3512 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0313 | 0.0322 | 0.0309 | 0.0346 | 0.0014 | 0.6309 | n/a | n/a | 14.3512 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0023 | 0.0025 | 0.0022 | 0.0032 | 0.0004 | 0.5619 | n/a | n/a | 16.1874 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0015 | 0.0013 | 0.0022 | 0.0004 | 0.5619 | n/a | n/a | 16.1874 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0372 | 0.0406 | 0.0324 | 0.0598 | 0.0100 | 0.5619 | n/a | n/a | 16.1874 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0042 | 0.0053 | 0.0038 | 0.0079 | 0.0017 | 1.9487 | n/a | n/a | 35.1189 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0082 | 0.0078 | 0.0060 | 0.0095 | 0.0013 | 1.9487 | n/a | n/a | 35.1189 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.1479 | 0.1315 | 0.0604 | 0.2259 | 0.0602 | 1.9487 | n/a | n/a | 35.1189 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0041 | 0.0040 | 0.0032 | 0.0046 | 0.0005 | 1.6221 | n/a | n/a | 24.4102 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0067 | 0.0073 | 0.0050 | 0.0108 | 0.0020 | 1.6221 | n/a | n/a | 24.4102 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.1003 | 0.1024 | 0.0757 | 0.1334 | 0.0207 | 1.6221 | n/a | n/a | 24.4102 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0033 | 0.0036 | 0.0030 | 0.0044 | 0.0005 | 0.5332 | n/a | n/a | 9.3793 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0018 | 0.0018 | 0.0014 | 0.0024 | 0.0004 | 0.5332 | n/a | n/a | 9.3793 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0309 | 0.0310 | 0.0298 | 0.0324 | 0.0009 | 0.5332 | n/a | n/a | 9.3793 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0032 | 0.0032 | 0.0029 | 0.0035 | 0.0002 | 0.5627 | n/a | n/a | 9.8375 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0018 | 0.0017 | 0.0014 | 0.0021 | 0.0002 | 0.5627 | n/a | n/a | 9.8375 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0311 | 0.0319 | 0.0309 | 0.0353 | 0.0017 | 0.5627 | n/a | n/a | 9.8375 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0033 | 0.0031 | 0.0028 | 0.0034 | 0.0003 | 0.4439 | n/a | n/a | 10.6470 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0017 | 0.0014 | 0.0021 | 0.0003 | 0.4439 | n/a | n/a | 10.6470 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0347 | 0.0353 | 0.0321 | 0.0384 | 0.0025 | 0.4439 | n/a | n/a | 10.6470 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0032 | 0.0030 | 0.0025 | 0.0033 | 0.0003 | 0.7493 | n/a | n/a | 25.0690 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0024 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | 0.7493 | n/a | n/a | 25.0690 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0795 | 0.0804 | 0.0638 | 0.1037 | 0.0140 | 0.7493 | n/a | n/a | 25.0690 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0112 | 0.0111 | 0.0105 | 0.0118 | 0.0005 | 0.6876 | n/a | n/a | 9.4200 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0077 | 0.0079 | 0.0070 | 0.0095 | 0.0009 | 0.6876 | n/a | n/a | 9.4200 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1052 | 0.1023 | 0.0925 | 0.1100 | 0.0075 | 0.6876 | n/a | n/a | 9.4200 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0154 | 0.0154 | 0.0147 | 0.0164 | 0.0007 | 0.2868 | n/a | n/a | 2.4671 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0044 | 0.0048 | 0.0042 | 0.0059 | 0.0006 | 0.2868 | n/a | n/a | 2.4671 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0379 | 0.0403 | 0.0360 | 0.0516 | 0.0058 | 0.2868 | n/a | n/a | 2.4671 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0117 | 0.0116 | 0.0092 | 0.0138 | 0.0018 | 0.2335 | n/a | n/a | 3.0765 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0027 | 0.0029 | 0.0026 | 0.0038 | 0.0004 | 0.2335 | n/a | n/a | 3.0765 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0361 | 0.0366 | 0.0354 | 0.0389 | 0.0013 | 0.2335 | n/a | n/a | 3.0765 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0094 | 0.0100 | 0.0089 | 0.0116 | 0.0012 | 0.3133 | n/a | n/a | 3.7487 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0033 | 0.0029 | 0.0040 | 0.0005 | 0.3133 | n/a | n/a | 3.7487 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0352 | 0.0354 | 0.0335 | 0.0378 | 0.0014 | 0.3133 | n/a | n/a | 3.7487 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0058 | 0.0063 | 0.0048 | 0.0097 | 0.0018 | 0.6132 | n/a | n/a | 16.6035 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0035 | 0.0042 | 0.0035 | 0.0066 | 0.0012 | 0.6132 | n/a | n/a | 16.6035 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0959 | 0.0930 | 0.0680 | 0.1237 | 0.0193 | 0.6132 | n/a | n/a | 16.6035 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0438 | 0.0475 | 0.0377 | 0.0670 | 0.0101 | 0.2921 | n/a | n/a | 2.7032 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0128 | 0.0149 | 0.0125 | 0.0197 | 0.0029 | 0.2921 | n/a | n/a | 2.7032 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1183 | 0.1186 | 0.1088 | 0.1289 | 0.0067 | 0.2921 | n/a | n/a | 2.7032 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0690 | 0.0677 | 0.0620 | 0.0719 | 0.0041 | 0.2052 | n/a | n/a | 0.6159 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0142 | 0.0146 | 0.0128 | 0.0185 | 0.0021 | 0.2052 | n/a | n/a | 0.6159 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0425 | 0.0474 | 0.0359 | 0.0669 | 0.0106 | 0.2052 | n/a | n/a | 0.6159 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0360 | 0.0377 | 0.0326 | 0.0481 | 0.0055 | 0.3311 | n/a | n/a | 1.9875 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0119 | 0.0116 | 0.0092 | 0.0130 | 0.0015 | 0.3311 | n/a | n/a | 1.9875 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0715 | 0.0699 | 0.0610 | 0.0814 | 0.0071 | 0.3311 | n/a | n/a | 1.9875 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0553 | 0.0550 | 0.0492 | 0.0584 | 0.0032 | 0.2045 | n/a | n/a | 1.0165 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0113 | 0.0108 | 0.0095 | 0.0116 | 0.0008 | 0.2045 | n/a | n/a | 1.0165 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0562 | 0.0600 | 0.0522 | 0.0765 | 0.0086 | 0.2045 | n/a | n/a | 1.0165 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0178 | 0.0184 | 0.0162 | 0.0215 | 0.0020 | 0.8354 | n/a | n/a | 6.9673 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0149 | 0.0155 | 0.0142 | 0.0187 | 0.0017 | 0.8354 | n/a | n/a | 6.9673 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.1242 | 0.1346 | 0.1031 | 0.1943 | 0.0314 | 0.8354 | n/a | n/a | 6.9673 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1716 | 0.1769 | 0.1602 | 0.2034 | 0.0155 | 0.2935 | n/a | n/a | 1.5822 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0504 | 0.0506 | 0.0388 | 0.0620 | 0.0074 | 0.2935 | n/a | n/a | 1.5822 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.2716 | 0.3105 | 0.1627 | 0.4943 | 0.1220 | 0.2935 | n/a | n/a | 1.5822 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3699 | 0.3724 | 0.3392 | 0.4246 | 0.0307 | 0.1556 | n/a | n/a | 0.1413 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0576 | 0.0585 | 0.0522 | 0.0677 | 0.0051 | 0.1556 | n/a | n/a | 0.1413 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0523 | 0.0659 | 0.0460 | 0.0976 | 0.0213 | 0.1556 | n/a | n/a | 0.1413 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1814 | 0.1802 | 0.1586 | 0.1950 | 0.0119 | 0.1641 | n/a | n/a | 0.2952 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0298 | 0.0296 | 0.0242 | 0.0347 | 0.0034 | 0.1641 | n/a | n/a | 0.2952 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0536 | 0.0650 | 0.0488 | 0.1080 | 0.0220 | 0.1641 | n/a | n/a | 0.2952 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1823 | 0.2125 | 0.1527 | 0.3374 | 0.0695 | 0.2968 | n/a | n/a | 0.4955 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0541 | 0.0855 | 0.0415 | 0.1538 | 0.0455 | 0.2968 | n/a | n/a | 0.4955 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0904 | 0.0894 | 0.0758 | 0.1066 | 0.0106 | 0.2968 | n/a | n/a | 0.4955 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0043 | 0.0043 | 0.0035 | 0.0052 | 0.0006 | 1.5308 | n/a | n/a | 5.2771 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0066 | 0.0060 | 0.0044 | 0.0076 | 0.0013 | 1.5308 | n/a | n/a | 5.2771 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0226 | 0.0234 | 0.0190 | 0.0301 | 0.0036 | 1.5308 | n/a | n/a | 5.2771 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0037 | 0.0037 | 0.0030 | 0.0045 | 0.0006 | 3.9297 | n/a | n/a | 5.2920 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0145 | 0.0151 | 0.0130 | 0.0193 | 0.0023 | 3.9297 | n/a | n/a | 5.2920 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0195 | 0.0204 | 0.0187 | 0.0243 | 0.0020 | 3.9297 | n/a | n/a | 5.2920 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0041 | 0.0040 | 0.0033 | 0.0043 | 0.0003 | 1.5172 | n/a | n/a | 5.3929 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0062 | 0.0058 | 0.0034 | 0.0070 | 0.0013 | 1.5172 | n/a | n/a | 5.3929 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0221 | 0.0209 | 0.0160 | 0.0244 | 0.0032 | 1.5172 | n/a | n/a | 5.3929 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0027 | 0.0031 | 0.0026 | 0.0043 | 0.0007 | 3.5204 | n/a | n/a | 5.5943 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0094 | 0.0097 | 0.0090 | 0.0111 | 0.0007 | 3.5204 | n/a | n/a | 5.5943 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0149 | 0.0175 | 0.0125 | 0.0248 | 0.0052 | 3.5204 | n/a | n/a | 5.5943 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0036 | 0.0040 | 0.0029 | 0.0059 | 0.0010 | 0.9372 | n/a | n/a | 5.3243 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0034 | 0.0033 | 0.0028 | 0.0037 | 0.0003 | 0.9372 | n/a | n/a | 5.3243 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0191 | 0.0195 | 0.0156 | 0.0230 | 0.0030 | 0.9372 | n/a | n/a | 5.3243 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0039 | 0.0037 | 0.0031 | 0.0040 | 0.0004 | 3.8033 | n/a | n/a | 4.3099 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0147 | 0.0146 | 0.0116 | 0.0178 | 0.0022 | 3.8033 | n/a | n/a | 4.3099 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0167 | 0.0172 | 0.0161 | 0.0195 | 0.0012 | 3.8033 | n/a | n/a | 4.3099 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0053 | 0.0049 | 0.0034 | 0.0058 | 0.0009 | 0.5228 | n/a | n/a | 3.4423 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0028 | 0.0028 | 0.0025 | 0.0034 | 0.0003 | 0.5228 | n/a | n/a | 3.4423 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0183 | 0.0185 | 0.0169 | 0.0215 | 0.0016 | 0.5228 | n/a | n/a | 3.4423 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0044 | 0.0046 | 0.0041 | 0.0052 | 0.0004 | 2.2412 | n/a | n/a | 4.2568 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0099 | 0.0101 | 0.0092 | 0.0114 | 0.0008 | 2.2412 | n/a | n/a | 4.2568 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0188 | 0.0199 | 0.0160 | 0.0242 | 0.0035 | 2.2412 | n/a | n/a | 4.2568 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0106 | 0.0109 | 0.0097 | 0.0121 | 0.0009 | 0.2872 | n/a | n/a | 1.8719 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0030 | 0.0030 | 0.0030 | 0.0032 | 0.0001 | 0.2872 | n/a | n/a | 1.8719 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0198 | 0.0195 | 0.0169 | 0.0217 | 0.0017 | 0.2872 | n/a | n/a | 1.8719 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0110 | 0.0109 | 0.0095 | 0.0119 | 0.0009 | 1.0168 | n/a | n/a | 2.2932 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0112 | 0.0123 | 0.0095 | 0.0176 | 0.0029 | 1.0168 | n/a | n/a | 2.2932 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0253 | 0.0257 | 0.0243 | 0.0281 | 0.0013 | 1.0168 | n/a | n/a | 2.2932 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0378 | 0.0386 | 0.0336 | 0.0433 | 0.0033 | 0.1080 | n/a | n/a | 0.5081 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0041 | 0.0042 | 0.0039 | 0.0047 | 0.0003 | 0.1080 | n/a | n/a | 0.5081 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0192 | 0.0199 | 0.0175 | 0.0229 | 0.0020 | 0.1080 | n/a | n/a | 0.5081 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0329 | 0.0334 | 0.0315 | 0.0369 | 0.0019 | 0.3532 | n/a | n/a | 0.5376 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0116 | 0.0113 | 0.0097 | 0.0122 | 0.0009 | 0.3532 | n/a | n/a | 0.5376 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0177 | 0.0181 | 0.0139 | 0.0213 | 0.0028 | 0.3532 | n/a | n/a | 0.5376 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1109 | 0.1104 | 0.1090 | 0.1116 | 0.0011 | 0.0813 | n/a | n/a | 0.2064 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0090 | 0.0096 | 0.0074 | 0.0113 | 0.0015 | 0.0813 | n/a | n/a | 0.2064 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0229 | 0.0270 | 0.0217 | 0.0356 | 0.0059 | 0.0813 | n/a | n/a | 0.2064 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1221 | 0.1185 | 0.1079 | 0.1273 | 0.0072 | 0.1103 | n/a | n/a | 0.2290 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0135 | 0.0171 | 0.0132 | 0.0271 | 0.0054 | 0.1103 | n/a | n/a | 0.2290 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0280 | 0.0327 | 0.0214 | 0.0581 | 0.0135 | 0.1103 | n/a | n/a | 0.2290 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0029 | 0.0030 | 0.0028 | 0.0032 | 0.0001 | 1.3176 | n/a | n/a | 5.9706 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0038 | 0.0037 | 0.0030 | 0.0045 | 0.0006 | 1.3176 | n/a | n/a | 5.9706 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0173 | 0.0161 | 0.0133 | 0.0183 | 0.0019 | 1.3176 | n/a | n/a | 5.9706 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0028 | 0.0029 | 0.0028 | 0.0033 | 0.0002 | 4.6402 | n/a | n/a | 5.0383 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0130 | 0.0127 | 0.0101 | 0.0163 | 0.0023 | 4.6402 | n/a | n/a | 5.0383 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0141 | 0.0138 | 0.0122 | 0.0152 | 0.0011 | 4.6402 | n/a | n/a | 5.0383 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0059 | 0.0060 | 0.0056 | 0.0067 | 0.0004 | 0.7675 | n/a | n/a | 2.9099 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0045 | 0.0051 | 0.0043 | 0.0067 | 0.0009 | 0.7675 | n/a | n/a | 2.9099 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0172 | 0.0171 | 0.0161 | 0.0182 | 0.0008 | 0.7675 | n/a | n/a | 2.9099 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0042 | 0.0043 | 0.0039 | 0.0048 | 0.0003 | 3.1773 | n/a | n/a | 4.0010 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0134 | 0.0132 | 0.0124 | 0.0139 | 0.0005 | 3.1773 | n/a | n/a | 4.0010 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0169 | 0.0197 | 0.0149 | 0.0268 | 0.0045 | 3.1773 | n/a | n/a | 4.0010 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0208 | 0.0215 | 0.0189 | 0.0248 | 0.0020 | 0.4041 | n/a | n/a | 0.9336 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0084 | 0.0084 | 0.0082 | 0.0086 | 0.0001 | 0.4041 | n/a | n/a | 0.9336 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0194 | 0.0210 | 0.0186 | 0.0272 | 0.0032 | 0.4041 | n/a | n/a | 0.9336 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0085 | 0.0091 | 0.0072 | 0.0115 | 0.0015 | 1.7860 | n/a | n/a | 1.8410 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0152 | 0.0153 | 0.0151 | 0.0155 | 0.0001 | 1.7860 | n/a | n/a | 1.8410 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0157 | 0.0211 | 0.0131 | 0.0374 | 0.0092 | 1.7860 | n/a | n/a | 1.8410 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0688 | 0.0714 | 0.0681 | 0.0780 | 0.0038 | 0.3504 | n/a | n/a | 1.2160 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0241 | 0.0254 | 0.0223 | 0.0319 | 0.0034 | 0.3504 | n/a | n/a | 1.2160 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0837 | 0.0846 | 0.0721 | 0.1028 | 0.0116 | 0.3504 | n/a | n/a | 1.2160 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0187 | 0.0187 | 0.0186 | 0.0188 | 0.0001 | 1.2572 | n/a | n/a | 1.0091 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0235 | 0.0277 | 0.0226 | 0.0377 | 0.0061 | 1.2572 | n/a | n/a | 1.0091 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0189 | 0.0297 | 0.0163 | 0.0724 | 0.0215 | 1.2572 | n/a | n/a | 1.0091 | yes | n/a | n/a | yes | TensorStudio | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0036 | 0.0036 | 0.0036 | 0.0037 | 0.0000 | 0.5338 | n/a | n/a | 3.5545 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0024 | 0.0019 | 0.0040 | 0.0008 | 0.5338 | n/a | n/a | 3.5545 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0129 | 0.0128 | 0.0100 | 0.0172 | 0.0026 | 0.5338 | n/a | n/a | 3.5545 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0401 | 0.0424 | 0.0369 | 0.0512 | 0.0055 | 0.2782 | n/a | n/a | 0.4396 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0111 | 0.0112 | 0.0111 | 0.0113 | 0.0001 | 0.2782 | n/a | n/a | 0.4396 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0176 | 0.0192 | 0.0168 | 0.0261 | 0.0035 | 0.2782 | n/a | n/a | 0.4396 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.2692 | 0.2944 | 0.2684 | 0.3874 | 0.0466 | 1.3096 | n/a | n/a | 0.3515 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3525 | 0.3553 | 0.2913 | 0.4634 | 0.0633 | 1.3096 | n/a | n/a | 0.3515 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.0946 | 0.0958 | 0.0823 | 0.1132 | 0.0101 | 1.3096 | n/a | n/a | 0.3515 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.6949 | 2.7436 | 2.3453 | 3.6088 | 0.4609 | 0.1555 | n/a | n/a | 0.0815 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4190 | 0.4624 | 0.4026 | 0.5893 | 0.0706 | 0.1555 | n/a | n/a | 0.0815 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2197 | 0.2217 | 0.2089 | 0.2403 | 0.0107 | 0.1555 | n/a | n/a | 0.0815 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1604 | 0.1611 | 0.1531 | 0.1687 | 0.0055 | 8.0736 | n/a | n/a | 0.8209 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2951 | 1.3143 | 1.2560 | 1.4412 | 0.0679 | 8.0736 | n/a | n/a | 0.8209 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1317 | 0.1229 | 0.0933 | 0.1384 | 0.0163 | 8.0736 | n/a | n/a | 0.8209 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.9332 | 7.7887 | 7.3962 | 8.1010 | 0.3054 | 8.1769 | n/a | n/a | 0.0432 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 64.8690 | 65.5600 | 62.7133 | 69.4730 | 2.6665 | 8.1769 | n/a | n/a | 0.0432 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3426 | 0.4059 | 0.3348 | 0.5699 | 0.0915 | 8.1769 | n/a | n/a | 0.0432 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0174 | 0.0170 | 0.0148 | 0.0185 | 0.0013 | 10.6909 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1857 | 0.2703 | 0.1727 | 0.6269 | 0.1784 | 10.6909 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0244 | 0.0226 | 0.0185 | 0.0266 | 0.0033 | 49.0325 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 1.1980 | 1.1834 | 0.8345 | 1.6821 | 0.3019 | 49.0325 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6140 | 0.6442 | 0.5707 | 0.7286 | 0.0675 | 14.6929 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 9.0217 | 9.5496 | 8.7657 | 10.7115 | 0.8281 | 14.6929 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5855 | 0.5899 | 0.5663 | 0.6344 | 0.0236 | 48.3411 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 28.3061 | 28.7944 | 28.0195 | 30.4261 | 0.9190 | 48.3411 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0140 | 0.0143 | 0.0140 | 0.0153 | 0.0005 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0331 | 0.0337 | 0.0306 | 0.0363 | 0.0020 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1583 | 0.1554 | 0.1482 | 0.1591 | 0.0043 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3449 | 0.3494 | 0.3422 | 0.3680 | 0.0096 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1052 | 1.1079 | 1.0969 | 1.1252 | 0.0101 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7537 | 2.6677 | 2.4719 | 2.8692 | 0.1602 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.7335 | 1.7421 | 1.7137 | 1.7856 | 0.0248 | 0.0903 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1566 | 0.1573 | 0.1535 | 0.1634 | 0.0033 | 0.0903 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
