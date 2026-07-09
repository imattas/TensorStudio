# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `2.1.0`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `28`
- TensorStudio losses versus NumPy: `75`
- TensorStudio wins versus JAX CPU dispatch: `87`
- TensorStudio losses versus JAX CPU dispatch: `11`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0042 | 0.0044 | 0.0039 | 0.0052 | 0.0005 | 0.2683 | n/a | n/a | 4.5699 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0011 | 0.0013 | 0.0008 | 0.0019 | 0.0004 | 0.2683 | n/a | n/a | 4.5699 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0192 | 0.0178 | 0.0140 | 0.0210 | 0.0026 | 0.2683 | n/a | n/a | 4.5699 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0041 | 0.0045 | 0.0036 | 0.0060 | 0.0009 | 0.2428 | n/a | n/a | 4.2436 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0010 | 0.0010 | 0.0008 | 0.0011 | 0.0001 | 0.2428 | n/a | n/a | 4.2436 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0173 | 0.0171 | 0.0164 | 0.0178 | 0.0006 | 0.2428 | n/a | n/a | 4.2436 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0034 | 0.0035 | 0.0031 | 0.0040 | 0.0004 | 0.2509 | n/a | n/a | 5.3459 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0008 | 0.0011 | 0.0008 | 0.0019 | 0.0004 | 0.2509 | n/a | n/a | 5.3459 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0180 | 0.0192 | 0.0167 | 0.0249 | 0.0029 | 0.2509 | n/a | n/a | 5.3459 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0044 | 0.0043 | 0.0034 | 0.0049 | 0.0006 | 0.2011 | n/a | n/a | 3.3968 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0009 | 0.0010 | 0.0008 | 0.0014 | 0.0002 | 0.2011 | n/a | n/a | 3.3968 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0149 | 0.0151 | 0.0117 | 0.0172 | 0.0019 | 0.2011 | n/a | n/a | 3.3968 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0169 | 0.0174 | 0.0154 | 0.0206 | 0.0018 | 0.6933 | n/a | n/a | 15.6140 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0117 | 0.0109 | 0.0083 | 0.0130 | 0.0017 | 0.6933 | n/a | n/a | 15.6140 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.2632 | 0.2718 | 0.2040 | 0.3365 | 0.0459 | 0.6933 | n/a | n/a | 15.6140 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0044 | 0.0050 | 0.0042 | 0.0062 | 0.0008 | 0.4136 | n/a | n/a | 4.9205 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0018 | 0.0019 | 0.0012 | 0.0026 | 0.0006 | 0.4136 | n/a | n/a | 4.9205 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0218 | 0.0213 | 0.0184 | 0.0236 | 0.0018 | 0.4136 | n/a | n/a | 4.9205 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0060 | 0.0059 | 0.0050 | 0.0065 | 0.0005 | 0.3156 | n/a | n/a | 3.7940 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0019 | 0.0019 | 0.0014 | 0.0025 | 0.0003 | 0.3156 | n/a | n/a | 3.7940 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0227 | 0.0231 | 0.0205 | 0.0259 | 0.0020 | 0.3156 | n/a | n/a | 3.7940 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0057 | 0.0053 | 0.0037 | 0.0061 | 0.0009 | 0.2407 | n/a | n/a | 3.9712 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0010 | 0.0020 | 0.0004 | 0.2407 | n/a | n/a | 3.9712 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0225 | 0.0212 | 0.0181 | 0.0234 | 0.0021 | 0.2407 | n/a | n/a | 3.9712 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0038 | 0.0038 | 0.0036 | 0.0039 | 0.0001 | 0.2511 | n/a | n/a | 3.6048 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0014 | 0.0002 | 0.2511 | n/a | n/a | 3.6048 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0137 | 0.0134 | 0.0121 | 0.0144 | 0.0008 | 0.2511 | n/a | n/a | 3.6048 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0196 | 0.0196 | 0.0143 | 0.0249 | 0.0039 | 0.6454 | n/a | n/a | 6.3193 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0126 | 0.0120 | 0.0103 | 0.0132 | 0.0012 | 0.6454 | n/a | n/a | 6.3193 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1236 | 0.1227 | 0.1118 | 0.1383 | 0.0091 | 0.6454 | n/a | n/a | 6.3193 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0042 | 0.0040 | 0.0034 | 0.0048 | 0.0005 | 0.2070 | n/a | n/a | 3.7164 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0009 | 0.0009 | 0.0008 | 0.0012 | 0.0002 | 0.2070 | n/a | n/a | 3.7164 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0156 | 0.0159 | 0.0138 | 0.0177 | 0.0013 | 0.2070 | n/a | n/a | 3.7164 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0043 | 0.0048 | 0.0036 | 0.0063 | 0.0011 | 0.2092 | n/a | n/a | 3.5098 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0009 | 0.0011 | 0.0008 | 0.0016 | 0.0003 | 0.2092 | n/a | n/a | 3.5098 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0151 | 0.0154 | 0.0140 | 0.0183 | 0.0015 | 0.2092 | n/a | n/a | 3.5098 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0039 | 0.0039 | 0.0033 | 0.0042 | 0.0003 | 0.2554 | n/a | n/a | 4.1584 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0010 | 0.0010 | 0.0008 | 0.0012 | 0.0001 | 0.2554 | n/a | n/a | 4.1584 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0161 | 0.0167 | 0.0155 | 0.0199 | 0.0016 | 0.2554 | n/a | n/a | 4.1584 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0036 | 0.0038 | 0.0034 | 0.0045 | 0.0004 | 0.3958 | n/a | n/a | 3.7504 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0011 | 0.0017 | 0.0002 | 0.3958 | n/a | n/a | 3.7504 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0136 | 0.0148 | 0.0123 | 0.0188 | 0.0024 | 0.3958 | n/a | n/a | 3.7504 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0163 | 0.0167 | 0.0139 | 0.0220 | 0.0029 | 0.5662 | n/a | n/a | 9.5999 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0092 | 0.0094 | 0.0086 | 0.0101 | 0.0006 | 0.5662 | n/a | n/a | 9.5999 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.1564 | 0.1608 | 0.1113 | 0.1961 | 0.0297 | 0.5662 | n/a | n/a | 9.5999 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0044 | 0.0047 | 0.0035 | 0.0068 | 0.0012 | 0.2307 | n/a | n/a | 4.1726 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0010 | 0.0011 | 0.0009 | 0.0014 | 0.0002 | 0.2307 | n/a | n/a | 4.1726 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0184 | 0.0190 | 0.0178 | 0.0207 | 0.0013 | 0.2307 | n/a | n/a | 4.1726 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0041 | 0.0042 | 0.0039 | 0.0047 | 0.0003 | 0.2796 | n/a | n/a | 6.2822 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0012 | 0.0012 | 0.0009 | 0.0017 | 0.0003 | 0.2796 | n/a | n/a | 6.2822 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0260 | 0.0363 | 0.0219 | 0.0826 | 0.0232 | 0.2796 | n/a | n/a | 6.2822 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0090 | 0.0091 | 0.0062 | 0.0117 | 0.0021 | 0.2583 | n/a | n/a | 3.1815 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0023 | 0.0023 | 0.0019 | 0.0030 | 0.0004 | 0.2583 | n/a | n/a | 3.1815 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0287 | 0.0318 | 0.0273 | 0.0381 | 0.0045 | 0.2583 | n/a | n/a | 3.1815 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0085 | 0.0097 | 0.0076 | 0.0154 | 0.0029 | 0.2829 | n/a | n/a | 2.3804 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0024 | 0.0027 | 0.0017 | 0.0046 | 0.0010 | 0.2829 | n/a | n/a | 2.3804 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0202 | 0.0192 | 0.0157 | 0.0216 | 0.0023 | 0.2829 | n/a | n/a | 2.3804 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0249 | 0.0257 | 0.0236 | 0.0292 | 0.0020 | 0.6011 | n/a | n/a | 9.6527 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0150 | 0.0160 | 0.0115 | 0.0256 | 0.0052 | 0.6011 | n/a | n/a | 9.6527 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.2404 | 0.2423 | 0.2341 | 0.2536 | 0.0073 | 0.6011 | n/a | n/a | 9.6527 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0095 | 0.0091 | 0.0054 | 0.0125 | 0.0025 | 0.2757 | n/a | n/a | 3.1277 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0026 | 0.0028 | 0.0025 | 0.0034 | 0.0003 | 0.2757 | n/a | n/a | 3.1277 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0297 | 0.0297 | 0.0276 | 0.0327 | 0.0018 | 0.2757 | n/a | n/a | 3.1277 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0064 | 0.0075 | 0.0050 | 0.0115 | 0.0025 | 0.3792 | n/a | n/a | 5.2430 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0024 | 0.0024 | 0.0014 | 0.0035 | 0.0007 | 0.3792 | n/a | n/a | 5.2430 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0335 | 0.0402 | 0.0293 | 0.0720 | 0.0160 | 0.3792 | n/a | n/a | 5.2430 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0101 | 0.0109 | 0.0081 | 0.0139 | 0.0020 | 0.3108 | n/a | n/a | 4.1774 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0031 | 0.0034 | 0.0028 | 0.0042 | 0.0005 | 0.3108 | n/a | n/a | 4.1774 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0422 | 0.0506 | 0.0309 | 0.0738 | 0.0182 | 0.3108 | n/a | n/a | 4.1774 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0093 | 0.0102 | 0.0075 | 0.0128 | 0.0021 | 0.4070 | n/a | n/a | 3.2989 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0038 | 0.0039 | 0.0030 | 0.0048 | 0.0008 | 0.4070 | n/a | n/a | 3.2989 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0308 | 0.0318 | 0.0234 | 0.0411 | 0.0061 | 0.4070 | n/a | n/a | 3.2989 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0348 | 0.0367 | 0.0292 | 0.0457 | 0.0060 | 0.5749 | n/a | n/a | 7.6457 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0200 | 0.0202 | 0.0187 | 0.0223 | 0.0012 | 0.5749 | n/a | n/a | 7.6457 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.2662 | 0.2924 | 0.2554 | 0.3788 | 0.0466 | 0.5749 | n/a | n/a | 7.6457 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0099 | 0.0088 | 0.0064 | 0.0106 | 0.0017 | 0.4207 | n/a | n/a | 2.9097 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0041 | 0.0040 | 0.0023 | 0.0052 | 0.0009 | 0.4207 | n/a | n/a | 2.9097 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0287 | 0.0314 | 0.0237 | 0.0402 | 0.0060 | 0.4207 | n/a | n/a | 2.9097 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0108 | 0.0116 | 0.0089 | 0.0152 | 0.0022 | 0.3362 | n/a | n/a | 2.2856 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0036 | 0.0039 | 0.0026 | 0.0053 | 0.0011 | 0.3362 | n/a | n/a | 2.2856 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0248 | 0.0272 | 0.0223 | 0.0346 | 0.0044 | 0.3362 | n/a | n/a | 2.2856 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0122 | 0.0105 | 0.0062 | 0.0128 | 0.0025 | 0.1609 | n/a | n/a | 2.1949 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0020 | 0.0024 | 0.0020 | 0.0038 | 0.0007 | 0.1609 | n/a | n/a | 2.1949 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0269 | 0.0266 | 0.0245 | 0.0279 | 0.0012 | 0.1609 | n/a | n/a | 2.1949 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0099 | 0.0102 | 0.0090 | 0.0125 | 0.0012 | 0.3816 | n/a | n/a | 3.2956 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0038 | 0.0036 | 0.0030 | 0.0039 | 0.0004 | 0.3816 | n/a | n/a | 3.2956 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0328 | 0.0368 | 0.0186 | 0.0672 | 0.0163 | 0.3816 | n/a | n/a | 3.2956 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0350 | 0.0359 | 0.0310 | 0.0405 | 0.0036 | 0.6579 | n/a | n/a | 8.5068 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0230 | 0.0213 | 0.0165 | 0.0244 | 0.0030 | 0.6579 | n/a | n/a | 8.5068 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.2980 | 0.3032 | 0.2707 | 0.3496 | 0.0257 | 0.6579 | n/a | n/a | 8.5068 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0137 | 0.0148 | 0.0135 | 0.0179 | 0.0017 | 0.6395 | n/a | n/a | 1.9371 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0088 | 0.0092 | 0.0083 | 0.0106 | 0.0009 | 0.6395 | n/a | n/a | 1.9371 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0266 | 0.0250 | 0.0168 | 0.0285 | 0.0042 | 0.6395 | n/a | n/a | 1.9371 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0211 | 0.0190 | 0.0149 | 0.0222 | 0.0034 | 0.3726 | n/a | n/a | 2.4764 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0079 | 0.0080 | 0.0072 | 0.0094 | 0.0008 | 0.3726 | n/a | n/a | 2.4764 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0523 | 0.0606 | 0.0351 | 0.0967 | 0.0236 | 0.3726 | n/a | n/a | 2.4764 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0207 | 0.0190 | 0.0136 | 0.0223 | 0.0035 | 0.3558 | n/a | n/a | 1.4157 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0074 | 0.0076 | 0.0070 | 0.0083 | 0.0005 | 0.3558 | n/a | n/a | 1.4157 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0293 | 0.0336 | 0.0203 | 0.0620 | 0.0149 | 0.3558 | n/a | n/a | 1.4157 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0247 | 0.0253 | 0.0212 | 0.0321 | 0.0036 | 0.3204 | n/a | n/a | 0.7894 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0079 | 0.0080 | 0.0077 | 0.0084 | 0.0003 | 0.3204 | n/a | n/a | 0.7894 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0195 | 0.0327 | 0.0172 | 0.0625 | 0.0181 | 0.3204 | n/a | n/a | 0.7894 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1004 | 0.1001 | 0.0719 | 0.1151 | 0.0155 | 0.6800 | n/a | n/a | 2.6086 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0682 | 0.0782 | 0.0492 | 0.1289 | 0.0276 | 0.6800 | n/a | n/a | 2.6086 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.2618 | 0.2869 | 0.2476 | 0.3476 | 0.0392 | 0.6800 | n/a | n/a | 2.6086 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0046 | 0.0045 | 0.0028 | 0.0059 | 0.0010 | 0.9302 | n/a | n/a | 20.6152 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0042 | 0.0042 | 0.0031 | 0.0052 | 0.0008 | 0.9302 | n/a | n/a | 20.6152 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0940 | 0.0926 | 0.0791 | 0.0996 | 0.0070 | 0.9302 | n/a | n/a | 20.6152 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0028 | 0.0030 | 0.0024 | 0.0044 | 0.0007 | 3.2249 | n/a | n/a | 54.6596 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0090 | 0.0093 | 0.0075 | 0.0117 | 0.0014 | 3.2249 | n/a | n/a | 54.6596 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.1524 | 0.1524 | 0.1386 | 0.1702 | 0.0104 | 3.2249 | n/a | n/a | 54.6596 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0039 | 0.0040 | 0.0035 | 0.0044 | 0.0003 | 0.5670 | n/a | n/a | 18.2220 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0022 | 0.0027 | 0.0018 | 0.0042 | 0.0009 | 0.5670 | n/a | n/a | 18.2220 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0705 | 0.0666 | 0.0464 | 0.0894 | 0.0165 | 0.5670 | n/a | n/a | 18.2220 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0053 | 0.0053 | 0.0043 | 0.0067 | 0.0008 | 0.6436 | n/a | n/a | 13.6016 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0034 | 0.0032 | 0.0024 | 0.0036 | 0.0005 | 0.6436 | n/a | n/a | 13.6016 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0723 | 0.0783 | 0.0492 | 0.1220 | 0.0252 | 0.6436 | n/a | n/a | 13.6016 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0041 | 0.0040 | 0.0032 | 0.0046 | 0.0005 | 0.7511 | n/a | n/a | 13.0356 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0031 | 0.0030 | 0.0022 | 0.0040 | 0.0006 | 0.7511 | n/a | n/a | 13.0356 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0538 | 0.0596 | 0.0501 | 0.0747 | 0.0102 | 0.7511 | n/a | n/a | 13.0356 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0025 | 0.0026 | 0.0022 | 0.0033 | 0.0004 | 1.1394 | n/a | n/a | 38.1678 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0028 | 0.0028 | 0.0020 | 0.0037 | 0.0006 | 1.1394 | n/a | n/a | 38.1678 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0954 | 0.1006 | 0.0900 | 0.1219 | 0.0118 | 1.1394 | n/a | n/a | 38.1678 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0034 | 0.0038 | 0.0026 | 0.0056 | 0.0010 | 2.7167 | n/a | n/a | 64.5284 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0093 | 0.0103 | 0.0089 | 0.0126 | 0.0015 | 2.7167 | n/a | n/a | 64.5284 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.2217 | 0.2320 | 0.1821 | 0.2859 | 0.0358 | 2.7167 | n/a | n/a | 64.5284 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0056 | 0.0056 | 0.0046 | 0.0066 | 0.0007 | 0.8233 | n/a | n/a | 19.0335 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0046 | 0.0045 | 0.0037 | 0.0051 | 0.0004 | 0.8233 | n/a | n/a | 19.0335 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.1059 | 0.0989 | 0.0676 | 0.1198 | 0.0196 | 0.8233 | n/a | n/a | 19.0335 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0077 | 0.0073 | 0.0063 | 0.0078 | 0.0006 | 0.6262 | n/a | n/a | 13.8190 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0048 | 0.0047 | 0.0042 | 0.0050 | 0.0003 | 0.6262 | n/a | n/a | 13.8190 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.1059 | 0.1533 | 0.1043 | 0.3392 | 0.0930 | 0.6262 | n/a | n/a | 13.8190 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0072 | 0.0074 | 0.0066 | 0.0085 | 0.0007 | 0.7065 | n/a | n/a | 15.2608 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0051 | 0.0054 | 0.0042 | 0.0070 | 0.0009 | 0.7065 | n/a | n/a | 15.2608 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.1105 | 0.1501 | 0.1063 | 0.3080 | 0.0791 | 0.7065 | n/a | n/a | 15.2608 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0059 | 0.0060 | 0.0052 | 0.0074 | 0.0008 | 1.0766 | n/a | n/a | 28.4246 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0063 | 0.0064 | 0.0059 | 0.0072 | 0.0004 | 1.0766 | n/a | n/a | 28.4246 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.1670 | 0.1626 | 0.1310 | 0.1851 | 0.0178 | 1.0766 | n/a | n/a | 28.4246 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0066 | 0.0076 | 0.0061 | 0.0113 | 0.0020 | 2.2809 | n/a | n/a | 43.3741 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0150 | 0.0150 | 0.0147 | 0.0153 | 0.0002 | 2.2809 | n/a | n/a | 43.3741 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.2849 | 0.3049 | 0.2753 | 0.3566 | 0.0309 | 2.2809 | n/a | n/a | 43.3741 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0066 | 0.0066 | 0.0063 | 0.0073 | 0.0003 | 0.7050 | n/a | n/a | 16.1028 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0046 | 0.0050 | 0.0044 | 0.0062 | 0.0007 | 0.7050 | n/a | n/a | 16.1028 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.1057 | 0.1138 | 0.0943 | 0.1474 | 0.0202 | 0.7050 | n/a | n/a | 16.1028 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0073 | 0.0073 | 0.0060 | 0.0084 | 0.0009 | 1.0521 | n/a | n/a | 14.8610 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0077 | 0.0075 | 0.0045 | 0.0093 | 0.0016 | 1.0521 | n/a | n/a | 14.8610 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.1085 | 0.1146 | 0.1062 | 0.1292 | 0.0092 | 1.0521 | n/a | n/a | 14.8610 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | log | `(32,)` | TensorStudio | 0.0115 | 0.0110 | 0.0081 | 0.0144 | 0.0023 | 0.4772 | n/a | n/a | 8.0510 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0055 | 0.0055 | 0.0047 | 0.0063 | 0.0005 | 0.4772 | n/a | n/a | 8.0510 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0927 | 0.0968 | 0.0897 | 0.1079 | 0.0070 | 0.4772 | n/a | n/a | 8.0510 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0052 | 0.0050 | 0.0033 | 0.0062 | 0.0010 | 1.1522 | n/a | n/a | 27.8288 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0060 | 0.0063 | 0.0057 | 0.0069 | 0.0005 | 1.1522 | n/a | n/a | 27.8288 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.1455 | 0.1414 | 0.1254 | 0.1579 | 0.0117 | 1.1522 | n/a | n/a | 27.8288 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0073 | 0.0072 | 0.0057 | 0.0086 | 0.0009 | 1.8742 | n/a | n/a | 18.5030 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0136 | 0.0139 | 0.0122 | 0.0157 | 0.0015 | 1.8742 | n/a | n/a | 18.5030 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.1343 | 0.1392 | 0.1296 | 0.1622 | 0.0117 | 1.8742 | n/a | n/a | 18.5030 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0051 | 0.0052 | 0.0036 | 0.0070 | 0.0011 | 0.5533 | n/a | n/a | 14.5226 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0028 | 0.0031 | 0.0024 | 0.0043 | 0.0007 | 0.5533 | n/a | n/a | 14.5226 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0735 | 0.0715 | 0.0509 | 0.0870 | 0.0137 | 0.5533 | n/a | n/a | 14.5226 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0066 | 0.0067 | 0.0053 | 0.0077 | 0.0008 | 0.6042 | n/a | n/a | 9.5223 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0040 | 0.0036 | 0.0018 | 0.0043 | 0.0009 | 0.6042 | n/a | n/a | 9.5223 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0628 | 0.0651 | 0.0581 | 0.0738 | 0.0056 | 0.6042 | n/a | n/a | 9.5223 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0050 | 0.0053 | 0.0037 | 0.0072 | 0.0014 | 0.6853 | n/a | n/a | 10.8214 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0034 | 0.0034 | 0.0027 | 0.0040 | 0.0004 | 0.6853 | n/a | n/a | 10.8214 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0543 | 0.0554 | 0.0534 | 0.0605 | 0.0026 | 0.6853 | n/a | n/a | 10.8214 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0036 | 0.0043 | 0.0031 | 0.0067 | 0.0013 | 1.3136 | n/a | n/a | 32.5906 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0047 | 0.0045 | 0.0031 | 0.0061 | 0.0011 | 1.3136 | n/a | n/a | 32.5906 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.1173 | 0.1307 | 0.1043 | 0.1945 | 0.0327 | 1.3136 | n/a | n/a | 32.5906 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0290 | 0.0299 | 0.0233 | 0.0368 | 0.0045 | 0.7318 | n/a | n/a | 7.0585 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0212 | 0.0209 | 0.0174 | 0.0239 | 0.0021 | 0.7318 | n/a | n/a | 7.0585 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.2045 | 0.2214 | 0.2000 | 0.2830 | 0.0315 | 0.7318 | n/a | n/a | 7.0585 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0321 | 0.0316 | 0.0280 | 0.0349 | 0.0022 | 0.2623 | n/a | n/a | 2.1096 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0084 | 0.0082 | 0.0064 | 0.0102 | 0.0013 | 0.2623 | n/a | n/a | 2.1096 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0676 | 0.0667 | 0.0561 | 0.0787 | 0.0093 | 0.2623 | n/a | n/a | 2.1096 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0172 | 0.0172 | 0.0143 | 0.0203 | 0.0021 | 0.2205 | n/a | n/a | 3.8203 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0038 | 0.0040 | 0.0032 | 0.0054 | 0.0008 | 0.2205 | n/a | n/a | 3.8203 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0658 | 0.0680 | 0.0603 | 0.0860 | 0.0094 | 0.2205 | n/a | n/a | 3.8203 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0187 | 0.0177 | 0.0117 | 0.0204 | 0.0030 | 0.4025 | n/a | n/a | 3.7082 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0075 | 0.0079 | 0.0073 | 0.0087 | 0.0006 | 0.4025 | n/a | n/a | 3.7082 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0695 | 0.0693 | 0.0567 | 0.0766 | 0.0070 | 0.4025 | n/a | n/a | 3.7082 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0104 | 0.0099 | 0.0066 | 0.0131 | 0.0023 | 0.9333 | n/a | n/a | 13.9090 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0097 | 0.0139 | 0.0076 | 0.0219 | 0.0065 | 0.9333 | n/a | n/a | 13.9090 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.1443 | 0.1801 | 0.1295 | 0.3210 | 0.0712 | 0.9333 | n/a | n/a | 13.9090 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0700 | 0.0672 | 0.0549 | 0.0754 | 0.0069 | 0.4396 | n/a | n/a | 3.2894 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0308 | 0.0320 | 0.0282 | 0.0416 | 0.0049 | 0.4396 | n/a | n/a | 3.2894 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.2302 | 0.2448 | 0.2218 | 0.2987 | 0.0290 | 0.4396 | n/a | n/a | 3.2894 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.1273 | 0.1279 | 0.1151 | 0.1415 | 0.0088 | 0.1748 | n/a | n/a | 0.5167 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0222 | 0.0224 | 0.0199 | 0.0251 | 0.0018 | 0.1748 | n/a | n/a | 0.5167 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0658 | 0.0661 | 0.0613 | 0.0734 | 0.0044 | 0.1748 | n/a | n/a | 0.5167 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0558 | 0.0552 | 0.0418 | 0.0668 | 0.0084 | 0.1806 | n/a | n/a | 1.3699 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0101 | 0.0104 | 0.0075 | 0.0137 | 0.0027 | 0.1806 | n/a | n/a | 1.3699 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0764 | 0.0757 | 0.0666 | 0.0855 | 0.0072 | 0.1806 | n/a | n/a | 1.3699 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0597 | 0.0565 | 0.0492 | 0.0623 | 0.0052 | 0.2610 | n/a | n/a | 1.2042 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0156 | 0.0150 | 0.0121 | 0.0163 | 0.0015 | 0.2610 | n/a | n/a | 1.2042 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0719 | 0.0718 | 0.0665 | 0.0763 | 0.0035 | 0.2610 | n/a | n/a | 1.2042 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0277 | 0.0276 | 0.0251 | 0.0303 | 0.0017 | 0.3818 | n/a | n/a | 4.7711 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0106 | 0.0106 | 0.0104 | 0.0107 | 0.0001 | 0.3818 | n/a | n/a | 4.7711 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.1321 | 0.1377 | 0.1236 | 0.1598 | 0.0125 | 0.3818 | n/a | n/a | 4.7711 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.2365 | 0.2392 | 0.2026 | 0.2640 | 0.0219 | 0.2592 | n/a | n/a | 0.8622 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0613 | 0.0607 | 0.0532 | 0.0653 | 0.0043 | 0.2592 | n/a | n/a | 0.8622 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.2039 | 0.2068 | 0.1839 | 0.2515 | 0.0236 | 0.2592 | n/a | n/a | 0.8622 | no | n/a | n/a | no | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3660 | 0.3810 | 0.3548 | 0.4439 | 0.0324 | 0.1766 | n/a | n/a | 0.2115 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0646 | 0.0630 | 0.0496 | 0.0790 | 0.0102 | 0.1766 | n/a | n/a | 0.2115 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0774 | 0.0728 | 0.0529 | 0.0842 | 0.0110 | 0.1766 | n/a | n/a | 0.2115 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.2171 | 0.2151 | 0.1925 | 0.2332 | 0.0146 | 0.1772 | n/a | n/a | 0.3083 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0385 | 0.0378 | 0.0323 | 0.0400 | 0.0028 | 0.1772 | n/a | n/a | 0.3083 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0669 | 0.0645 | 0.0462 | 0.0895 | 0.0155 | 0.1772 | n/a | n/a | 0.3083 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1779 | 0.1803 | 0.1591 | 0.2067 | 0.0157 | 0.2317 | n/a | n/a | 0.4675 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0412 | 0.0396 | 0.0316 | 0.0488 | 0.0070 | 0.2317 | n/a | n/a | 0.4675 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0831 | 0.0789 | 0.0556 | 0.0949 | 0.0131 | 0.2317 | n/a | n/a | 0.4675 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0048 | 0.0048 | 0.0041 | 0.0053 | 0.0005 | 1.0688 | n/a | n/a | 5.4507 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0052 | 0.0050 | 0.0042 | 0.0059 | 0.0007 | 1.0688 | n/a | n/a | 5.4507 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0263 | 0.0316 | 0.0240 | 0.0440 | 0.0079 | 1.0688 | n/a | n/a | 5.4507 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0050 | 0.0048 | 0.0035 | 0.0057 | 0.0008 | 3.5178 | n/a | n/a | 4.0760 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0177 | 0.0184 | 0.0124 | 0.0259 | 0.0049 | 3.5178 | n/a | n/a | 4.0760 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0205 | 0.0199 | 0.0171 | 0.0226 | 0.0019 | 3.5178 | n/a | n/a | 4.0760 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0045 | 0.0043 | 0.0031 | 0.0049 | 0.0006 | 1.1808 | n/a | n/a | 4.9638 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0053 | 0.0049 | 0.0036 | 0.0059 | 0.0009 | 1.1808 | n/a | n/a | 4.9638 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0225 | 0.0226 | 0.0182 | 0.0261 | 0.0026 | 1.1808 | n/a | n/a | 4.9638 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0044 | 0.0046 | 0.0036 | 0.0056 | 0.0007 | 3.1767 | n/a | n/a | 4.7717 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0139 | 0.0147 | 0.0121 | 0.0174 | 0.0021 | 3.1767 | n/a | n/a | 4.7717 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0208 | 0.0201 | 0.0175 | 0.0227 | 0.0018 | 3.1767 | n/a | n/a | 4.7717 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0037 | 0.0038 | 0.0028 | 0.0049 | 0.0008 | 0.8518 | n/a | n/a | 5.4771 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0032 | 0.0033 | 0.0024 | 0.0048 | 0.0008 | 0.8518 | n/a | n/a | 5.4771 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0203 | 0.0208 | 0.0201 | 0.0230 | 0.0011 | 0.8518 | n/a | n/a | 5.4771 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0032 | 0.0035 | 0.0027 | 0.0047 | 0.0008 | 4.0458 | n/a | n/a | 6.7654 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0131 | 0.0141 | 0.0116 | 0.0167 | 0.0019 | 4.0458 | n/a | n/a | 6.7654 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0219 | 0.0213 | 0.0179 | 0.0259 | 0.0029 | 4.0458 | n/a | n/a | 6.7654 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0046 | 0.0047 | 0.0040 | 0.0060 | 0.0008 | 0.8020 | n/a | n/a | 4.5672 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0037 | 0.0045 | 0.0032 | 0.0063 | 0.0012 | 0.8020 | n/a | n/a | 4.5672 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0211 | 0.0209 | 0.0176 | 0.0255 | 0.0027 | 0.8020 | n/a | n/a | 4.5672 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0052 | 0.0049 | 0.0040 | 0.0052 | 0.0005 | 3.3396 | n/a | n/a | 4.6353 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0173 | 0.0154 | 0.0111 | 0.0188 | 0.0034 | 3.3396 | n/a | n/a | 4.6353 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0240 | 0.0230 | 0.0197 | 0.0261 | 0.0023 | 3.3396 | n/a | n/a | 4.6353 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0047 | 0.0052 | 0.0040 | 0.0068 | 0.0010 | 1.2410 | n/a | n/a | 6.4529 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0059 | 0.0059 | 0.0044 | 0.0071 | 0.0011 | 1.2410 | n/a | n/a | 6.4529 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0305 | 0.0306 | 0.0268 | 0.0360 | 0.0030 | 1.2410 | n/a | n/a | 6.4529 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0072 | 0.0072 | 0.0069 | 0.0074 | 0.0002 | 3.1618 | n/a | n/a | 4.0261 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0228 | 0.0207 | 0.0130 | 0.0300 | 0.0067 | 3.1618 | n/a | n/a | 4.0261 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0290 | 0.0293 | 0.0269 | 0.0324 | 0.0019 | 3.1618 | n/a | n/a | 4.0261 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0088 | 0.0088 | 0.0080 | 0.0096 | 0.0007 | 0.8699 | n/a | n/a | 2.7795 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0077 | 0.0067 | 0.0046 | 0.0085 | 0.0017 | 0.8699 | n/a | n/a | 2.7795 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0246 | 0.0264 | 0.0236 | 0.0310 | 0.0030 | 0.8699 | n/a | n/a | 2.7795 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0097 | 0.0097 | 0.0080 | 0.0113 | 0.0014 | 1.7610 | n/a | n/a | 2.7865 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0171 | 0.0181 | 0.0122 | 0.0232 | 0.0042 | 1.7610 | n/a | n/a | 2.7865 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0270 | 0.0275 | 0.0250 | 0.0301 | 0.0018 | 1.7610 | n/a | n/a | 2.7865 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0240 | 0.0243 | 0.0232 | 0.0256 | 0.0009 | 0.6271 | n/a | n/a | 1.7640 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0151 | 0.0149 | 0.0131 | 0.0168 | 0.0015 | 0.6271 | n/a | n/a | 1.7640 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0424 | 0.0425 | 0.0366 | 0.0500 | 0.0048 | 0.6271 | n/a | n/a | 1.7640 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0259 | 0.0263 | 0.0230 | 0.0303 | 0.0023 | 0.6786 | n/a | n/a | 1.4857 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0176 | 0.0186 | 0.0157 | 0.0219 | 0.0025 | 0.6786 | n/a | n/a | 1.4857 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0385 | 0.0381 | 0.0351 | 0.0400 | 0.0017 | 0.6786 | n/a | n/a | 1.4857 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0067 | 0.0063 | 0.0034 | 0.0080 | 0.0015 | 0.9826 | n/a | n/a | 3.2614 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0066 | 0.0057 | 0.0030 | 0.0067 | 0.0014 | 0.9826 | n/a | n/a | 3.2614 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0218 | 0.0212 | 0.0158 | 0.0265 | 0.0040 | 0.9826 | n/a | n/a | 3.2614 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0033 | 0.0039 | 0.0030 | 0.0056 | 0.0010 | 6.3463 | n/a | n/a | 5.7630 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0210 | 0.0215 | 0.0155 | 0.0258 | 0.0037 | 6.3463 | n/a | n/a | 5.7630 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0191 | 0.0199 | 0.0149 | 0.0246 | 0.0035 | 6.3463 | n/a | n/a | 5.7630 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0094 | 0.0088 | 0.0062 | 0.0108 | 0.0018 | 0.8765 | n/a | n/a | 2.8161 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0082 | 0.0085 | 0.0068 | 0.0106 | 0.0014 | 0.8765 | n/a | n/a | 2.8161 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0264 | 0.0263 | 0.0244 | 0.0292 | 0.0017 | 0.8765 | n/a | n/a | 2.8161 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0055 | 0.0063 | 0.0045 | 0.0088 | 0.0018 | 3.4624 | n/a | n/a | 4.9394 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0191 | 0.0181 | 0.0141 | 0.0193 | 0.0020 | 3.4624 | n/a | n/a | 4.9394 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0272 | 0.0273 | 0.0249 | 0.0319 | 0.0025 | 3.4624 | n/a | n/a | 4.9394 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0277 | 0.0270 | 0.0256 | 0.0279 | 0.0010 | 0.4594 | n/a | n/a | 1.1284 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0127 | 0.0142 | 0.0103 | 0.0198 | 0.0033 | 0.4594 | n/a | n/a | 1.1284 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0313 | 0.0321 | 0.0290 | 0.0364 | 0.0029 | 0.4594 | n/a | n/a | 1.1284 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0142 | 0.0136 | 0.0109 | 0.0152 | 0.0015 | 2.2562 | n/a | n/a | 2.0096 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0321 | 0.0300 | 0.0214 | 0.0327 | 0.0043 | 2.2562 | n/a | n/a | 2.0096 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0286 | 0.0279 | 0.0205 | 0.0340 | 0.0049 | 2.2562 | n/a | n/a | 2.0096 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0849 | 0.0841 | 0.0750 | 0.0900 | 0.0051 | 0.3717 | n/a | n/a | 1.0878 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0316 | 0.0333 | 0.0250 | 0.0438 | 0.0075 | 0.3717 | n/a | n/a | 1.0878 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0923 | 0.0932 | 0.0894 | 0.0970 | 0.0028 | 0.3717 | n/a | n/a | 1.0878 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0342 | 0.0321 | 0.0220 | 0.0380 | 0.0058 | 1.4194 | n/a | n/a | 1.1627 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0486 | 0.0484 | 0.0448 | 0.0529 | 0.0032 | 1.4194 | n/a | n/a | 1.1627 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0398 | 0.0467 | 0.0359 | 0.0657 | 0.0113 | 1.4194 | n/a | n/a | 1.1627 | yes | n/a | n/a | yes | TensorStudio | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0083 | 0.0079 | 0.0053 | 0.0100 | 0.0016 | 0.3342 | n/a | n/a | 2.0760 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0028 | 0.0029 | 0.0024 | 0.0035 | 0.0004 | 0.3342 | n/a | n/a | 2.0760 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0172 | 0.0182 | 0.0133 | 0.0231 | 0.0041 | 0.3342 | n/a | n/a | 2.0760 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1300 | 0.1507 | 0.1156 | 0.2102 | 0.0355 | 0.1555 | n/a | n/a | 0.3839 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0202 | 0.0199 | 0.0162 | 0.0230 | 0.0025 | 0.1555 | n/a | n/a | 0.3839 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0499 | 0.0501 | 0.0417 | 0.0568 | 0.0054 | 0.1555 | n/a | n/a | 0.3839 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 1.3680 | 1.3082 | 0.9559 | 1.4660 | 0.1865 | 0.1762 | n/a | n/a | 0.0786 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2410 | 0.3071 | 0.2305 | 0.5836 | 0.1384 | 0.1762 | n/a | n/a | 0.0786 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1075 | 0.1115 | 0.1011 | 0.1224 | 0.0084 | 0.1762 | n/a | n/a | 0.0786 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 10.3296 | 10.1065 | 8.2585 | 12.6126 | 1.6763 | 0.0404 | n/a | n/a | 0.0220 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4178 | 0.4220 | 0.3844 | 0.4499 | 0.0239 | 0.0404 | n/a | n/a | 0.0220 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2270 | 0.2342 | 0.2169 | 0.2716 | 0.0192 | 0.0404 | n/a | n/a | 0.0220 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2238 | 0.2189 | 0.2002 | 0.2315 | 0.0110 | 11.0568 | n/a | n/a | 0.6666 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 2.4749 | 2.5680 | 2.2748 | 3.0959 | 0.2892 | 11.0568 | n/a | n/a | 0.6666 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1492 | 0.1551 | 0.1460 | 0.1823 | 0.0137 | 11.0568 | n/a | n/a | 0.6666 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 9.6670 | 9.8401 | 9.3121 | 10.3733 | 0.3989 | 13.8558 | n/a | n/a | 0.0513 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 133.9443 | 138.7675 | 132.3712 | 150.5267 | 7.0718 | 13.8558 | n/a | n/a | 0.0513 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.4963 | 0.5564 | 0.4543 | 0.8018 | 0.1253 | 13.8558 | n/a | n/a | 0.0513 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0275 | 0.0274 | 0.0245 | 0.0294 | 0.0018 | 12.1065 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.3333 | 0.3229 | 0.2733 | 0.3495 | 0.0266 | 12.1065 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0222 | 0.0247 | 0.0192 | 0.0317 | 0.0050 | 53.4440 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 1.1843 | 1.1949 | 1.1364 | 1.2722 | 0.0510 | 53.4440 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7143 | 0.7068 | 0.6684 | 0.7312 | 0.0239 | 24.0510 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 17.1800 | 17.7729 | 16.0675 | 20.1212 | 1.5348 | 24.0510 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.8210 | 0.8297 | 0.7961 | 0.8997 | 0.0370 | 67.5064 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 55.4257 | 54.7311 | 49.2860 | 58.8530 | 3.8803 | 67.5064 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0444 | 0.0478 | 0.0413 | 0.0634 | 0.0082 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0717 | 0.0801 | 0.0635 | 0.0999 | 0.0151 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.3745 | 0.4223 | 0.3333 | 0.6099 | 0.1020 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.9699 | 1.0795 | 0.5871 | 1.7719 | 0.3988 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.6731 | 1.8364 | 1.5392 | 2.7282 | 0.4497 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 3.4431 | 3.3689 | 3.1443 | 3.5273 | 0.1623 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 2.2542 | 2.2780 | 2.0629 | 2.6926 | 0.2259 | 0.0915 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.2062 | 0.2226 | 0.2011 | 0.2668 | 0.0254 | 0.0915 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
