# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.9.0`
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
- TensorStudio wins versus JAX CPU dispatch: `55`
- TensorStudio losses versus JAX CPU dispatch: `43`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0238 | 0.0240 | 0.0192 | 0.0296 | 0.0034 | 0.0368 | n/a | n/a | 0.7944 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0009 | 0.0009 | 0.0007 | 0.0010 | 0.0001 | 0.0368 | n/a | n/a | 0.7944 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0189 | 0.0190 | 0.0164 | 0.0212 | 0.0017 | 0.0368 | n/a | n/a | 0.7944 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0242 | 0.0244 | 0.0220 | 0.0271 | 0.0018 | 0.0707 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0017 | 0.0016 | 0.0010 | 0.0020 | 0.0004 | 0.0707 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0215 | 0.0219 | 0.0202 | 0.0241 | 0.0015 | 0.0707 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0259 | 0.0266 | 0.0234 | 0.0315 | 0.0032 | 0.0544 | n/a | n/a | 0.7035 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0014 | 0.0021 | 0.0009 | 0.0050 | 0.0015 | 0.0544 | n/a | n/a | 0.7035 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0182 | 0.0186 | 0.0153 | 0.0229 | 0.0030 | 0.0544 | n/a | n/a | 0.7035 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0211 | 0.0206 | 0.0189 | 0.0216 | 0.0010 | 0.0403 | n/a | n/a | 0.6099 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0010 | 0.0008 | 0.0014 | 0.0002 | 0.0403 | n/a | n/a | 0.6099 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0128 | 0.0137 | 0.0120 | 0.0159 | 0.0015 | 0.0403 | n/a | n/a | 0.6099 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.1114 | 0.1141 | 0.1063 | 0.1285 | 0.0078 | 0.0530 | n/a | n/a | 1.0910 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0059 | 0.0060 | 0.0058 | 0.0063 | 0.0002 | 0.0530 | n/a | n/a | 1.0910 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.1216 | 0.1322 | 0.1024 | 0.1849 | 0.0293 | 0.0530 | n/a | n/a | 1.0910 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0200 | 0.0202 | 0.0170 | 0.0247 | 0.0029 | 0.0373 | n/a | n/a | 0.7387 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0373 | n/a | n/a | 0.7387 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0148 | 0.0159 | 0.0145 | 0.0180 | 0.0016 | 0.0373 | n/a | n/a | 0.7387 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0202 | 0.0200 | 0.0171 | 0.0220 | 0.0016 | 0.1110 | n/a | n/a | 0.6474 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0022 | 0.0019 | 0.0007 | 0.0023 | 0.0006 | 0.1110 | n/a | n/a | 0.6474 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0131 | 0.0133 | 0.0124 | 0.0152 | 0.0010 | 0.1110 | n/a | n/a | 0.6474 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0168 | 0.0167 | 0.0158 | 0.0176 | 0.0006 | 0.0449 | n/a | n/a | 0.8475 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0449 | n/a | n/a | 0.8475 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0142 | 0.0158 | 0.0130 | 0.0232 | 0.0038 | 0.0449 | n/a | n/a | 0.8475 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0176 | 0.0177 | 0.0162 | 0.0197 | 0.0012 | 0.0428 | n/a | n/a | 0.5693 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0428 | n/a | n/a | 0.5693 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0100 | 0.0104 | 0.0097 | 0.0120 | 0.0009 | 0.0428 | n/a | n/a | 0.5693 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0884 | 0.0907 | 0.0878 | 0.0985 | 0.0041 | 0.0654 | n/a | n/a | 1.1728 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0058 | 0.0066 | 0.0056 | 0.0099 | 0.0017 | 0.0654 | n/a | n/a | 1.1728 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1036 | 0.1075 | 0.1022 | 0.1187 | 0.0063 | 0.0654 | n/a | n/a | 1.1728 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0221 | 0.0271 | 0.0193 | 0.0429 | 0.0086 | 0.0363 | n/a | n/a | 0.6218 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0009 | 0.0000 | 0.0363 | n/a | n/a | 0.6218 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0137 | 0.0196 | 0.0132 | 0.0428 | 0.0117 | 0.0363 | n/a | n/a | 0.6218 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0198 | 0.0211 | 0.0167 | 0.0304 | 0.0049 | 0.0343 | n/a | n/a | 0.6625 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0343 | n/a | n/a | 0.6625 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0131 | 0.0137 | 0.0127 | 0.0167 | 0.0015 | 0.0343 | n/a | n/a | 0.6625 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0197 | 0.0200 | 0.0171 | 0.0227 | 0.0019 | 0.0468 | n/a | n/a | 0.6888 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0009 | 0.0009 | 0.0008 | 0.0009 | 0.0001 | 0.0468 | n/a | n/a | 0.6888 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0136 | 0.0139 | 0.0129 | 0.0158 | 0.0010 | 0.0468 | n/a | n/a | 0.6888 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0177 | 0.0177 | 0.0166 | 0.0188 | 0.0008 | 0.0411 | n/a | n/a | 0.6117 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0411 | n/a | n/a | 0.6117 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0108 | 0.0114 | 0.0102 | 0.0130 | 0.0013 | 0.0411 | n/a | n/a | 0.6117 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0896 | 0.0922 | 0.0872 | 0.0996 | 0.0047 | 0.0722 | n/a | n/a | 1.1086 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0065 | 0.0072 | 0.0060 | 0.0093 | 0.0012 | 0.0722 | n/a | n/a | 1.1086 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0993 | 0.0985 | 0.0932 | 0.1027 | 0.0031 | 0.0722 | n/a | n/a | 1.1086 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0167 | 0.0167 | 0.0163 | 0.0173 | 0.0004 | 0.0410 | n/a | n/a | 0.7518 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0410 | n/a | n/a | 0.7518 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0125 | 0.0131 | 0.0123 | 0.0153 | 0.0011 | 0.0410 | n/a | n/a | 0.7518 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0161 | 0.0169 | 0.0160 | 0.0195 | 0.0013 | 0.0447 | n/a | n/a | 0.8452 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0447 | n/a | n/a | 0.8452 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0136 | 0.0136 | 0.0127 | 0.0144 | 0.0006 | 0.0447 | n/a | n/a | 0.8452 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0172 | 0.0172 | 0.0164 | 0.0182 | 0.0008 | 0.0473 | n/a | n/a | 0.7641 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0473 | n/a | n/a | 0.7641 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0131 | 0.0132 | 0.0127 | 0.0136 | 0.0003 | 0.0473 | n/a | n/a | 0.7641 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0163 | 0.0166 | 0.0162 | 0.0181 | 0.0007 | 0.0446 | n/a | n/a | 0.8087 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0446 | n/a | n/a | 0.8087 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0132 | 0.0134 | 0.0097 | 0.0204 | 0.0038 | 0.0446 | n/a | n/a | 0.8087 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0957 | 0.0956 | 0.0886 | 0.1039 | 0.0050 | 0.0596 | n/a | n/a | 1.0768 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0057 | 0.0057 | 0.0055 | 0.0058 | 0.0001 | 0.0596 | n/a | n/a | 1.0768 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.1030 | 0.1044 | 0.0987 | 0.1135 | 0.0049 | 0.0596 | n/a | n/a | 1.0768 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0183 | 0.0182 | 0.0174 | 0.0187 | 0.0004 | 0.0538 | n/a | n/a | 1.1097 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0009 | 0.0012 | 0.0001 | 0.0538 | n/a | n/a | 1.1097 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0203 | 0.0192 | 0.0139 | 0.0245 | 0.0036 | 0.0538 | n/a | n/a | 1.1097 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0174 | 0.0175 | 0.0166 | 0.0184 | 0.0007 | 0.0579 | n/a | n/a | 1.1837 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0579 | n/a | n/a | 1.1837 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0207 | 0.0205 | 0.0182 | 0.0223 | 0.0016 | 0.0579 | n/a | n/a | 1.1837 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0194 | 0.0202 | 0.0179 | 0.0229 | 0.0019 | 0.0575 | n/a | n/a | 0.8880 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0013 | 0.0011 | 0.0020 | 0.0004 | 0.0575 | n/a | n/a | 0.8880 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0172 | 0.0173 | 0.0157 | 0.0190 | 0.0011 | 0.0575 | n/a | n/a | 0.8880 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0168 | 0.0177 | 0.0163 | 0.0216 | 0.0020 | 0.0642 | n/a | n/a | 0.8266 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.0642 | n/a | n/a | 0.8266 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0139 | 0.0141 | 0.0121 | 0.0163 | 0.0017 | 0.0642 | n/a | n/a | 0.8266 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0917 | 0.0923 | 0.0869 | 0.0981 | 0.0038 | 0.0793 | n/a | n/a | 1.3938 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0073 | 0.0074 | 0.0070 | 0.0078 | 0.0003 | 0.0793 | n/a | n/a | 1.3938 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1278 | 0.1304 | 0.1194 | 0.1483 | 0.0110 | 0.0793 | n/a | n/a | 1.3938 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0189 | 0.0189 | 0.0179 | 0.0200 | 0.0007 | 0.0760 | n/a | n/a | 0.9627 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.0760 | n/a | n/a | 0.9627 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0182 | 0.0198 | 0.0173 | 0.0252 | 0.0029 | 0.0760 | n/a | n/a | 0.9627 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0186 | 0.0192 | 0.0173 | 0.0224 | 0.0018 | 0.0798 | n/a | n/a | 0.9968 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0021 | 0.0002 | 0.0798 | n/a | n/a | 0.9968 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0185 | 0.0196 | 0.0168 | 0.0235 | 0.0024 | 0.0798 | n/a | n/a | 0.9968 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0193 | 0.0195 | 0.0187 | 0.0203 | 0.0006 | 0.1402 | n/a | n/a | 1.0963 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0027 | 0.0024 | 0.0016 | 0.0028 | 0.0004 | 0.1402 | n/a | n/a | 1.0963 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0212 | 0.0213 | 0.0182 | 0.0246 | 0.0024 | 0.1402 | n/a | n/a | 1.0963 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0191 | 0.0198 | 0.0176 | 0.0221 | 0.0018 | 0.1099 | n/a | n/a | 0.7211 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0021 | 0.0020 | 0.0017 | 0.0022 | 0.0002 | 0.1099 | n/a | n/a | 0.7211 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0138 | 0.0133 | 0.0117 | 0.0143 | 0.0009 | 0.1099 | n/a | n/a | 0.7211 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.1009 | 0.1015 | 0.0989 | 0.1070 | 0.0030 | 0.1154 | n/a | n/a | 1.5165 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0116 | 0.0119 | 0.0101 | 0.0138 | 0.0012 | 0.1154 | n/a | n/a | 1.5165 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1531 | 0.1500 | 0.1366 | 0.1661 | 0.0110 | 0.1154 | n/a | n/a | 1.5165 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0216 | 0.0228 | 0.0209 | 0.0281 | 0.0027 | 0.2097 | n/a | n/a | 1.0245 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0045 | 0.0043 | 0.0036 | 0.0048 | 0.0004 | 0.2097 | n/a | n/a | 1.0245 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0222 | 0.0216 | 0.0149 | 0.0280 | 0.0047 | 0.2097 | n/a | n/a | 1.0245 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0266 | 0.0256 | 0.0224 | 0.0280 | 0.0021 | 0.1390 | n/a | n/a | 0.9400 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0037 | 0.0038 | 0.0035 | 0.0041 | 0.0002 | 0.1390 | n/a | n/a | 0.9400 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0250 | 0.0239 | 0.0146 | 0.0323 | 0.0078 | 0.1390 | n/a | n/a | 0.9400 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0221 | 0.0250 | 0.0208 | 0.0365 | 0.0059 | 0.2815 | n/a | n/a | 1.2414 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0062 | 0.0062 | 0.0054 | 0.0068 | 0.0005 | 0.2815 | n/a | n/a | 1.2414 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0275 | 0.0276 | 0.0219 | 0.0327 | 0.0036 | 0.2815 | n/a | n/a | 1.2414 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0234 | 0.0235 | 0.0210 | 0.0256 | 0.0015 | 0.1795 | n/a | n/a | 0.5271 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0043 | 0.0042 | 0.0044 | 0.0001 | 0.1795 | n/a | n/a | 0.5271 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0123 | 0.0129 | 0.0114 | 0.0165 | 0.0018 | 0.1795 | n/a | n/a | 0.5271 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1060 | 0.1089 | 0.1033 | 0.1213 | 0.0064 | 0.2132 | n/a | n/a | 1.5989 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0226 | 0.0227 | 0.0219 | 0.0237 | 0.0007 | 0.2132 | n/a | n/a | 1.5989 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1695 | 0.1706 | 0.1598 | 0.1819 | 0.0072 | 0.2132 | n/a | n/a | 1.5989 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0150 | 0.0151 | 0.0150 | 0.0154 | 0.0001 | 0.1146 | n/a | n/a | 3.0939 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.1146 | n/a | n/a | 3.0939 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0464 | 0.0494 | 0.0455 | 0.0546 | 0.0042 | 0.1146 | n/a | n/a | 3.0939 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0163 | 0.0164 | 0.0153 | 0.0177 | 0.0010 | 0.2714 | n/a | n/a | 4.8278 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0047 | 0.0001 | 0.2714 | n/a | n/a | 4.8278 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0786 | 0.0795 | 0.0776 | 0.0837 | 0.0022 | 0.2714 | n/a | n/a | 4.8278 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0151 | 0.0153 | 0.0149 | 0.0158 | 0.0004 | 0.0884 | n/a | n/a | 2.1079 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0019 | 0.0003 | 0.0884 | n/a | n/a | 2.1079 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0317 | 0.0318 | 0.0311 | 0.0334 | 0.0008 | 0.0884 | n/a | n/a | 2.1079 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0157 | 0.0165 | 0.0152 | 0.0202 | 0.0018 | 0.0841 | n/a | n/a | 2.0138 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0016 | 0.0001 | 0.0841 | n/a | n/a | 2.0138 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0317 | 0.0321 | 0.0302 | 0.0352 | 0.0017 | 0.0841 | n/a | n/a | 2.0138 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0151 | 0.0152 | 0.0148 | 0.0160 | 0.0004 | 0.0937 | n/a | n/a | 2.0627 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0019 | 0.0013 | 0.0027 | 0.0007 | 0.0937 | n/a | n/a | 2.0627 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0311 | 0.0326 | 0.0301 | 0.0366 | 0.0025 | 0.0937 | n/a | n/a | 2.0627 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0159 | 0.0158 | 0.0149 | 0.0166 | 0.0007 | 0.1151 | n/a | n/a | 2.9988 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0018 | 0.0021 | 0.0017 | 0.0031 | 0.0005 | 0.1151 | n/a | n/a | 2.9988 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0476 | 0.0482 | 0.0465 | 0.0518 | 0.0018 | 0.1151 | n/a | n/a | 2.9988 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0153 | 0.0155 | 0.0151 | 0.0166 | 0.0005 | 0.2949 | n/a | n/a | 5.4377 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0045 | 0.0047 | 0.0044 | 0.0055 | 0.0004 | 0.2949 | n/a | n/a | 5.4377 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0829 | 0.0809 | 0.0761 | 0.0844 | 0.0032 | 0.2949 | n/a | n/a | 5.4377 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0158 | 0.0162 | 0.0151 | 0.0189 | 0.0013 | 0.1908 | n/a | n/a | 2.2017 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0030 | 0.0025 | 0.0013 | 0.0031 | 0.0007 | 0.1908 | n/a | n/a | 2.2017 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0349 | 0.0374 | 0.0339 | 0.0453 | 0.0042 | 0.1908 | n/a | n/a | 2.2017 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0171 | 0.0171 | 0.0152 | 0.0188 | 0.0015 | 0.0770 | n/a | n/a | 1.8615 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.0770 | n/a | n/a | 1.8615 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0318 | 0.0322 | 0.0312 | 0.0338 | 0.0009 | 0.0770 | n/a | n/a | 1.8615 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0154 | 0.0153 | 0.0150 | 0.0158 | 0.0003 | 0.0877 | n/a | n/a | 2.1846 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.0877 | n/a | n/a | 2.1846 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0337 | 0.0361 | 0.0313 | 0.0460 | 0.0054 | 0.0877 | n/a | n/a | 2.1846 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0160 | 0.0163 | 0.0157 | 0.0178 | 0.0007 | 0.1064 | n/a | n/a | 3.0131 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.1064 | n/a | n/a | 3.0131 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0483 | 0.0489 | 0.0474 | 0.0510 | 0.0015 | 0.1064 | n/a | n/a | 3.0131 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0154 | 0.0155 | 0.0153 | 0.0160 | 0.0003 | 0.3098 | n/a | n/a | 5.5404 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0048 | 0.0047 | 0.0044 | 0.0050 | 0.0002 | 0.3098 | n/a | n/a | 5.5404 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0855 | 0.0841 | 0.0792 | 0.0861 | 0.0026 | 0.3098 | n/a | n/a | 5.5404 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0157 | 0.0163 | 0.0153 | 0.0175 | 0.0009 | 0.0866 | n/a | n/a | 2.2677 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0014 | 0.0017 | 0.0013 | 0.0028 | 0.0006 | 0.0866 | n/a | n/a | 2.2677 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0355 | 0.0351 | 0.0309 | 0.0386 | 0.0025 | 0.0866 | n/a | n/a | 2.2677 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0154 | 0.0154 | 0.0152 | 0.0156 | 0.0001 | 0.0895 | n/a | n/a | 2.2267 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0014 | 0.0015 | 0.0012 | 0.0020 | 0.0003 | 0.0895 | n/a | n/a | 2.2267 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0342 | 0.0334 | 0.0311 | 0.0351 | 0.0015 | 0.0895 | n/a | n/a | 2.2267 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0158 | 0.0157 | 0.0152 | 0.0162 | 0.0004 | 0.0955 | n/a | n/a | 1.9782 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0015 | 0.0016 | 0.0014 | 0.0020 | 0.0003 | 0.0955 | n/a | n/a | 1.9782 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0312 | 0.0313 | 0.0310 | 0.0317 | 0.0003 | 0.0955 | n/a | n/a | 1.9782 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0152 | 0.0151 | 0.0149 | 0.0152 | 0.0001 | 0.1213 | n/a | n/a | 3.3391 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.1213 | n/a | n/a | 3.3391 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0506 | 0.0586 | 0.0492 | 0.0919 | 0.0167 | 0.1213 | n/a | n/a | 3.3391 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0166 | 0.0172 | 0.0157 | 0.0199 | 0.0015 | 0.2828 | n/a | n/a | 4.7918 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0047 | 0.0049 | 0.0044 | 0.0062 | 0.0007 | 0.2828 | n/a | n/a | 4.7918 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0795 | 0.0803 | 0.0780 | 0.0842 | 0.0022 | 0.2828 | n/a | n/a | 4.7918 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0166 | 0.0168 | 0.0165 | 0.0176 | 0.0004 | 0.1047 | n/a | n/a | 1.8671 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0017 | 0.0017 | 0.0015 | 0.0018 | 0.0001 | 0.1047 | n/a | n/a | 1.8671 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0309 | 0.0312 | 0.0306 | 0.0324 | 0.0006 | 0.1047 | n/a | n/a | 1.8671 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0162 | 0.0165 | 0.0155 | 0.0180 | 0.0009 | 0.0900 | n/a | n/a | 2.0390 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.0900 | n/a | n/a | 2.0390 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0329 | 0.0336 | 0.0316 | 0.0363 | 0.0017 | 0.0900 | n/a | n/a | 2.0390 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0326 | 0.0322 | 0.0300 | 0.0335 | 0.0012 | 0.1195 | n/a | n/a | 4.1409 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0039 | 0.0039 | 0.0036 | 0.0045 | 0.0003 | 0.1195 | n/a | n/a | 4.1409 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.1349 | 0.1320 | 0.0995 | 0.1568 | 0.0222 | 0.1195 | n/a | n/a | 4.1409 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0300 | 0.0299 | 0.0294 | 0.0302 | 0.0003 | 0.2116 | n/a | n/a | 6.3432 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0064 | 0.0064 | 0.0061 | 0.0066 | 0.0002 | 0.2116 | n/a | n/a | 6.3432 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.1904 | 0.1922 | 0.1795 | 0.2087 | 0.0097 | 0.2116 | n/a | n/a | 6.3432 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0913 | 0.0889 | 0.0577 | 0.1251 | 0.0223 | 0.3455 | n/a | n/a | 2.3762 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0316 | 0.0305 | 0.0257 | 0.0323 | 0.0025 | 0.3455 | n/a | n/a | 2.3762 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.2170 | 0.2221 | 0.2005 | 0.2464 | 0.0156 | 0.3455 | n/a | n/a | 2.3762 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0515 | 0.0514 | 0.0446 | 0.0570 | 0.0041 | 0.1916 | n/a | n/a | 1.8712 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0099 | 0.0092 | 0.0074 | 0.0104 | 0.0011 | 0.1916 | n/a | n/a | 1.8712 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0963 | 0.0968 | 0.0917 | 0.1041 | 0.0045 | 0.1916 | n/a | n/a | 1.8712 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0454 | 0.0484 | 0.0424 | 0.0554 | 0.0056 | 0.1638 | n/a | n/a | 1.8993 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0074 | 0.0069 | 0.0052 | 0.0080 | 0.0010 | 0.1638 | n/a | n/a | 1.8993 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0862 | 0.0818 | 0.0696 | 0.0884 | 0.0071 | 0.1638 | n/a | n/a | 1.8993 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0481 | 0.0490 | 0.0457 | 0.0545 | 0.0030 | 0.1621 | n/a | n/a | 1.6538 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0078 | 0.0077 | 0.0075 | 0.0079 | 0.0001 | 0.1621 | n/a | n/a | 1.6538 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0796 | 0.0792 | 0.0681 | 0.0870 | 0.0062 | 0.1621 | n/a | n/a | 1.6538 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0406 | 0.0406 | 0.0370 | 0.0450 | 0.0032 | 0.1586 | n/a | n/a | 3.3336 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0064 | 0.0079 | 0.0059 | 0.0107 | 0.0021 | 0.1586 | n/a | n/a | 3.3336 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.1352 | 0.1332 | 0.1177 | 0.1440 | 0.0089 | 0.1586 | n/a | n/a | 3.3336 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0712 | 0.0712 | 0.0586 | 0.0804 | 0.0082 | 0.4195 | n/a | n/a | 3.2327 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0299 | 0.0298 | 0.0296 | 0.0300 | 0.0002 | 0.4195 | n/a | n/a | 3.2327 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.2302 | 0.2362 | 0.2261 | 0.2623 | 0.0132 | 0.4195 | n/a | n/a | 3.2327 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.1344 | 0.1364 | 0.1257 | 0.1487 | 0.0097 | 0.1733 | n/a | n/a | 0.5114 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0233 | 0.0241 | 0.0229 | 0.0263 | 0.0013 | 0.1733 | n/a | n/a | 0.5114 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0687 | 0.0723 | 0.0603 | 0.0953 | 0.0126 | 0.1733 | n/a | n/a | 0.5114 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0685 | 0.0668 | 0.0586 | 0.0708 | 0.0044 | 0.2084 | n/a | n/a | 1.1158 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0143 | 0.0143 | 0.0141 | 0.0145 | 0.0002 | 0.2084 | n/a | n/a | 1.1158 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0765 | 0.0731 | 0.0546 | 0.0821 | 0.0098 | 0.2084 | n/a | n/a | 1.1158 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0785 | 0.0760 | 0.0686 | 0.0810 | 0.0046 | 0.1965 | n/a | n/a | 1.0645 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0154 | 0.0144 | 0.0112 | 0.0179 | 0.0025 | 0.1965 | n/a | n/a | 1.0645 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0836 | 0.0822 | 0.0535 | 0.1072 | 0.0170 | 0.1965 | n/a | n/a | 1.0645 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0446 | 0.0447 | 0.0429 | 0.0462 | 0.0012 | 0.3919 | n/a | n/a | 3.4962 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0175 | 0.0173 | 0.0115 | 0.0205 | 0.0033 | 0.3919 | n/a | n/a | 3.4962 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.1558 | 0.1553 | 0.1490 | 0.1600 | 0.0041 | 0.3919 | n/a | n/a | 3.4962 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1827 | 0.1704 | 0.1251 | 0.2067 | 0.0347 | 0.3151 | n/a | n/a | 1.3443 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0575 | 0.0592 | 0.0454 | 0.0795 | 0.0124 | 0.3151 | n/a | n/a | 1.3443 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.2455 | 0.2589 | 0.2145 | 0.3416 | 0.0440 | 0.3151 | n/a | n/a | 1.3443 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.4448 | 0.4353 | 0.3905 | 0.4544 | 0.0228 | 0.1930 | n/a | n/a | 0.1931 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0859 | 0.0828 | 0.0704 | 0.0875 | 0.0064 | 0.1930 | n/a | n/a | 0.1931 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0859 | 0.0897 | 0.0821 | 0.0998 | 0.0069 | 0.1930 | n/a | n/a | 0.1931 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1605 | 0.1648 | 0.1244 | 0.2056 | 0.0286 | 0.1933 | n/a | n/a | 0.5668 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0310 | 0.0340 | 0.0246 | 0.0426 | 0.0072 | 0.1933 | n/a | n/a | 0.5668 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0910 | 0.0881 | 0.0744 | 0.0995 | 0.0086 | 0.1933 | n/a | n/a | 0.5668 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.2192 | 0.2100 | 0.1787 | 0.2348 | 0.0235 | 0.2327 | n/a | n/a | 0.3488 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0510 | 0.0514 | 0.0510 | 0.0527 | 0.0007 | 0.2327 | n/a | n/a | 0.3488 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0765 | 0.0810 | 0.0612 | 0.1026 | 0.0182 | 0.2327 | n/a | n/a | 0.3488 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0282 | 0.0286 | 0.0245 | 0.0315 | 0.0025 | 0.1312 | n/a | n/a | 0.7108 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0037 | 0.0041 | 0.0031 | 0.0054 | 0.0009 | 0.1312 | n/a | n/a | 0.7108 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0201 | 0.0214 | 0.0165 | 0.0277 | 0.0042 | 0.1312 | n/a | n/a | 0.7108 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0238 | 0.0252 | 0.0222 | 0.0284 | 0.0026 | 0.4702 | n/a | n/a | 0.8721 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0112 | 0.0117 | 0.0103 | 0.0148 | 0.0016 | 0.4702 | n/a | n/a | 0.8721 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0208 | 0.0207 | 0.0158 | 0.0251 | 0.0031 | 0.4702 | n/a | n/a | 0.8721 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0209 | 0.0205 | 0.0180 | 0.0218 | 0.0013 | 0.1204 | n/a | n/a | 0.9360 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0025 | 0.0027 | 0.0024 | 0.0030 | 0.0002 | 0.1204 | n/a | n/a | 0.9360 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0196 | 0.0196 | 0.0158 | 0.0239 | 0.0033 | 0.1204 | n/a | n/a | 0.9360 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0202 | 0.0211 | 0.0175 | 0.0248 | 0.0031 | 0.7495 | n/a | n/a | 0.8731 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0151 | 0.0141 | 0.0107 | 0.0175 | 0.0026 | 0.7495 | n/a | n/a | 0.8731 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0176 | 0.0185 | 0.0145 | 0.0259 | 0.0039 | 0.7495 | n/a | n/a | 0.8731 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0198 | 0.0201 | 0.0181 | 0.0232 | 0.0017 | 0.1845 | n/a | n/a | 1.0314 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0037 | 0.0038 | 0.0024 | 0.0051 | 0.0009 | 0.1845 | n/a | n/a | 1.0314 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0205 | 0.0204 | 0.0149 | 0.0271 | 0.0039 | 0.1845 | n/a | n/a | 1.0314 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0187 | 0.0199 | 0.0177 | 0.0254 | 0.0028 | 0.5575 | n/a | n/a | 0.7616 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0104 | 0.0113 | 0.0094 | 0.0143 | 0.0020 | 0.5575 | n/a | n/a | 0.7616 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0143 | 0.0145 | 0.0137 | 0.0155 | 0.0006 | 0.5575 | n/a | n/a | 0.7616 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0185 | 0.0180 | 0.0163 | 0.0196 | 0.0012 | 0.1392 | n/a | n/a | 1.0935 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0026 | 0.0027 | 0.0023 | 0.0034 | 0.0004 | 0.1392 | n/a | n/a | 1.0935 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0202 | 0.0201 | 0.0165 | 0.0227 | 0.0021 | 0.1392 | n/a | n/a | 1.0935 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0187 | 0.0196 | 0.0178 | 0.0217 | 0.0015 | 0.5459 | n/a | n/a | 0.7800 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0102 | 0.0100 | 0.0088 | 0.0110 | 0.0009 | 0.5459 | n/a | n/a | 0.7800 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0146 | 0.0151 | 0.0135 | 0.0171 | 0.0014 | 0.5459 | n/a | n/a | 0.7800 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0173 | 0.0176 | 0.0169 | 0.0188 | 0.0007 | 0.1564 | n/a | n/a | 1.2349 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.1564 | n/a | n/a | 1.2349 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0213 | 0.0208 | 0.0188 | 0.0226 | 0.0013 | 0.1564 | n/a | n/a | 1.2349 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0173 | 0.0176 | 0.0164 | 0.0198 | 0.0012 | 0.5724 | n/a | n/a | 1.1735 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0099 | 0.0107 | 0.0089 | 0.0139 | 0.0018 | 0.5724 | n/a | n/a | 1.1735 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0203 | 0.0204 | 0.0175 | 0.0231 | 0.0020 | 0.5724 | n/a | n/a | 1.1735 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0213 | 0.0210 | 0.0196 | 0.0216 | 0.0007 | 0.1748 | n/a | n/a | 1.1591 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0037 | 0.0037 | 0.0035 | 0.0039 | 0.0002 | 0.1748 | n/a | n/a | 1.1591 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0247 | 0.0259 | 0.0224 | 0.0302 | 0.0031 | 0.1748 | n/a | n/a | 1.1591 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0208 | 0.0209 | 0.0202 | 0.0217 | 0.0005 | 0.4867 | n/a | n/a | 1.1564 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0101 | 0.0101 | 0.0095 | 0.0109 | 0.0006 | 0.4867 | n/a | n/a | 1.1564 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0241 | 0.0221 | 0.0156 | 0.0275 | 0.0047 | 0.4867 | n/a | n/a | 1.1564 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0353 | 0.0366 | 0.0342 | 0.0411 | 0.0025 | 0.3506 | n/a | n/a | 0.7505 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0124 | 0.0104 | 0.0068 | 0.0127 | 0.0026 | 0.3506 | n/a | n/a | 0.7505 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0265 | 0.0290 | 0.0252 | 0.0345 | 0.0037 | 0.3506 | n/a | n/a | 0.7505 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0352 | 0.0348 | 0.0325 | 0.0366 | 0.0017 | 0.4094 | n/a | n/a | 1.1359 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0144 | 0.0152 | 0.0137 | 0.0187 | 0.0018 | 0.4094 | n/a | n/a | 1.1359 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0399 | 0.0390 | 0.0339 | 0.0465 | 0.0046 | 0.4094 | n/a | n/a | 1.1359 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0172 | 0.0173 | 0.0167 | 0.0180 | 0.0004 | 0.1551 | n/a | n/a | 0.7942 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0027 | 0.0028 | 0.0027 | 0.0031 | 0.0002 | 0.1551 | n/a | n/a | 0.7942 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0136 | 0.0136 | 0.0126 | 0.0143 | 0.0006 | 0.1551 | n/a | n/a | 0.7942 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0103 | 0.0103 | 0.0099 | 0.0107 | 0.0003 | 1.0867 | n/a | n/a | 1.2464 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0111 | 0.0111 | 0.0107 | 0.0117 | 0.0004 | 1.0867 | n/a | n/a | 1.2464 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0128 | 0.0129 | 0.0121 | 0.0138 | 0.0005 | 1.0867 | n/a | n/a | 1.2464 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0197 | 0.0195 | 0.0190 | 0.0198 | 0.0003 | 0.2351 | n/a | n/a | 1.0896 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0046 | 0.0047 | 0.0041 | 0.0057 | 0.0005 | 0.2351 | n/a | n/a | 1.0896 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0214 | 0.0216 | 0.0196 | 0.0230 | 0.0012 | 0.2351 | n/a | n/a | 1.0896 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0143 | 0.0149 | 0.0131 | 0.0168 | 0.0015 | 0.9116 | n/a | n/a | 1.5547 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0130 | 0.0129 | 0.0120 | 0.0137 | 0.0006 | 0.9116 | n/a | n/a | 1.5547 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0222 | 0.0200 | 0.0139 | 0.0244 | 0.0040 | 0.9116 | n/a | n/a | 1.5547 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0343 | 0.0347 | 0.0308 | 0.0403 | 0.0031 | 0.2426 | n/a | n/a | 1.1097 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0083 | 0.0086 | 0.0081 | 0.0099 | 0.0007 | 0.2426 | n/a | n/a | 1.1097 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0381 | 0.0366 | 0.0230 | 0.0457 | 0.0074 | 0.2426 | n/a | n/a | 1.1097 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0235 | 0.0243 | 0.0228 | 0.0279 | 0.0018 | 0.7191 | n/a | n/a | 0.9271 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0169 | 0.0170 | 0.0159 | 0.0182 | 0.0008 | 0.7191 | n/a | n/a | 0.9271 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0218 | 0.0250 | 0.0187 | 0.0361 | 0.0067 | 0.7191 | n/a | n/a | 0.9271 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0803 | 0.0826 | 0.0788 | 0.0895 | 0.0040 | 0.2725 | n/a | n/a | 0.9070 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0219 | 0.0227 | 0.0218 | 0.0250 | 0.0012 | 0.2725 | n/a | n/a | 0.9070 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0728 | 0.0758 | 0.0699 | 0.0827 | 0.0056 | 0.2725 | n/a | n/a | 0.9070 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0815 | 0.0765 | 0.0579 | 0.1013 | 0.0165 | 0.2938 | n/a | n/a | 0.3869 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0240 | 0.0253 | 0.0237 | 0.0305 | 0.0026 | 0.2938 | n/a | n/a | 0.3869 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0315 | 0.0288 | 0.0194 | 0.0365 | 0.0066 | 0.2938 | n/a | n/a | 0.3869 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0185 | 0.0192 | 0.0182 | 0.0219 | 0.0014 | 0.1113 | n/a | n/a | 1.0484 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0021 | 0.0021 | 0.0019 | 0.0025 | 0.0002 | 0.1113 | n/a | n/a | 1.0484 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0194 | 0.0201 | 0.0162 | 0.0253 | 0.0035 | 0.1113 | n/a | n/a | 1.0484 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1025 | 0.1125 | 0.1019 | 0.1396 | 0.0146 | 0.1044 | n/a | n/a | 0.1585 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0107 | 0.0110 | 0.0106 | 0.0124 | 0.0007 | 0.1044 | n/a | n/a | 0.1585 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0163 | 0.0223 | 0.0161 | 0.0328 | 0.0075 | 0.1044 | n/a | n/a | 0.1585 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4062 | 0.4056 | 0.3955 | 0.4179 | 0.0073 | 0.9232 | n/a | n/a | 0.2978 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3750 | 0.3578 | 0.3180 | 0.3922 | 0.0295 | 0.9232 | n/a | n/a | 0.2978 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1210 | 0.1251 | 0.1162 | 0.1362 | 0.0076 | 0.9232 | n/a | n/a | 0.2978 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.3777 | 2.3616 | 2.0810 | 2.5529 | 0.1567 | 0.1820 | n/a | n/a | 0.1047 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4328 | 0.4810 | 0.4189 | 0.6413 | 0.0834 | 0.1820 | n/a | n/a | 0.1047 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2489 | 0.2482 | 0.2092 | 0.2913 | 0.0307 | 0.1820 | n/a | n/a | 0.1047 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2216 | 0.2202 | 0.2146 | 0.2250 | 0.0039 | 6.2770 | n/a | n/a | 0.4841 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.3908 | 1.3711 | 1.2640 | 1.4629 | 0.0865 | 6.2770 | n/a | n/a | 0.4841 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1073 | 0.1124 | 0.0980 | 0.1428 | 0.0157 | 6.2770 | n/a | n/a | 0.4841 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.1611 | 7.1286 | 6.7045 | 7.6052 | 0.3059 | 9.3614 | n/a | n/a | 0.0502 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 67.0382 | 69.2606 | 63.9432 | 78.8307 | 5.5685 | 9.3614 | n/a | n/a | 0.0502 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3592 | 0.3531 | 0.3214 | 0.3878 | 0.0262 | 9.3614 | n/a | n/a | 0.0502 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0280 | 0.0278 | 0.0271 | 0.0286 | 0.0006 | 6.5816 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1846 | 0.1843 | 0.1678 | 0.2091 | 0.0153 | 6.5816 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0281 | 0.0300 | 0.0276 | 0.0333 | 0.0026 | 19.8874 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5595 | 0.5841 | 0.5525 | 0.6913 | 0.0537 | 19.8874 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7594 | 0.7538 | 0.7309 | 0.7722 | 0.0164 | 11.7539 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.9265 | 8.9882 | 8.0946 | 10.4373 | 0.7987 | 11.7539 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7817 | 0.7712 | 0.7490 | 0.7901 | 0.0179 | 37.2751 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 29.1372 | 29.2477 | 28.1779 | 30.8764 | 0.9794 | 37.2751 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1214 | 0.1207 | 0.1164 | 0.1267 | 0.0038 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.4898 | 0.4570 | 0.2995 | 0.5584 | 0.0921 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2587 | 0.2640 | 0.2498 | 0.2812 | 0.0134 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.7483 | 0.7425 | 0.6799 | 0.8047 | 0.0404 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1741 | 1.1929 | 1.1624 | 1.2794 | 0.0435 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.8255 | 2.8814 | 2.7657 | 3.1272 | 0.1302 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.5061 | 5.5186 | 5.4325 | 5.6569 | 0.0796 | 0.0295 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1624 | 0.1678 | 0.1592 | 0.1807 | 0.0087 | 0.0295 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
