# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.11.0`
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
- TensorStudio wins versus JAX CPU dispatch: `42`
- TensorStudio losses versus JAX CPU dispatch: `56`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0176 | 0.0176 | 0.0167 | 0.0187 | 0.0008 | 0.0400 | n/a | n/a | 0.7753 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0015 | 0.0003 | 0.0400 | n/a | n/a | 0.7753 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0136 | 0.0141 | 0.0129 | 0.0167 | 0.0013 | 0.0400 | n/a | n/a | 0.7753 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0173 | 0.0177 | 0.0171 | 0.0196 | 0.0010 | 0.0426 | n/a | n/a | 0.8633 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0426 | n/a | n/a | 0.8633 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0149 | 0.0144 | 0.0131 | 0.0156 | 0.0011 | 0.0426 | n/a | n/a | 0.8633 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0172 | 0.0173 | 0.0168 | 0.0179 | 0.0004 | 0.0984 | n/a | n/a | 0.9386 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0017 | 0.0014 | 0.0008 | 0.0017 | 0.0004 | 0.0984 | n/a | n/a | 0.9386 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0161 | 0.0155 | 0.0132 | 0.0181 | 0.0018 | 0.0984 | n/a | n/a | 0.9386 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0177 | 0.0180 | 0.0169 | 0.0193 | 0.0008 | 0.0436 | n/a | n/a | 0.9476 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0009 | 0.0000 | 0.0436 | n/a | n/a | 0.9476 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0168 | 0.0149 | 0.0105 | 0.0183 | 0.0032 | 0.0436 | n/a | n/a | 0.9476 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0922 | 0.1023 | 0.0886 | 0.1353 | 0.0173 | 0.0605 | n/a | n/a | 0.9957 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0056 | 0.0057 | 0.0055 | 0.0062 | 0.0002 | 0.0605 | n/a | n/a | 0.9957 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0918 | 0.0933 | 0.0915 | 0.0984 | 0.0026 | 0.0605 | n/a | n/a | 0.9957 | no | n/a | n/a | no | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0164 | 0.0164 | 0.0161 | 0.0168 | 0.0002 | 0.0392 | n/a | n/a | 0.7861 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0007 | 0.0000 | 0.0392 | n/a | n/a | 0.7861 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0129 | 0.0135 | 0.0120 | 0.0170 | 0.0018 | 0.0392 | n/a | n/a | 0.7861 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0175 | 0.0178 | 0.0162 | 0.0200 | 0.0013 | 0.0419 | n/a | n/a | 0.7906 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0419 | n/a | n/a | 0.7906 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0138 | 0.0140 | 0.0126 | 0.0154 | 0.0011 | 0.0419 | n/a | n/a | 0.7906 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0213 | 0.0255 | 0.0173 | 0.0464 | 0.0108 | 0.0384 | n/a | n/a | 0.6219 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0009 | 0.0000 | 0.0384 | n/a | n/a | 0.6219 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0132 | 0.0132 | 0.0129 | 0.0134 | 0.0002 | 0.0384 | n/a | n/a | 0.6219 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0181 | 0.0185 | 0.0166 | 0.0218 | 0.0018 | 0.0414 | n/a | n/a | 0.5719 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0012 | 0.0002 | 0.0414 | n/a | n/a | 0.5719 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0103 | 0.0106 | 0.0093 | 0.0119 | 0.0009 | 0.0414 | n/a | n/a | 0.5719 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0887 | 0.0882 | 0.0858 | 0.0901 | 0.0015 | 0.0617 | n/a | n/a | 1.1292 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0055 | 0.0055 | 0.0053 | 0.0057 | 0.0002 | 0.0617 | n/a | n/a | 1.1292 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1001 | 0.1030 | 0.0919 | 0.1287 | 0.0134 | 0.0617 | n/a | n/a | 1.1292 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0160 | 0.0164 | 0.0160 | 0.0177 | 0.0007 | 0.0405 | n/a | n/a | 0.8057 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0006 | 0.0008 | 0.0006 | 0.0013 | 0.0003 | 0.0405 | n/a | n/a | 0.8057 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0129 | 0.0132 | 0.0122 | 0.0151 | 0.0011 | 0.0405 | n/a | n/a | 0.8057 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0170 | 0.0168 | 0.0161 | 0.0171 | 0.0004 | 0.0406 | n/a | n/a | 0.8311 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0406 | n/a | n/a | 0.8311 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0141 | 0.0150 | 0.0127 | 0.0198 | 0.0026 | 0.0406 | n/a | n/a | 0.8311 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0193 | 0.0185 | 0.0160 | 0.0204 | 0.0019 | 0.0377 | n/a | n/a | 0.8686 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0377 | n/a | n/a | 0.8686 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0168 | 0.0154 | 0.0124 | 0.0185 | 0.0025 | 0.0377 | n/a | n/a | 0.8686 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0163 | 0.0170 | 0.0160 | 0.0186 | 0.0011 | 0.0455 | n/a | n/a | 0.6108 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.0455 | n/a | n/a | 0.6108 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0100 | 0.0113 | 0.0095 | 0.0150 | 0.0021 | 0.0455 | n/a | n/a | 0.6108 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0915 | 0.1013 | 0.0887 | 0.1456 | 0.0222 | 0.0645 | n/a | n/a | 1.0467 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0059 | 0.0063 | 0.0056 | 0.0081 | 0.0009 | 0.0645 | n/a | n/a | 1.0467 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0958 | 0.0960 | 0.0889 | 0.1063 | 0.0058 | 0.0645 | n/a | n/a | 1.0467 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0166 | 0.0168 | 0.0161 | 0.0179 | 0.0006 | 0.0390 | n/a | n/a | 0.7356 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0390 | n/a | n/a | 0.7356 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0122 | 0.0123 | 0.0121 | 0.0128 | 0.0003 | 0.0390 | n/a | n/a | 0.7356 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0170 | 0.0169 | 0.0161 | 0.0180 | 0.0007 | 0.0751 | n/a | n/a | 0.7478 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0013 | 0.0012 | 0.0007 | 0.0016 | 0.0004 | 0.0751 | n/a | n/a | 0.7478 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0127 | 0.0130 | 0.0126 | 0.0144 | 0.0007 | 0.0751 | n/a | n/a | 0.7478 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0165 | 0.0167 | 0.0160 | 0.0180 | 0.0007 | 0.0447 | n/a | n/a | 0.8121 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0447 | n/a | n/a | 0.8121 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0134 | 0.0137 | 0.0129 | 0.0152 | 0.0008 | 0.0447 | n/a | n/a | 0.8121 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0168 | 0.0172 | 0.0163 | 0.0189 | 0.0009 | 0.0436 | n/a | n/a | 0.5996 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0436 | n/a | n/a | 0.5996 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0100 | 0.0103 | 0.0098 | 0.0115 | 0.0006 | 0.0436 | n/a | n/a | 0.5996 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0908 | 0.0968 | 0.0872 | 0.1247 | 0.0141 | 0.0699 | n/a | n/a | 1.0613 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0063 | 0.0063 | 0.0058 | 0.0069 | 0.0004 | 0.0699 | n/a | n/a | 1.0613 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0964 | 0.0959 | 0.0918 | 0.0987 | 0.0023 | 0.0699 | n/a | n/a | 1.0613 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0168 | 0.0165 | 0.0159 | 0.0170 | 0.0005 | 0.0598 | n/a | n/a | 0.8265 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0598 | n/a | n/a | 0.8265 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0139 | 0.0158 | 0.0135 | 0.0216 | 0.0031 | 0.0598 | n/a | n/a | 0.8265 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0166 | 0.0166 | 0.0160 | 0.0176 | 0.0006 | 0.1234 | n/a | n/a | 0.9351 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0020 | 0.0017 | 0.0011 | 0.0021 | 0.0004 | 0.1234 | n/a | n/a | 0.9351 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0155 | 0.0152 | 0.0141 | 0.0163 | 0.0008 | 0.1234 | n/a | n/a | 0.9351 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0165 | 0.0165 | 0.0162 | 0.0172 | 0.0003 | 0.0622 | n/a | n/a | 0.9044 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0622 | n/a | n/a | 0.9044 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0149 | 0.0146 | 0.0123 | 0.0159 | 0.0013 | 0.0622 | n/a | n/a | 0.9044 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0182 | 0.0184 | 0.0161 | 0.0219 | 0.0021 | 0.0604 | n/a | n/a | 0.6860 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.0604 | n/a | n/a | 0.6860 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0125 | 0.0134 | 0.0107 | 0.0161 | 0.0021 | 0.0604 | n/a | n/a | 0.6860 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0896 | 0.0903 | 0.0869 | 0.0948 | 0.0027 | 0.0852 | n/a | n/a | 1.3421 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0076 | 0.0077 | 0.0071 | 0.0081 | 0.0004 | 0.0852 | n/a | n/a | 1.3421 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1203 | 0.1229 | 0.1144 | 0.1368 | 0.0077 | 0.0852 | n/a | n/a | 1.3421 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0179 | 0.0181 | 0.0175 | 0.0191 | 0.0006 | 0.0823 | n/a | n/a | 0.9196 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0018 | 0.0001 | 0.0823 | n/a | n/a | 0.9196 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0165 | 0.0165 | 0.0149 | 0.0182 | 0.0011 | 0.0823 | n/a | n/a | 0.9196 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0177 | 0.0185 | 0.0173 | 0.0220 | 0.0018 | 0.1460 | n/a | n/a | 0.9198 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0026 | 0.0000 | 0.1460 | n/a | n/a | 0.9198 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0163 | 0.0176 | 0.0134 | 0.0222 | 0.0032 | 0.1460 | n/a | n/a | 0.9198 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0178 | 0.0177 | 0.0173 | 0.0183 | 0.0004 | 0.0940 | n/a | n/a | 0.8145 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0017 | 0.0019 | 0.0015 | 0.0031 | 0.0006 | 0.0940 | n/a | n/a | 0.8145 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0145 | 0.0153 | 0.0131 | 0.0203 | 0.0026 | 0.0940 | n/a | n/a | 0.8145 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0213 | 0.0215 | 0.0180 | 0.0283 | 0.0038 | 0.1721 | n/a | n/a | 1.3479 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0037 | 0.0071 | 0.0029 | 0.0212 | 0.0071 | 0.1721 | n/a | n/a | 1.3479 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0287 | 0.0327 | 0.0172 | 0.0624 | 0.0156 | 0.1721 | n/a | n/a | 1.3479 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0925 | 0.1008 | 0.0904 | 0.1301 | 0.0151 | 0.1102 | n/a | n/a | 1.4073 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0102 | 0.0102 | 0.0098 | 0.0112 | 0.0005 | 0.1102 | n/a | n/a | 1.4073 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1301 | 0.1317 | 0.1246 | 0.1452 | 0.0071 | 0.1102 | n/a | n/a | 1.4073 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0210 | 0.0209 | 0.0200 | 0.0218 | 0.0007 | 0.1693 | n/a | n/a | 0.6253 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0035 | 0.0037 | 0.0035 | 0.0040 | 0.0002 | 0.1693 | n/a | n/a | 0.6253 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0131 | 0.0138 | 0.0129 | 0.0166 | 0.0014 | 0.1693 | n/a | n/a | 0.6253 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0211 | 0.0211 | 0.0207 | 0.0217 | 0.0003 | 0.1837 | n/a | n/a | 0.6273 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0000 | 0.1837 | n/a | n/a | 0.6273 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0132 | 0.0141 | 0.0128 | 0.0168 | 0.0015 | 0.1837 | n/a | n/a | 0.6273 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0225 | 0.0245 | 0.0203 | 0.0311 | 0.0042 | 0.2454 | n/a | n/a | 0.6729 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0055 | 0.0056 | 0.0051 | 0.0064 | 0.0004 | 0.2454 | n/a | n/a | 0.6729 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0152 | 0.0150 | 0.0141 | 0.0160 | 0.0008 | 0.2454 | n/a | n/a | 0.6729 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0223 | 0.0226 | 0.0219 | 0.0248 | 0.0011 | 0.1959 | n/a | n/a | 0.5358 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0044 | 0.0045 | 0.0043 | 0.0052 | 0.0003 | 0.1959 | n/a | n/a | 0.5358 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0119 | 0.0122 | 0.0117 | 0.0134 | 0.0006 | 0.1959 | n/a | n/a | 0.5358 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1044 | 0.1039 | 0.1020 | 0.1055 | 0.0013 | 0.2116 | n/a | n/a | 1.3017 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0221 | 0.0230 | 0.0219 | 0.0261 | 0.0016 | 0.2116 | n/a | n/a | 1.3017 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1359 | 0.1351 | 0.1169 | 0.1532 | 0.0137 | 0.2116 | n/a | n/a | 1.3017 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0148 | 0.0155 | 0.0147 | 0.0183 | 0.0014 | 0.1189 | n/a | n/a | 3.2840 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0018 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.1189 | n/a | n/a | 3.2840 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0487 | 0.0503 | 0.0473 | 0.0561 | 0.0034 | 0.1189 | n/a | n/a | 3.2840 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0152 | 0.0153 | 0.0149 | 0.0161 | 0.0004 | 0.3011 | n/a | n/a | 5.0528 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0046 | 0.0054 | 0.0044 | 0.0074 | 0.0012 | 0.3011 | n/a | n/a | 5.0528 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0766 | 0.0781 | 0.0749 | 0.0838 | 0.0032 | 0.3011 | n/a | n/a | 5.0528 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0161 | 0.0180 | 0.0148 | 0.0255 | 0.0040 | 0.2033 | n/a | n/a | 1.9925 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0033 | 0.0035 | 0.0031 | 0.0040 | 0.0004 | 0.2033 | n/a | n/a | 1.9925 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0320 | 0.0356 | 0.0303 | 0.0500 | 0.0073 | 0.2033 | n/a | n/a | 1.9925 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0155 | 0.0160 | 0.0149 | 0.0180 | 0.0012 | 0.0803 | n/a | n/a | 2.0253 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0803 | n/a | n/a | 2.0253 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0313 | 0.0320 | 0.0300 | 0.0345 | 0.0016 | 0.0803 | n/a | n/a | 2.0253 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0153 | 0.0162 | 0.0150 | 0.0184 | 0.0013 | 0.0958 | n/a | n/a | 2.0922 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0015 | 0.0019 | 0.0013 | 0.0028 | 0.0006 | 0.0958 | n/a | n/a | 2.0922 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0321 | 0.0322 | 0.0308 | 0.0348 | 0.0014 | 0.0958 | n/a | n/a | 2.0922 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0165 | 0.0162 | 0.0147 | 0.0173 | 0.0009 | 0.1068 | n/a | n/a | 2.7773 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0022 | 0.0002 | 0.1068 | n/a | n/a | 2.7773 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0459 | 0.0533 | 0.0441 | 0.0830 | 0.0149 | 0.1068 | n/a | n/a | 2.7773 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0157 | 0.0157 | 0.0151 | 0.0163 | 0.0004 | 0.2830 | n/a | n/a | 4.8123 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0044 | 0.0045 | 0.0043 | 0.0047 | 0.0002 | 0.2830 | n/a | n/a | 4.8123 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0757 | 0.0786 | 0.0739 | 0.0899 | 0.0060 | 0.2830 | n/a | n/a | 4.8123 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0151 | 0.0151 | 0.0148 | 0.0153 | 0.0002 | 0.0837 | n/a | n/a | 2.0825 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0837 | n/a | n/a | 2.0825 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0314 | 0.0313 | 0.0294 | 0.0329 | 0.0012 | 0.0837 | n/a | n/a | 2.0825 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0150 | 0.0155 | 0.0149 | 0.0168 | 0.0007 | 0.0848 | n/a | n/a | 2.1348 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0848 | n/a | n/a | 2.1348 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0319 | 0.0320 | 0.0295 | 0.0359 | 0.0022 | 0.0848 | n/a | n/a | 2.1348 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0151 | 0.0156 | 0.0149 | 0.0166 | 0.0007 | 0.0922 | n/a | n/a | 2.0387 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0014 | 0.0015 | 0.0013 | 0.0019 | 0.0002 | 0.0922 | n/a | n/a | 2.0387 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0307 | 0.0326 | 0.0301 | 0.0376 | 0.0030 | 0.0922 | n/a | n/a | 2.0387 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0156 | 0.0162 | 0.0149 | 0.0188 | 0.0014 | 0.1136 | n/a | n/a | 2.9313 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0018 | 0.0021 | 0.0017 | 0.0034 | 0.0007 | 0.1136 | n/a | n/a | 2.9313 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0458 | 0.0462 | 0.0456 | 0.0472 | 0.0007 | 0.1136 | n/a | n/a | 2.9313 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0151 | 0.0194 | 0.0149 | 0.0315 | 0.0064 | 0.5609 | n/a | n/a | 5.2163 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0085 | 0.0079 | 0.0064 | 0.0089 | 0.0010 | 0.5609 | n/a | n/a | 5.2163 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0788 | 0.0791 | 0.0750 | 0.0827 | 0.0026 | 0.5609 | n/a | n/a | 5.2163 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0157 | 0.0159 | 0.0154 | 0.0172 | 0.0006 | 0.0826 | n/a | n/a | 2.0394 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0018 | 0.0002 | 0.0826 | n/a | n/a | 2.0394 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0320 | 0.0316 | 0.0304 | 0.0328 | 0.0009 | 0.0826 | n/a | n/a | 2.0394 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0151 | 0.0153 | 0.0148 | 0.0163 | 0.0005 | 0.0839 | n/a | n/a | 2.1748 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0839 | n/a | n/a | 2.1748 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0328 | 0.0326 | 0.0297 | 0.0349 | 0.0017 | 0.0839 | n/a | n/a | 2.1748 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0151 | 0.0151 | 0.0150 | 0.0151 | 0.0000 | 0.0867 | n/a | n/a | 2.0842 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 0.0867 | n/a | n/a | 2.0842 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0314 | 0.0312 | 0.0307 | 0.0317 | 0.0004 | 0.0867 | n/a | n/a | 2.0842 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0240 | 0.0213 | 0.0151 | 0.0258 | 0.0042 | 0.0783 | n/a | n/a | 1.8857 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0019 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.0783 | n/a | n/a | 1.8857 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0453 | 0.0453 | 0.0447 | 0.0457 | 0.0004 | 0.0783 | n/a | n/a | 1.8857 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0153 | 0.0167 | 0.0153 | 0.0221 | 0.0027 | 0.4842 | n/a | n/a | 4.8851 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0074 | 0.0077 | 0.0047 | 0.0136 | 0.0032 | 0.4842 | n/a | n/a | 4.8851 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0748 | 0.0765 | 0.0747 | 0.0794 | 0.0021 | 0.4842 | n/a | n/a | 4.8851 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0165 | 0.0164 | 0.0161 | 0.0168 | 0.0003 | 0.0906 | n/a | n/a | 1.9296 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.0906 | n/a | n/a | 1.9296 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0319 | 0.0324 | 0.0300 | 0.0368 | 0.0023 | 0.0906 | n/a | n/a | 1.9296 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0155 | 0.0159 | 0.0153 | 0.0173 | 0.0007 | 0.0924 | n/a | n/a | 2.0738 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0013 | 0.0017 | 0.0001 | 0.0924 | n/a | n/a | 2.0738 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0322 | 0.0325 | 0.0307 | 0.0343 | 0.0015 | 0.0924 | n/a | n/a | 2.0738 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0167 | 0.0188 | 0.0156 | 0.0281 | 0.0047 | 0.2310 | n/a | n/a | 1.8631 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0039 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.2310 | n/a | n/a | 1.8631 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0312 | 0.0323 | 0.0311 | 0.0345 | 0.0014 | 0.2310 | n/a | n/a | 1.8631 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0153 | 0.0154 | 0.0150 | 0.0160 | 0.0003 | 0.1559 | n/a | n/a | 4.2896 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0001 | 0.1559 | n/a | n/a | 4.2896 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0655 | 0.0669 | 0.0636 | 0.0742 | 0.0037 | 0.1559 | n/a | n/a | 4.2896 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0204 | 0.0216 | 0.0202 | 0.0267 | 0.0026 | 0.3860 | n/a | n/a | 5.3167 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0079 | 0.0091 | 0.0068 | 0.0125 | 0.0023 | 0.3860 | n/a | n/a | 5.3167 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1087 | 0.1082 | 0.0947 | 0.1204 | 0.0082 | 0.3860 | n/a | n/a | 5.3167 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0260 | 0.0261 | 0.0258 | 0.0268 | 0.0004 | 0.1591 | n/a | n/a | 1.1834 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0041 | 0.0041 | 0.0039 | 0.0042 | 0.0001 | 0.1591 | n/a | n/a | 1.1834 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0307 | 0.0308 | 0.0305 | 0.0316 | 0.0004 | 0.1591 | n/a | n/a | 1.1834 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0190 | 0.0212 | 0.0184 | 0.0290 | 0.0040 | 0.1382 | n/a | n/a | 1.6186 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.1382 | n/a | n/a | 1.6186 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0308 | 0.0317 | 0.0300 | 0.0357 | 0.0020 | 0.1382 | n/a | n/a | 1.6186 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0218 | 0.0221 | 0.0204 | 0.0254 | 0.0018 | 0.1415 | n/a | n/a | 1.8856 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0031 | 0.0031 | 0.0030 | 0.0033 | 0.0001 | 0.1415 | n/a | n/a | 1.8856 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0411 | 0.0397 | 0.0316 | 0.0490 | 0.0067 | 0.1415 | n/a | n/a | 1.8856 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0167 | 0.0168 | 0.0163 | 0.0181 | 0.0007 | 0.2080 | n/a | n/a | 3.6915 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0035 | 0.0035 | 0.0035 | 0.0035 | 0.0000 | 0.2080 | n/a | n/a | 3.6915 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0616 | 0.0632 | 0.0577 | 0.0697 | 0.0044 | 0.2080 | n/a | n/a | 3.6915 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0339 | 0.0348 | 0.0336 | 0.0386 | 0.0019 | 0.3724 | n/a | n/a | 2.9776 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0126 | 0.0126 | 0.0118 | 0.0135 | 0.0006 | 0.3724 | n/a | n/a | 2.9776 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1011 | 0.1016 | 0.0946 | 0.1094 | 0.0052 | 0.3724 | n/a | n/a | 2.9776 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0626 | 0.0678 | 0.0621 | 0.0801 | 0.0071 | 0.1807 | n/a | n/a | 0.5363 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0113 | 0.0112 | 0.0105 | 0.0123 | 0.0007 | 0.1807 | n/a | n/a | 0.5363 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0336 | 0.0339 | 0.0327 | 0.0356 | 0.0010 | 0.1807 | n/a | n/a | 0.5363 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0312 | 0.0361 | 0.0311 | 0.0451 | 0.0062 | 0.1962 | n/a | n/a | 1.0695 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0061 | 0.0062 | 0.0060 | 0.0064 | 0.0001 | 0.1962 | n/a | n/a | 1.0695 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0334 | 0.0349 | 0.0327 | 0.0378 | 0.0021 | 0.1962 | n/a | n/a | 1.0695 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0384 | 0.0416 | 0.0369 | 0.0542 | 0.0065 | 0.2008 | n/a | n/a | 0.9485 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0077 | 0.0082 | 0.0076 | 0.0102 | 0.0010 | 0.2008 | n/a | n/a | 0.9485 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0365 | 0.0361 | 0.0342 | 0.0370 | 0.0010 | 0.2008 | n/a | n/a | 0.9485 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0221 | 0.0247 | 0.0216 | 0.0313 | 0.0038 | 0.6971 | n/a | n/a | 3.0324 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0154 | 0.0153 | 0.0144 | 0.0158 | 0.0005 | 0.6971 | n/a | n/a | 3.0324 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0669 | 0.0709 | 0.0548 | 0.0984 | 0.0151 | 0.6971 | n/a | n/a | 3.0324 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0905 | 0.0899 | 0.0866 | 0.0910 | 0.0016 | 0.3739 | n/a | n/a | 1.3793 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0338 | 0.0427 | 0.0327 | 0.0669 | 0.0131 | 0.3739 | n/a | n/a | 1.3793 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1248 | 0.1213 | 0.1033 | 0.1442 | 0.0150 | 0.3739 | n/a | n/a | 1.3793 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2144 | 0.2220 | 0.2125 | 0.2554 | 0.0167 | 0.1744 | n/a | n/a | 0.2207 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0374 | 0.0373 | 0.0365 | 0.0380 | 0.0006 | 0.1744 | n/a | n/a | 0.2207 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0473 | 0.0547 | 0.0391 | 0.0749 | 0.0151 | 0.1744 | n/a | n/a | 0.2207 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0777 | 0.0777 | 0.0763 | 0.0790 | 0.0009 | 0.2658 | n/a | n/a | 0.5427 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0207 | 0.0207 | 0.0203 | 0.0214 | 0.0004 | 0.2658 | n/a | n/a | 0.5427 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0422 | 0.0531 | 0.0405 | 0.0738 | 0.0148 | 0.2658 | n/a | n/a | 0.5427 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.2030 | 0.2164 | 0.1959 | 0.2722 | 0.0286 | 0.1867 | n/a | n/a | 0.2611 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0379 | 0.0372 | 0.0277 | 0.0486 | 0.0070 | 0.1867 | n/a | n/a | 0.2611 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0530 | 0.0511 | 0.0460 | 0.0536 | 0.0031 | 0.1867 | n/a | n/a | 0.2611 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0154 | 0.0159 | 0.0153 | 0.0180 | 0.0011 | 0.1457 | n/a | n/a | 0.8439 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0022 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.1457 | n/a | n/a | 0.8439 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0130 | 0.0131 | 0.0127 | 0.0135 | 0.0003 | 0.1457 | n/a | n/a | 0.8439 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0155 | 0.0158 | 0.0152 | 0.0172 | 0.0007 | 0.6506 | n/a | n/a | 0.7742 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0101 | 0.0109 | 0.0081 | 0.0138 | 0.0021 | 0.6506 | n/a | n/a | 0.7742 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0120 | 0.0119 | 0.0114 | 0.0123 | 0.0004 | 0.6506 | n/a | n/a | 0.7742 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0157 | 0.0164 | 0.0155 | 0.0182 | 0.0010 | 0.1487 | n/a | n/a | 0.8178 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0023 | 0.0026 | 0.0022 | 0.0038 | 0.0006 | 0.1487 | n/a | n/a | 0.8178 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0129 | 0.0136 | 0.0124 | 0.0173 | 0.0018 | 0.1487 | n/a | n/a | 0.8178 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0156 | 0.0159 | 0.0155 | 0.0168 | 0.0005 | 0.5401 | n/a | n/a | 0.8643 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0084 | 0.0085 | 0.0079 | 0.0096 | 0.0006 | 0.5401 | n/a | n/a | 0.8643 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0135 | 0.0139 | 0.0123 | 0.0165 | 0.0016 | 0.5401 | n/a | n/a | 0.8643 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0170 | 0.0168 | 0.0157 | 0.0178 | 0.0009 | 0.1483 | n/a | n/a | 0.7628 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0025 | 0.0025 | 0.0023 | 0.0026 | 0.0001 | 0.1483 | n/a | n/a | 0.7628 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0129 | 0.0134 | 0.0127 | 0.0156 | 0.0011 | 0.1483 | n/a | n/a | 0.7628 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0162 | 0.0163 | 0.0154 | 0.0172 | 0.0006 | 0.5228 | n/a | n/a | 0.9868 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0085 | 0.0096 | 0.0081 | 0.0147 | 0.0026 | 0.5228 | n/a | n/a | 0.9868 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0160 | 0.0155 | 0.0123 | 0.0195 | 0.0028 | 0.5228 | n/a | n/a | 0.9868 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0156 | 0.0167 | 0.0155 | 0.0192 | 0.0015 | 0.1482 | n/a | n/a | 1.3650 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0023 | 0.0026 | 0.0022 | 0.0036 | 0.0005 | 0.1482 | n/a | n/a | 1.3650 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0213 | 0.0222 | 0.0146 | 0.0334 | 0.0070 | 0.1482 | n/a | n/a | 1.3650 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0161 | 0.0001 | 0.5191 | n/a | n/a | 0.8752 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0082 | 0.0087 | 0.0080 | 0.0104 | 0.0009 | 0.5191 | n/a | n/a | 0.8752 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0138 | 0.0140 | 0.0131 | 0.0155 | 0.0008 | 0.5191 | n/a | n/a | 0.8752 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0161 | 0.0163 | 0.0159 | 0.0170 | 0.0004 | 0.1834 | n/a | n/a | 1.2019 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0030 | 0.0030 | 0.0028 | 0.0033 | 0.0002 | 0.1834 | n/a | n/a | 1.2019 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0193 | 0.0186 | 0.0163 | 0.0202 | 0.0014 | 0.1834 | n/a | n/a | 1.2019 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0167 | 0.0167 | 0.0165 | 0.0170 | 0.0002 | 0.5333 | n/a | n/a | 0.8764 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0089 | 0.0089 | 0.0086 | 0.0092 | 0.0002 | 0.5333 | n/a | n/a | 0.8764 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0146 | 0.0149 | 0.0135 | 0.0175 | 0.0014 | 0.5333 | n/a | n/a | 0.8764 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0198 | 0.0198 | 0.0195 | 0.0201 | 0.0002 | 0.1747 | n/a | n/a | 0.8955 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0035 | 0.0035 | 0.0034 | 0.0036 | 0.0001 | 0.1747 | n/a | n/a | 0.8955 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0177 | 0.0177 | 0.0169 | 0.0186 | 0.0005 | 0.1747 | n/a | n/a | 0.8955 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0199 | 0.0199 | 0.0198 | 0.0201 | 0.0001 | 0.5266 | n/a | n/a | 0.8398 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0105 | 0.0107 | 0.0090 | 0.0140 | 0.0018 | 0.5266 | n/a | n/a | 0.8398 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0167 | 0.0179 | 0.0151 | 0.0224 | 0.0027 | 0.5266 | n/a | n/a | 0.8398 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0336 | 0.0356 | 0.0331 | 0.0400 | 0.0028 | 0.2054 | n/a | n/a | 0.6284 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0069 | 0.0073 | 0.0068 | 0.0089 | 0.0008 | 0.2054 | n/a | n/a | 0.6284 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0211 | 0.0234 | 0.0207 | 0.0315 | 0.0041 | 0.2054 | n/a | n/a | 0.6284 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0407 | 0.0407 | 0.0404 | 0.0412 | 0.0003 | 0.5964 | n/a | n/a | 0.6682 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0243 | 0.0242 | 0.0233 | 0.0249 | 0.0006 | 0.5964 | n/a | n/a | 0.6682 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0272 | 0.0306 | 0.0244 | 0.0427 | 0.0068 | 0.5964 | n/a | n/a | 0.6682 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0162 | 0.0163 | 0.0161 | 0.0165 | 0.0002 | 0.1645 | n/a | n/a | 0.9159 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0027 | 0.0027 | 0.0027 | 0.0027 | 0.0000 | 0.1645 | n/a | n/a | 0.9159 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0148 | 0.0154 | 0.0136 | 0.0191 | 0.0020 | 0.1645 | n/a | n/a | 0.9159 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0102 | 0.0102 | 0.0099 | 0.0105 | 0.0002 | 1.0501 | n/a | n/a | 1.2391 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0107 | 0.0106 | 0.0101 | 0.0108 | 0.0003 | 1.0501 | n/a | n/a | 1.2391 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0127 | 0.0128 | 0.0125 | 0.0133 | 0.0003 | 1.0501 | n/a | n/a | 1.2391 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0186 | 0.0188 | 0.0185 | 0.0195 | 0.0004 | 0.2263 | n/a | n/a | 0.8653 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0046 | 0.0041 | 0.0060 | 0.0007 | 0.2263 | n/a | n/a | 0.8653 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0161 | 0.0161 | 0.0150 | 0.0169 | 0.0006 | 0.2263 | n/a | n/a | 0.8653 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0144 | 0.0148 | 0.0134 | 0.0163 | 0.0011 | 0.8557 | n/a | n/a | 1.1566 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0124 | 0.0125 | 0.0119 | 0.0134 | 0.0005 | 0.8557 | n/a | n/a | 1.1566 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0167 | 0.0173 | 0.0144 | 0.0210 | 0.0023 | 0.8557 | n/a | n/a | 1.1566 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0288 | 0.0296 | 0.0287 | 0.0312 | 0.0011 | 0.2663 | n/a | n/a | 1.1667 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0077 | 0.0078 | 0.0076 | 0.0081 | 0.0002 | 0.2663 | n/a | n/a | 1.1667 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0336 | 0.0318 | 0.0191 | 0.0382 | 0.0070 | 0.2663 | n/a | n/a | 1.1667 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0229 | 0.0234 | 0.0220 | 0.0249 | 0.0012 | 0.6530 | n/a | n/a | 0.5999 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0150 | 0.0172 | 0.0147 | 0.0235 | 0.0034 | 0.6530 | n/a | n/a | 0.5999 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0138 | 0.0155 | 0.0134 | 0.0214 | 0.0030 | 0.6530 | n/a | n/a | 0.5999 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0782 | 0.0783 | 0.0770 | 0.0801 | 0.0012 | 0.2908 | n/a | n/a | 0.8371 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0227 | 0.0231 | 0.0218 | 0.0259 | 0.0014 | 0.2908 | n/a | n/a | 0.8371 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0654 | 0.0657 | 0.0604 | 0.0741 | 0.0051 | 0.2908 | n/a | n/a | 0.8371 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0574 | 0.0587 | 0.0564 | 0.0628 | 0.0025 | 0.4073 | n/a | n/a | 0.4455 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0234 | 0.0235 | 0.0233 | 0.0239 | 0.0003 | 0.4073 | n/a | n/a | 0.4455 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0256 | 0.0249 | 0.0162 | 0.0380 | 0.0082 | 0.4073 | n/a | n/a | 0.4455 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0190 | 0.0195 | 0.0186 | 0.0216 | 0.0011 | 0.1362 | n/a | n/a | 0.6602 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0026 | 0.0030 | 0.0021 | 0.0042 | 0.0008 | 0.1362 | n/a | n/a | 0.6602 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0125 | 0.0124 | 0.0107 | 0.0144 | 0.0013 | 0.1362 | n/a | n/a | 0.6602 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1254 | 0.1319 | 0.1042 | 0.1667 | 0.0208 | 0.1551 | n/a | n/a | 0.2106 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0194 | 0.0199 | 0.0191 | 0.0211 | 0.0007 | 0.1551 | n/a | n/a | 0.2106 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0264 | 0.0384 | 0.0169 | 0.0915 | 0.0279 | 0.1551 | n/a | n/a | 0.2106 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3996 | 0.4031 | 0.3782 | 0.4359 | 0.0206 | 0.7419 | n/a | n/a | 0.3708 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2965 | 0.2942 | 0.2772 | 0.3131 | 0.0144 | 0.7419 | n/a | n/a | 0.3708 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1482 | 0.1470 | 0.1374 | 0.1533 | 0.0059 | 0.7419 | n/a | n/a | 0.3708 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.5442 | 2.5022 | 2.2974 | 2.6475 | 0.1336 | 0.1641 | n/a | n/a | 0.1127 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4174 | 0.4194 | 0.4035 | 0.4399 | 0.0118 | 0.1641 | n/a | n/a | 0.1127 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2867 | 0.2826 | 0.2566 | 0.3081 | 0.0196 | 0.1641 | n/a | n/a | 0.1127 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2048 | 0.2026 | 0.1888 | 0.2164 | 0.0106 | 6.9654 | n/a | n/a | 0.5814 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.4268 | 1.3852 | 1.2479 | 1.5141 | 0.1023 | 6.9654 | n/a | n/a | 0.5814 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1191 | 0.1197 | 0.1102 | 0.1335 | 0.0076 | 6.9654 | n/a | n/a | 0.5814 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 6.4851 | 6.5479 | 6.3371 | 6.7976 | 0.1631 | 10.1300 | n/a | n/a | 0.0561 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 65.6935 | 65.4174 | 60.4994 | 71.2823 | 3.7973 | 10.1300 | n/a | n/a | 0.0561 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3640 | 0.3617 | 0.3402 | 0.3783 | 0.0156 | 10.1300 | n/a | n/a | 0.0561 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0278 | 0.0279 | 0.0271 | 0.0291 | 0.0007 | 6.9988 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1944 | 0.2176 | 0.1677 | 0.3382 | 0.0615 | 6.9988 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0286 | 0.0285 | 0.0275 | 0.0295 | 0.0008 | 22.6805 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.6478 | 0.6320 | 0.5834 | 0.6610 | 0.0284 | 22.6805 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7380 | 0.7318 | 0.7039 | 0.7457 | 0.0156 | 10.6882 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.8879 | 8.5000 | 7.6999 | 10.3875 | 1.0105 | 10.6882 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6529 | 0.6580 | 0.6503 | 0.6734 | 0.0085 | 40.7899 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 26.6323 | 27.3687 | 26.4690 | 29.0299 | 1.0604 | 40.7899 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1164 | 0.1217 | 0.1149 | 0.1450 | 0.0117 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2930 | 0.2960 | 0.2916 | 0.3100 | 0.0070 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2696 | 0.2845 | 0.2474 | 0.3527 | 0.0365 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.6228 | 0.7485 | 0.5973 | 1.0104 | 0.1768 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1682 | 1.1712 | 1.1476 | 1.2067 | 0.0207 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 3.0411 | 3.0503 | 2.7251 | 3.4620 | 0.2943 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.2061 | 5.1833 | 4.7411 | 5.6082 | 0.3415 | 0.0301 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1566 | 0.1629 | 0.1563 | 0.1763 | 0.0083 | 0.0301 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
