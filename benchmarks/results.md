# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.7.0`
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
- TensorStudio wins versus JAX CPU dispatch: `45`
- TensorStudio losses versus JAX CPU dispatch: `53`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0162 | 0.0165 | 0.0160 | 0.0183 | 0.0009 | 0.0400 | n/a | n/a | 0.7717 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0007 | 0.0000 | 0.0400 | n/a | n/a | 0.7717 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0125 | 0.0125 | 0.0118 | 0.0133 | 0.0006 | 0.0400 | n/a | n/a | 0.7717 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0163 | 0.0165 | 0.0161 | 0.0171 | 0.0004 | 0.0408 | n/a | n/a | 0.7618 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0408 | n/a | n/a | 0.7618 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0124 | 0.0128 | 0.0121 | 0.0143 | 0.0009 | 0.0408 | n/a | n/a | 0.7618 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0165 | 0.0168 | 0.0160 | 0.0183 | 0.0009 | 0.0924 | n/a | n/a | 0.7160 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0015 | 0.0014 | 0.0008 | 0.0017 | 0.0003 | 0.0924 | n/a | n/a | 0.7160 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0118 | 0.0120 | 0.0118 | 0.0126 | 0.0003 | 0.0924 | n/a | n/a | 0.7160 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0206 | 0.0217 | 0.0159 | 0.0322 | 0.0056 | 0.0354 | n/a | n/a | 0.4208 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.0354 | n/a | n/a | 0.4208 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0087 | 0.0088 | 0.0086 | 0.0096 | 0.0004 | 0.0354 | n/a | n/a | 0.4208 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0862 | 0.0860 | 0.0847 | 0.0869 | 0.0008 | 0.0657 | n/a | n/a | 1.0682 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0057 | 0.0060 | 0.0055 | 0.0073 | 0.0007 | 0.0657 | n/a | n/a | 1.0682 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0920 | 0.0929 | 0.0906 | 0.0973 | 0.0024 | 0.0657 | n/a | n/a | 1.0682 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0160 | 0.0160 | 0.0159 | 0.0162 | 0.0001 | 0.0418 | n/a | n/a | 0.7365 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0006 | 0.0013 | 0.0002 | 0.0418 | n/a | n/a | 0.7365 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0118 | 0.0116 | 0.0111 | 0.0119 | 0.0003 | 0.0418 | n/a | n/a | 0.7365 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0161 | 0.0164 | 0.0159 | 0.0174 | 0.0005 | 0.0436 | n/a | n/a | 0.7130 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0436 | n/a | n/a | 0.7130 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0115 | 0.0115 | 0.0113 | 0.0116 | 0.0001 | 0.0436 | n/a | n/a | 0.7130 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0162 | 0.0164 | 0.0160 | 0.0174 | 0.0005 | 0.0442 | n/a | n/a | 0.7230 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0442 | n/a | n/a | 0.7230 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0117 | 0.0116 | 0.0114 | 0.0118 | 0.0002 | 0.0442 | n/a | n/a | 0.7230 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0198 | 0.0218 | 0.0157 | 0.0326 | 0.0064 | 0.1110 | n/a | n/a | 0.9406 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0022 | 0.0019 | 0.0011 | 0.0026 | 0.0006 | 0.1110 | n/a | n/a | 0.9406 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0186 | 0.0188 | 0.0153 | 0.0222 | 0.0022 | 0.1110 | n/a | n/a | 0.9406 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0904 | 0.0902 | 0.0882 | 0.0911 | 0.0010 | 0.0639 | n/a | n/a | 0.9850 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0058 | 0.0065 | 0.0055 | 0.0085 | 0.0011 | 0.0639 | n/a | n/a | 0.9850 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0891 | 0.0901 | 0.0883 | 0.0940 | 0.0020 | 0.0639 | n/a | n/a | 0.9850 | no | n/a | n/a | no | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0182 | 0.0214 | 0.0160 | 0.0324 | 0.0062 | 0.0372 | n/a | n/a | 0.6308 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0372 | n/a | n/a | 0.6308 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0115 | 0.0115 | 0.0114 | 0.0116 | 0.0001 | 0.0372 | n/a | n/a | 0.6308 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0160 | 0.0160 | 0.0159 | 0.0164 | 0.0002 | 0.0420 | n/a | n/a | 0.7107 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0420 | n/a | n/a | 0.7107 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0114 | 0.0115 | 0.0111 | 0.0121 | 0.0003 | 0.0420 | n/a | n/a | 0.7107 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0158 | 0.0158 | 0.0157 | 0.0159 | 0.0001 | 0.0434 | n/a | n/a | 0.7547 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0434 | n/a | n/a | 0.7547 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0119 | 0.0121 | 0.0119 | 0.0124 | 0.0002 | 0.0434 | n/a | n/a | 0.7547 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0160 | 0.0160 | 0.0159 | 0.0162 | 0.0001 | 0.0451 | n/a | n/a | 0.5535 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0451 | n/a | n/a | 0.5535 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0089 | 0.0090 | 0.0087 | 0.0095 | 0.0003 | 0.0451 | n/a | n/a | 0.5535 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0848 | 0.0849 | 0.0843 | 0.0859 | 0.0006 | 0.0675 | n/a | n/a | 1.0616 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0057 | 0.0059 | 0.0056 | 0.0068 | 0.0005 | 0.0675 | n/a | n/a | 1.0616 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0901 | 0.0916 | 0.0895 | 0.0956 | 0.0025 | 0.0675 | n/a | n/a | 1.0616 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0159 | 0.0159 | 0.0157 | 0.0161 | 0.0002 | 0.0412 | n/a | n/a | 0.7355 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0412 | n/a | n/a | 0.7355 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0117 | 0.0119 | 0.0114 | 0.0127 | 0.0005 | 0.0412 | n/a | n/a | 0.7355 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0159 | 0.0160 | 0.0158 | 0.0165 | 0.0003 | 0.0428 | n/a | n/a | 0.7321 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0428 | n/a | n/a | 0.7321 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0116 | 0.0117 | 0.0113 | 0.0124 | 0.0004 | 0.0428 | n/a | n/a | 0.7321 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0161 | 0.0160 | 0.0158 | 0.0161 | 0.0001 | 0.0434 | n/a | n/a | 0.7205 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0434 | n/a | n/a | 0.7205 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0116 | 0.0116 | 0.0115 | 0.0117 | 0.0001 | 0.0434 | n/a | n/a | 0.7205 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0160 | 0.0162 | 0.0159 | 0.0171 | 0.0005 | 0.0563 | n/a | n/a | 0.5410 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0009 | 0.0011 | 0.0007 | 0.0017 | 0.0004 | 0.0563 | n/a | n/a | 0.5410 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0087 | 0.0087 | 0.0086 | 0.0089 | 0.0001 | 0.0563 | n/a | n/a | 0.5410 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0867 | 0.0883 | 0.0860 | 0.0923 | 0.0024 | 0.0721 | n/a | n/a | 1.0641 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0062 | 0.0067 | 0.0057 | 0.0082 | 0.0009 | 0.0721 | n/a | n/a | 1.0641 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0923 | 0.0969 | 0.0873 | 0.1183 | 0.0110 | 0.0721 | n/a | n/a | 1.0641 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0163 | 0.0163 | 0.0161 | 0.0169 | 0.0003 | 0.0590 | n/a | n/a | 1.0228 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0010 | 0.0000 | 0.0590 | n/a | n/a | 1.0228 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0167 | 0.0165 | 0.0139 | 0.0178 | 0.0014 | 0.0590 | n/a | n/a | 1.0228 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0170 | 0.0170 | 0.0162 | 0.0179 | 0.0006 | 0.0560 | n/a | n/a | 0.8559 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0009 | 0.0010 | 0.0009 | 0.0010 | 0.0000 | 0.0560 | n/a | n/a | 0.8559 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0145 | 0.0153 | 0.0121 | 0.0201 | 0.0026 | 0.0560 | n/a | n/a | 0.8559 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0163 | 0.0163 | 0.0162 | 0.0165 | 0.0001 | 0.0623 | n/a | n/a | 0.9637 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0623 | n/a | n/a | 0.9637 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0157 | 0.0166 | 0.0151 | 0.0193 | 0.0017 | 0.0623 | n/a | n/a | 0.9637 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0163 | 0.0162 | 0.0160 | 0.0163 | 0.0001 | 0.0646 | n/a | n/a | 0.8483 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.0646 | n/a | n/a | 0.8483 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0138 | 0.0135 | 0.0127 | 0.0142 | 0.0006 | 0.0646 | n/a | n/a | 0.8483 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0900 | 0.0897 | 0.0860 | 0.0923 | 0.0020 | 0.0837 | n/a | n/a | 1.2814 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0075 | 0.0075 | 0.0073 | 0.0076 | 0.0001 | 0.0837 | n/a | n/a | 1.2814 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1153 | 0.1186 | 0.1136 | 0.1289 | 0.0057 | 0.0837 | n/a | n/a | 1.2814 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0174 | 0.0177 | 0.0171 | 0.0189 | 0.0006 | 0.0864 | n/a | n/a | 1.1428 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.0864 | n/a | n/a | 1.1428 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0199 | 0.0185 | 0.0151 | 0.0217 | 0.0027 | 0.0864 | n/a | n/a | 1.1428 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0173 | 0.0174 | 0.0172 | 0.0178 | 0.0002 | 0.0829 | n/a | n/a | 1.0807 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.0829 | n/a | n/a | 1.0807 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0187 | 0.0193 | 0.0179 | 0.0230 | 0.0018 | 0.0829 | n/a | n/a | 1.0807 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0178 | 0.0178 | 0.0171 | 0.0189 | 0.0006 | 0.0834 | n/a | n/a | 0.9009 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | 0.0834 | n/a | n/a | 0.9009 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0160 | 0.0175 | 0.0143 | 0.0237 | 0.0033 | 0.0834 | n/a | n/a | 0.9009 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0178 | 0.0177 | 0.0174 | 0.0180 | 0.0002 | 0.0993 | n/a | n/a | 0.7932 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0018 | 0.0019 | 0.0017 | 0.0022 | 0.0002 | 0.0993 | n/a | n/a | 0.7932 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0141 | 0.0140 | 0.0109 | 0.0173 | 0.0022 | 0.0993 | n/a | n/a | 0.7932 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0919 | 0.0915 | 0.0902 | 0.0925 | 0.0009 | 0.1198 | n/a | n/a | 1.4345 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0110 | 0.0110 | 0.0101 | 0.0116 | 0.0005 | 0.1198 | n/a | n/a | 1.4345 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1318 | 0.1322 | 0.1241 | 0.1375 | 0.0050 | 0.1198 | n/a | n/a | 1.4345 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0257 | 0.0257 | 0.0211 | 0.0318 | 0.0036 | 0.1477 | n/a | n/a | 1.0181 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0039 | 0.0001 | 0.1477 | n/a | n/a | 1.0181 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0261 | 0.0240 | 0.0162 | 0.0268 | 0.0040 | 0.1477 | n/a | n/a | 1.0181 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0209 | 0.0215 | 0.0204 | 0.0243 | 0.0014 | 0.2000 | n/a | n/a | 0.8343 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0042 | 0.0042 | 0.0040 | 0.0043 | 0.0001 | 0.2000 | n/a | n/a | 0.8343 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0174 | 0.0209 | 0.0132 | 0.0321 | 0.0073 | 0.2000 | n/a | n/a | 0.8343 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0213 | 0.0252 | 0.0207 | 0.0329 | 0.0054 | 0.3369 | n/a | n/a | 0.7262 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0072 | 0.0073 | 0.0070 | 0.0078 | 0.0003 | 0.3369 | n/a | n/a | 0.7262 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0155 | 0.0187 | 0.0139 | 0.0287 | 0.0057 | 0.3369 | n/a | n/a | 0.7262 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0212 | 0.0213 | 0.0207 | 0.0221 | 0.0005 | 0.1948 | n/a | n/a | 0.9532 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0041 | 0.0042 | 0.0041 | 0.0043 | 0.0001 | 0.1948 | n/a | n/a | 0.9532 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0202 | 0.0195 | 0.0148 | 0.0218 | 0.0025 | 0.1948 | n/a | n/a | 0.9532 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1082 | 0.1118 | 0.1043 | 0.1221 | 0.0070 | 0.3175 | n/a | n/a | 1.6711 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0343 | 0.0311 | 0.0215 | 0.0391 | 0.0080 | 0.3175 | n/a | n/a | 1.6711 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1808 | 0.1775 | 0.1387 | 0.2222 | 0.0318 | 0.3175 | n/a | n/a | 1.6711 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0152 | 0.0153 | 0.0148 | 0.0161 | 0.0005 | 0.1132 | n/a | n/a | 2.9786 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.1132 | n/a | n/a | 2.9786 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0453 | 0.0473 | 0.0442 | 0.0520 | 0.0031 | 0.1132 | n/a | n/a | 2.9786 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0148 | 0.0148 | 0.0148 | 0.0149 | 0.0000 | 0.2936 | n/a | n/a | 5.0408 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0043 | 0.0043 | 0.0042 | 0.0044 | 0.0001 | 0.2936 | n/a | n/a | 5.0408 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0746 | 0.0748 | 0.0723 | 0.0784 | 0.0023 | 0.2936 | n/a | n/a | 5.0408 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0150 | 0.0171 | 0.0147 | 0.0210 | 0.0028 | 0.0795 | n/a | n/a | 2.0633 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0001 | 0.0795 | n/a | n/a | 2.0633 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0310 | 0.0321 | 0.0300 | 0.0380 | 0.0030 | 0.0795 | n/a | n/a | 2.0633 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0151 | 0.0152 | 0.0149 | 0.0158 | 0.0003 | 0.0841 | n/a | n/a | 1.9661 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0001 | 0.0841 | n/a | n/a | 1.9661 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0297 | 0.0301 | 0.0296 | 0.0315 | 0.0007 | 0.0841 | n/a | n/a | 1.9661 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0157 | 0.0156 | 0.0152 | 0.0161 | 0.0003 | 0.0810 | n/a | n/a | 1.9117 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.0810 | n/a | n/a | 1.9117 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0300 | 0.0301 | 0.0296 | 0.0310 | 0.0005 | 0.0810 | n/a | n/a | 1.9117 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0149 | 0.0151 | 0.0147 | 0.0162 | 0.0005 | 0.1222 | n/a | n/a | 3.7567 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.1222 | n/a | n/a | 3.7567 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0559 | 0.0586 | 0.0504 | 0.0708 | 0.0073 | 0.1222 | n/a | n/a | 3.7567 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0154 | 0.0163 | 0.0152 | 0.0196 | 0.0017 | 0.4339 | n/a | n/a | 4.8860 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0067 | 0.0070 | 0.0049 | 0.0109 | 0.0022 | 0.4339 | n/a | n/a | 4.8860 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0755 | 0.0775 | 0.0734 | 0.0820 | 0.0035 | 0.4339 | n/a | n/a | 4.8860 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0150 | 0.0151 | 0.0149 | 0.0154 | 0.0002 | 0.0812 | n/a | n/a | 2.0275 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0001 | 0.0812 | n/a | n/a | 2.0275 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0304 | 0.0307 | 0.0296 | 0.0321 | 0.0008 | 0.0812 | n/a | n/a | 2.0275 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0149 | 0.0149 | 0.0149 | 0.0150 | 0.0000 | 0.0831 | n/a | n/a | 1.9619 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.0831 | n/a | n/a | 1.9619 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0293 | 0.0295 | 0.0289 | 0.0308 | 0.0007 | 0.0831 | n/a | n/a | 1.9619 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0150 | 0.0152 | 0.0148 | 0.0158 | 0.0004 | 0.0860 | n/a | n/a | 1.9773 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.0860 | n/a | n/a | 1.9773 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0297 | 0.0299 | 0.0294 | 0.0308 | 0.0005 | 0.0860 | n/a | n/a | 1.9773 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0151 | 0.0001 | 0.1138 | n/a | n/a | 3.0152 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.1138 | n/a | n/a | 3.0152 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0453 | 0.0458 | 0.0444 | 0.0492 | 0.0017 | 0.1138 | n/a | n/a | 3.0152 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0152 | 0.0153 | 0.0150 | 0.0155 | 0.0002 | 0.2909 | n/a | n/a | 4.7701 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0044 | 0.0046 | 0.0043 | 0.0054 | 0.0004 | 0.2909 | n/a | n/a | 4.7701 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0727 | 0.0733 | 0.0719 | 0.0766 | 0.0017 | 0.2909 | n/a | n/a | 4.7701 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0153 | 0.0153 | 0.0151 | 0.0157 | 0.0002 | 0.0845 | n/a | n/a | 1.9250 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0018 | 0.0002 | 0.0845 | n/a | n/a | 1.9250 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0294 | 0.0301 | 0.0292 | 0.0324 | 0.0012 | 0.0845 | n/a | n/a | 1.9250 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0150 | 0.0001 | 0.0835 | n/a | n/a | 2.0132 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.0835 | n/a | n/a | 2.0132 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0302 | 0.0303 | 0.0299 | 0.0314 | 0.0005 | 0.0835 | n/a | n/a | 2.0132 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0152 | 0.0156 | 0.0150 | 0.0172 | 0.0008 | 0.0862 | n/a | n/a | 1.9965 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0001 | 0.0862 | n/a | n/a | 1.9965 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0303 | 0.0308 | 0.0298 | 0.0321 | 0.0009 | 0.0862 | n/a | n/a | 1.9965 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0148 | 0.0001 | 0.1244 | n/a | n/a | 3.0798 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0018 | 0.0019 | 0.0017 | 0.0020 | 0.0001 | 0.1244 | n/a | n/a | 3.0798 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0456 | 0.0463 | 0.0447 | 0.0486 | 0.0015 | 0.1244 | n/a | n/a | 3.0798 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0157 | 0.0159 | 0.0156 | 0.0168 | 0.0005 | 0.2939 | n/a | n/a | 4.7010 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0046 | 0.0046 | 0.0045 | 0.0047 | 0.0000 | 0.2939 | n/a | n/a | 4.7010 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0740 | 0.0736 | 0.0715 | 0.0759 | 0.0015 | 0.2939 | n/a | n/a | 4.7010 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0161 | 0.0161 | 0.0159 | 0.0163 | 0.0001 | 0.0957 | n/a | n/a | 1.8519 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.0957 | n/a | n/a | 1.8519 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0298 | 0.0306 | 0.0294 | 0.0342 | 0.0018 | 0.0957 | n/a | n/a | 1.8519 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0155 | 0.0155 | 0.0153 | 0.0156 | 0.0001 | 0.0875 | n/a | n/a | 1.9069 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.0875 | n/a | n/a | 1.9069 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0296 | 0.0304 | 0.0293 | 0.0329 | 0.0013 | 0.0875 | n/a | n/a | 1.9069 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0157 | 0.0157 | 0.0156 | 0.0157 | 0.0000 | 0.0991 | n/a | n/a | 1.9003 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0016 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.0991 | n/a | n/a | 1.9003 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0298 | 0.0296 | 0.0290 | 0.0299 | 0.0003 | 0.0991 | n/a | n/a | 1.9003 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0152 | 0.0153 | 0.0149 | 0.0158 | 0.0003 | 0.1505 | n/a | n/a | 4.4873 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.1505 | n/a | n/a | 4.4873 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0683 | 0.0682 | 0.0588 | 0.0733 | 0.0051 | 0.1505 | n/a | n/a | 4.4873 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0203 | 0.0208 | 0.0196 | 0.0239 | 0.0015 | 0.3370 | n/a | n/a | 5.0696 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0068 | 0.0071 | 0.0067 | 0.0077 | 0.0004 | 0.3370 | n/a | n/a | 5.0696 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1029 | 0.1023 | 0.0965 | 0.1067 | 0.0033 | 0.3370 | n/a | n/a | 5.0696 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0260 | 0.0260 | 0.0257 | 0.0262 | 0.0002 | 0.1472 | n/a | n/a | 1.1481 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0038 | 0.0041 | 0.0038 | 0.0050 | 0.0005 | 0.1472 | n/a | n/a | 1.1481 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0298 | 0.0303 | 0.0296 | 0.0319 | 0.0009 | 0.1472 | n/a | n/a | 1.1481 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0195 | 0.0215 | 0.0193 | 0.0260 | 0.0028 | 0.1310 | n/a | n/a | 2.7873 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0025 | 0.0027 | 0.0025 | 0.0030 | 0.0002 | 0.1310 | n/a | n/a | 2.7873 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0542 | 0.0494 | 0.0298 | 0.0731 | 0.0160 | 0.1310 | n/a | n/a | 2.7873 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0208 | 0.0208 | 0.0203 | 0.0215 | 0.0004 | 0.1517 | n/a | n/a | 1.4854 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0032 | 0.0032 | 0.0030 | 0.0033 | 0.0001 | 0.1517 | n/a | n/a | 1.4854 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0310 | 0.0309 | 0.0301 | 0.0320 | 0.0006 | 0.1517 | n/a | n/a | 1.4854 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0166 | 0.0168 | 0.0164 | 0.0173 | 0.0003 | 0.2280 | n/a | n/a | 4.1518 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0038 | 0.0038 | 0.0036 | 0.0040 | 0.0001 | 0.2280 | n/a | n/a | 4.1518 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0690 | 0.0724 | 0.0624 | 0.0833 | 0.0077 | 0.2280 | n/a | n/a | 4.1518 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0337 | 0.0337 | 0.0333 | 0.0340 | 0.0003 | 0.3649 | n/a | n/a | 3.1018 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0123 | 0.0130 | 0.0119 | 0.0159 | 0.0015 | 0.3649 | n/a | n/a | 3.1018 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1044 | 0.1042 | 0.0953 | 0.1145 | 0.0063 | 0.3649 | n/a | n/a | 3.1018 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0637 | 0.0653 | 0.0633 | 0.0684 | 0.0023 | 0.1636 | n/a | n/a | 0.5737 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0104 | 0.0108 | 0.0103 | 0.0127 | 0.0009 | 0.1636 | n/a | n/a | 0.5737 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0366 | 0.0366 | 0.0334 | 0.0399 | 0.0029 | 0.1636 | n/a | n/a | 0.5737 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0329 | 0.0328 | 0.0309 | 0.0347 | 0.0014 | 0.1995 | n/a | n/a | 1.0309 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0066 | 0.0068 | 0.0062 | 0.0079 | 0.0006 | 0.1995 | n/a | n/a | 1.0309 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0339 | 0.0341 | 0.0339 | 0.0344 | 0.0002 | 0.1995 | n/a | n/a | 1.0309 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0377 | 0.0380 | 0.0372 | 0.0396 | 0.0008 | 0.1988 | n/a | n/a | 0.9109 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0075 | 0.0076 | 0.0073 | 0.0083 | 0.0004 | 0.1988 | n/a | n/a | 0.9109 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0343 | 0.0349 | 0.0337 | 0.0376 | 0.0015 | 0.1988 | n/a | n/a | 0.9109 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0216 | 0.0218 | 0.0213 | 0.0226 | 0.0005 | 0.4013 | n/a | n/a | 4.1407 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0087 | 0.0088 | 0.0086 | 0.0093 | 0.0003 | 0.4013 | n/a | n/a | 4.1407 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0893 | 0.0946 | 0.0787 | 0.1209 | 0.0153 | 0.4013 | n/a | n/a | 4.1407 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0914 | 0.0918 | 0.0893 | 0.0950 | 0.0021 | 0.3694 | n/a | n/a | 1.3697 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0338 | 0.0340 | 0.0331 | 0.0351 | 0.0007 | 0.3694 | n/a | n/a | 1.3697 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1252 | 0.1217 | 0.0981 | 0.1380 | 0.0144 | 0.3694 | n/a | n/a | 1.3697 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2145 | 0.2154 | 0.2122 | 0.2191 | 0.0026 | 0.1894 | n/a | n/a | 0.1766 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0406 | 0.0426 | 0.0379 | 0.0525 | 0.0055 | 0.1894 | n/a | n/a | 0.1766 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0379 | 0.0382 | 0.0371 | 0.0399 | 0.0010 | 0.1894 | n/a | n/a | 0.1766 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0776 | 0.0779 | 0.0769 | 0.0791 | 0.0008 | 0.2495 | n/a | n/a | 0.5318 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0194 | 0.0193 | 0.0186 | 0.0197 | 0.0004 | 0.2495 | n/a | n/a | 0.5318 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0413 | 0.0411 | 0.0402 | 0.0419 | 0.0008 | 0.2495 | n/a | n/a | 0.5318 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1033 | 0.1054 | 0.1005 | 0.1108 | 0.0042 | 0.3756 | n/a | n/a | 0.4115 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0388 | 0.0368 | 0.0265 | 0.0411 | 0.0052 | 0.3756 | n/a | n/a | 0.4115 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0425 | 0.0429 | 0.0414 | 0.0454 | 0.0013 | 0.3756 | n/a | n/a | 0.4115 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0158 | 0.0173 | 0.0155 | 0.0208 | 0.0021 | 0.1444 | n/a | n/a | 0.7495 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0027 | 0.0002 | 0.1444 | n/a | n/a | 0.7495 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0118 | 0.0127 | 0.0116 | 0.0157 | 0.0015 | 0.1444 | n/a | n/a | 0.7495 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0159 | 0.0159 | 0.0156 | 0.0163 | 0.0003 | 0.5052 | n/a | n/a | 0.6866 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0080 | 0.0081 | 0.0079 | 0.0084 | 0.0002 | 0.5052 | n/a | n/a | 0.6866 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0109 | 0.0110 | 0.0108 | 0.0114 | 0.0002 | 0.5052 | n/a | n/a | 0.6866 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0156 | 0.0155 | 0.0154 | 0.0156 | 0.0001 | 0.1555 | n/a | n/a | 0.7812 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0024 | 0.0024 | 0.0021 | 0.0028 | 0.0003 | 0.1555 | n/a | n/a | 0.7812 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0118 | 0.0125 | 0.0002 | 0.1555 | n/a | n/a | 0.7812 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0159 | 0.0160 | 0.0157 | 0.0167 | 0.0003 | 0.5111 | n/a | n/a | 0.8256 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0081 | 0.0080 | 0.0078 | 0.0082 | 0.0002 | 0.5111 | n/a | n/a | 0.8256 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0131 | 0.0129 | 0.0115 | 0.0145 | 0.0011 | 0.5111 | n/a | n/a | 0.8256 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0156 | 0.0164 | 0.0154 | 0.0187 | 0.0012 | 0.1431 | n/a | n/a | 0.7675 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0001 | 0.1431 | n/a | n/a | 0.7675 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0120 | 0.0121 | 0.0118 | 0.0126 | 0.0003 | 0.1431 | n/a | n/a | 0.7675 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0177 | 0.0181 | 0.0158 | 0.0215 | 0.0021 | 0.4715 | n/a | n/a | 0.6400 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0083 | 0.0088 | 0.0080 | 0.0109 | 0.0011 | 0.4715 | n/a | n/a | 0.6400 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0113 | 0.0115 | 0.0112 | 0.0120 | 0.0003 | 0.4715 | n/a | n/a | 0.6400 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0156 | 0.0156 | 0.0155 | 0.0158 | 0.0001 | 0.1393 | n/a | n/a | 0.8506 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0025 | 0.0001 | 0.1393 | n/a | n/a | 0.8506 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0133 | 0.0134 | 0.0131 | 0.0138 | 0.0003 | 0.1393 | n/a | n/a | 0.8506 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0157 | 0.0157 | 0.0155 | 0.0158 | 0.0001 | 0.5119 | n/a | n/a | 0.8270 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0080 | 0.0080 | 0.0079 | 0.0080 | 0.0001 | 0.5119 | n/a | n/a | 0.8270 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0129 | 0.0130 | 0.0127 | 0.0131 | 0.0001 | 0.5119 | n/a | n/a | 0.8270 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0167 | 0.0167 | 0.0164 | 0.0170 | 0.0002 | 0.1521 | n/a | n/a | 1.2139 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0025 | 0.0026 | 0.0025 | 0.0026 | 0.0001 | 0.1521 | n/a | n/a | 1.2139 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0202 | 0.0191 | 0.0157 | 0.0211 | 0.0020 | 0.1521 | n/a | n/a | 1.2139 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0171 | 0.0176 | 0.0166 | 0.0193 | 0.0010 | 0.4950 | n/a | n/a | 1.0192 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0085 | 0.0087 | 0.0083 | 0.0094 | 0.0004 | 0.4950 | n/a | n/a | 1.0192 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0174 | 0.0171 | 0.0156 | 0.0184 | 0.0010 | 0.4950 | n/a | n/a | 1.0192 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0201 | 0.0205 | 0.0194 | 0.0231 | 0.0013 | 0.1807 | n/a | n/a | 0.9647 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0036 | 0.0038 | 0.0035 | 0.0042 | 0.0003 | 0.1807 | n/a | n/a | 0.9647 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0194 | 0.0198 | 0.0154 | 0.0236 | 0.0028 | 0.1807 | n/a | n/a | 0.9647 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0205 | 0.0216 | 0.0198 | 0.0236 | 0.0016 | 0.6062 | n/a | n/a | 0.7579 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0124 | 0.0141 | 0.0115 | 0.0184 | 0.0027 | 0.6062 | n/a | n/a | 0.7579 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0155 | 0.0159 | 0.0152 | 0.0169 | 0.0007 | 0.6062 | n/a | n/a | 0.7579 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0340 | 0.0335 | 0.0323 | 0.0344 | 0.0009 | 0.2116 | n/a | n/a | 0.7880 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0072 | 0.0077 | 0.0069 | 0.0091 | 0.0009 | 0.2116 | n/a | n/a | 0.7880 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0268 | 0.0304 | 0.0229 | 0.0495 | 0.0097 | 0.2116 | n/a | n/a | 0.7880 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0335 | 0.0343 | 0.0324 | 0.0371 | 0.0018 | 0.3951 | n/a | n/a | 0.9769 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0133 | 0.0134 | 0.0130 | 0.0140 | 0.0003 | 0.3951 | n/a | n/a | 0.9769 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0328 | 0.0319 | 0.0219 | 0.0435 | 0.0080 | 0.3951 | n/a | n/a | 0.9769 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0163 | 0.0162 | 0.0158 | 0.0164 | 0.0002 | 0.1901 | n/a | n/a | 0.7536 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0031 | 0.0032 | 0.0028 | 0.0039 | 0.0004 | 0.1901 | n/a | n/a | 0.7536 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0123 | 0.0133 | 0.0118 | 0.0170 | 0.0019 | 0.1901 | n/a | n/a | 0.7536 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0102 | 0.0103 | 0.0102 | 0.0106 | 0.0002 | 1.0355 | n/a | n/a | 1.1283 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0106 | 0.0105 | 0.0099 | 0.0110 | 0.0004 | 1.0355 | n/a | n/a | 1.1283 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0115 | 0.0120 | 0.0112 | 0.0132 | 0.0008 | 1.0355 | n/a | n/a | 1.1283 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0197 | 0.0201 | 0.0187 | 0.0232 | 0.0016 | 0.2109 | n/a | n/a | 0.9878 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0043 | 0.0039 | 0.0054 | 0.0005 | 0.2109 | n/a | n/a | 0.9878 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0195 | 0.0206 | 0.0174 | 0.0244 | 0.0028 | 0.2109 | n/a | n/a | 0.9878 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0135 | 0.0135 | 0.0130 | 0.0140 | 0.0003 | 0.8920 | n/a | n/a | 1.4188 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0121 | 0.0120 | 0.0118 | 0.0122 | 0.0001 | 0.8920 | n/a | n/a | 1.4188 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0192 | 0.0187 | 0.0132 | 0.0232 | 0.0032 | 0.8920 | n/a | n/a | 1.4188 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0297 | 0.0330 | 0.0293 | 0.0414 | 0.0047 | 0.2566 | n/a | n/a | 1.1091 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0076 | 0.0076 | 0.0075 | 0.0077 | 0.0001 | 0.2566 | n/a | n/a | 1.1091 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0330 | 0.0330 | 0.0196 | 0.0533 | 0.0114 | 0.2566 | n/a | n/a | 1.1091 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0227 | 0.0228 | 0.0225 | 0.0235 | 0.0004 | 0.6759 | n/a | n/a | 0.7686 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0153 | 0.0159 | 0.0145 | 0.0182 | 0.0013 | 0.6759 | n/a | n/a | 0.7686 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0174 | 0.0191 | 0.0140 | 0.0266 | 0.0050 | 0.6759 | n/a | n/a | 0.7686 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0815 | 0.0801 | 0.0776 | 0.0819 | 0.0019 | 0.2645 | n/a | n/a | 0.8579 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0216 | 0.0219 | 0.0213 | 0.0234 | 0.0008 | 0.2645 | n/a | n/a | 0.8579 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0699 | 0.0718 | 0.0656 | 0.0831 | 0.0064 | 0.2645 | n/a | n/a | 0.8579 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0583 | 0.0598 | 0.0572 | 0.0664 | 0.0034 | 0.3966 | n/a | n/a | 0.4630 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0231 | 0.0241 | 0.0230 | 0.0271 | 0.0016 | 0.3966 | n/a | n/a | 0.4630 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0270 | 0.0261 | 0.0159 | 0.0412 | 0.0091 | 0.3966 | n/a | n/a | 0.4630 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0180 | 0.0183 | 0.0179 | 0.0193 | 0.0005 | 0.1067 | n/a | n/a | 0.7088 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0020 | 0.0019 | 0.0021 | 0.0001 | 0.1067 | n/a | n/a | 0.7088 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0128 | 0.0164 | 0.0113 | 0.0271 | 0.0061 | 0.1067 | n/a | n/a | 0.7088 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1050 | 0.1071 | 0.0962 | 0.1252 | 0.0099 | 0.1022 | n/a | n/a | 0.2566 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0107 | 0.0111 | 0.0107 | 0.0120 | 0.0005 | 0.1022 | n/a | n/a | 0.2566 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0269 | 0.0430 | 0.0156 | 0.0767 | 0.0248 | 0.1022 | n/a | n/a | 0.2566 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3909 | 0.3893 | 0.3782 | 0.4006 | 0.0079 | 0.9373 | n/a | n/a | 0.3221 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3663 | 0.3736 | 0.3502 | 0.4032 | 0.0195 | 0.9373 | n/a | n/a | 0.3221 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1259 | 0.1483 | 0.0990 | 0.2688 | 0.0617 | 0.9373 | n/a | n/a | 0.3221 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.2252 | 2.1348 | 1.9338 | 2.2832 | 0.1546 | 0.2136 | n/a | n/a | 0.1236 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4754 | 0.4652 | 0.4402 | 0.4907 | 0.0198 | 0.2136 | n/a | n/a | 0.1236 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2750 | 0.2764 | 0.2393 | 0.3240 | 0.0289 | 0.2136 | n/a | n/a | 0.1236 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1915 | 0.1936 | 0.1892 | 0.1991 | 0.0037 | 6.7064 | n/a | n/a | 0.5646 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2845 | 1.2893 | 1.2485 | 1.3224 | 0.0275 | 6.7064 | n/a | n/a | 0.5646 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1081 | 0.1104 | 0.0972 | 0.1336 | 0.0126 | 6.7064 | n/a | n/a | 0.5646 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 6.5326 | 6.4987 | 6.2352 | 6.7116 | 0.1951 | 9.6046 | n/a | n/a | 0.0515 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 62.7432 | 62.8729 | 61.6382 | 64.6830 | 0.9993 | 9.6046 | n/a | n/a | 0.0515 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3367 | 0.3441 | 0.3342 | 0.3674 | 0.0123 | 9.6046 | n/a | n/a | 0.0515 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0277 | 0.0276 | 0.0273 | 0.0279 | 0.0002 | 5.7024 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1577 | 0.1720 | 0.1539 | 0.2343 | 0.0312 | 5.7024 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0278 | 0.0279 | 0.0278 | 0.0282 | 0.0001 | 19.8919 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5538 | 0.5569 | 0.5479 | 0.5711 | 0.0078 | 19.8919 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7240 | 0.7820 | 0.7142 | 1.0108 | 0.1147 | 10.4443 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.5613 | 7.6415 | 7.5322 | 7.9480 | 0.1576 | 10.4443 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7131 | 0.7041 | 0.6786 | 0.7313 | 0.0199 | 39.5038 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 28.1714 | 28.6007 | 27.4622 | 30.7085 | 1.1172 | 39.5038 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1183 | 0.1182 | 0.1163 | 0.1198 | 0.0012 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.3012 | 0.3105 | 0.3004 | 0.3349 | 0.0134 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2628 | 0.2883 | 0.2510 | 0.3816 | 0.0479 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.6357 | 0.6646 | 0.6111 | 0.7459 | 0.0536 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.2122 | 1.2135 | 1.1481 | 1.2535 | 0.0382 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7344 | 2.7431 | 2.7156 | 2.7863 | 0.0248 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.2573 | 5.3420 | 5.0627 | 5.6072 | 0.2042 | 0.0294 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1544 | 0.1579 | 0.1512 | 0.1702 | 0.0069 | 0.0294 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
