# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.14.0`
- TensorStudio threads: `12`
- TensorStudio BLAS enabled: `False`
- TensorStudio SIMD level: `sse2`
- TensorStudio storage pool enabled: `True`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `6`
- TensorStudio losses versus NumPy: `97`
- TensorStudio wins versus JAX CPU dispatch: `38`
- TensorStudio losses versus JAX CPU dispatch: `60`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0167 | 0.0174 | 0.0165 | 0.0199 | 0.0013 | 0.0426 | n/a | n/a | 0.7285 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0426 | n/a | n/a | 0.7285 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0122 | 0.0156 | 0.0110 | 0.0281 | 0.0064 | 0.0426 | n/a | n/a | 0.7285 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0183 | 0.0176 | 0.0162 | 0.0187 | 0.0011 | 0.0387 | n/a | n/a | 0.6260 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0387 | n/a | n/a | 0.6260 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0114 | 0.0115 | 0.0111 | 0.0119 | 0.0003 | 0.0387 | n/a | n/a | 0.6260 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0171 | 0.0191 | 0.0156 | 0.0277 | 0.0045 | 0.0593 | n/a | n/a | 0.6856 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0010 | 0.0013 | 0.0007 | 0.0019 | 0.0004 | 0.0593 | n/a | n/a | 0.6856 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0117 | 0.0119 | 0.0114 | 0.0124 | 0.0004 | 0.0593 | n/a | n/a | 0.6856 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0159 | 0.0159 | 0.0157 | 0.0162 | 0.0002 | 0.0442 | n/a | n/a | 0.5554 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0013 | 0.0003 | 0.0442 | n/a | n/a | 0.5554 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0088 | 0.0089 | 0.0086 | 0.0094 | 0.0003 | 0.0442 | n/a | n/a | 0.5554 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0853 | 0.0856 | 0.0846 | 0.0864 | 0.0007 | 0.0627 | n/a | n/a | 1.0394 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0054 | 0.0055 | 0.0053 | 0.0060 | 0.0002 | 0.0627 | n/a | n/a | 1.0394 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0887 | 0.0888 | 0.0877 | 0.0899 | 0.0008 | 0.0627 | n/a | n/a | 1.0394 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0164 | 0.0168 | 0.0157 | 0.0180 | 0.0009 | 0.0413 | n/a | n/a | 0.8821 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0011 | 0.0002 | 0.0413 | n/a | n/a | 0.8821 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0144 | 0.0152 | 0.0113 | 0.0207 | 0.0032 | 0.0413 | n/a | n/a | 0.8821 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0160 | 0.0001 | 0.0430 | n/a | n/a | 0.7269 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0430 | n/a | n/a | 0.7269 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0115 | 0.0116 | 0.0113 | 0.0121 | 0.0003 | 0.0430 | n/a | n/a | 0.7269 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0159 | 0.0159 | 0.0156 | 0.0162 | 0.0002 | 0.0533 | n/a | n/a | 0.7236 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0533 | n/a | n/a | 0.7236 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0115 | 0.0117 | 0.0114 | 0.0120 | 0.0003 | 0.0533 | n/a | n/a | 0.7236 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0160 | 0.0160 | 0.0158 | 0.0161 | 0.0001 | 0.0475 | n/a | n/a | 0.5983 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.0475 | n/a | n/a | 0.5983 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0096 | 0.0096 | 0.0091 | 0.0100 | 0.0003 | 0.0475 | n/a | n/a | 0.5983 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0844 | 0.0845 | 0.0839 | 0.0852 | 0.0005 | 0.0665 | n/a | n/a | 1.0883 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0056 | 0.0059 | 0.0053 | 0.0066 | 0.0006 | 0.0665 | n/a | n/a | 1.0883 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0918 | 0.0993 | 0.0889 | 0.1309 | 0.0159 | 0.0665 | n/a | n/a | 1.0883 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0159 | 0.0159 | 0.0157 | 0.0163 | 0.0002 | 0.0436 | n/a | n/a | 0.7259 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0006 | 0.0011 | 0.0002 | 0.0436 | n/a | n/a | 0.7259 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0115 | 0.0120 | 0.0114 | 0.0138 | 0.0009 | 0.0436 | n/a | n/a | 0.7259 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0161 | 0.0162 | 0.0159 | 0.0166 | 0.0002 | 0.0481 | n/a | n/a | 0.7139 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0008 | 0.0009 | 0.0007 | 0.0013 | 0.0002 | 0.0481 | n/a | n/a | 0.7139 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0115 | 0.0116 | 0.0113 | 0.0121 | 0.0003 | 0.0481 | n/a | n/a | 0.7139 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0157 | 0.0157 | 0.0156 | 0.0158 | 0.0001 | 0.0448 | n/a | n/a | 0.7557 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0448 | n/a | n/a | 0.7557 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0119 | 0.0121 | 0.0112 | 0.0138 | 0.0009 | 0.0448 | n/a | n/a | 0.7557 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0159 | 0.0162 | 0.0158 | 0.0173 | 0.0005 | 0.0446 | n/a | n/a | 0.5748 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0446 | n/a | n/a | 0.5748 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0091 | 0.0092 | 0.0089 | 0.0094 | 0.0002 | 0.0446 | n/a | n/a | 0.5748 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0852 | 0.0877 | 0.0846 | 0.0962 | 0.0044 | 0.0677 | n/a | n/a | 1.0556 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0058 | 0.0058 | 0.0054 | 0.0064 | 0.0003 | 0.0677 | n/a | n/a | 1.0556 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0900 | 0.0959 | 0.0891 | 0.1184 | 0.0113 | 0.0677 | n/a | n/a | 1.0556 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0159 | 0.0159 | 0.0158 | 0.0161 | 0.0001 | 0.0427 | n/a | n/a | 0.7299 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0010 | 0.0001 | 0.0427 | n/a | n/a | 0.7299 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0116 | 0.0116 | 0.0112 | 0.0121 | 0.0003 | 0.0427 | n/a | n/a | 0.7299 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0161 | 0.0162 | 0.0159 | 0.0166 | 0.0002 | 0.0423 | n/a | n/a | 0.7186 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0423 | n/a | n/a | 0.7186 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0116 | 0.0118 | 0.0112 | 0.0129 | 0.0006 | 0.0423 | n/a | n/a | 0.7186 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0162 | 0.0163 | 0.0158 | 0.0170 | 0.0004 | 0.0430 | n/a | n/a | 0.7214 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0430 | n/a | n/a | 0.7214 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0117 | 0.0118 | 0.0114 | 0.0126 | 0.0004 | 0.0430 | n/a | n/a | 0.7214 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0165 | 0.0165 | 0.0162 | 0.0169 | 0.0002 | 0.0451 | n/a | n/a | 0.5472 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0451 | n/a | n/a | 0.5472 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0090 | 0.0091 | 0.0089 | 0.0098 | 0.0003 | 0.0451 | n/a | n/a | 0.5472 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0856 | 0.0899 | 0.0846 | 0.1037 | 0.0072 | 0.0642 | n/a | n/a | 1.0463 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0055 | 0.0056 | 0.0054 | 0.0059 | 0.0002 | 0.0642 | n/a | n/a | 1.0463 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0896 | 0.0893 | 0.0880 | 0.0903 | 0.0008 | 0.0642 | n/a | n/a | 1.0463 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0161 | 0.0161 | 0.0160 | 0.0163 | 0.0001 | 0.0629 | n/a | n/a | 0.8069 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0015 | 0.0010 | 0.0022 | 0.0006 | 0.0629 | n/a | n/a | 0.8069 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0130 | 0.0142 | 0.0119 | 0.0194 | 0.0027 | 0.0629 | n/a | n/a | 0.8069 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0163 | 0.0163 | 0.0160 | 0.0165 | 0.0002 | 0.0595 | n/a | n/a | 0.7785 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0015 | 0.0002 | 0.0595 | n/a | n/a | 0.7785 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0127 | 0.0132 | 0.0124 | 0.0149 | 0.0010 | 0.0595 | n/a | n/a | 0.7785 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0162 | 0.0163 | 0.0160 | 0.0168 | 0.0003 | 0.0623 | n/a | n/a | 0.8466 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.0623 | n/a | n/a | 0.8466 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0137 | 0.0139 | 0.0122 | 0.0165 | 0.0015 | 0.0623 | n/a | n/a | 0.8466 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0167 | 0.0166 | 0.0160 | 0.0172 | 0.0004 | 0.0676 | n/a | n/a | 0.6226 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0013 | 0.0011 | 0.0016 | 0.0002 | 0.0676 | n/a | n/a | 0.6226 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0104 | 0.0110 | 0.0098 | 0.0128 | 0.0012 | 0.0676 | n/a | n/a | 0.6226 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0856 | 0.0857 | 0.0854 | 0.0861 | 0.0003 | 0.0840 | n/a | n/a | 1.3601 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0072 | 0.0080 | 0.0071 | 0.0105 | 0.0013 | 0.0840 | n/a | n/a | 1.3601 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1164 | 0.1204 | 0.1131 | 0.1386 | 0.0093 | 0.0840 | n/a | n/a | 1.3601 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0256 | 0.0257 | 0.0186 | 0.0319 | 0.0053 | 0.1267 | n/a | n/a | 0.6408 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0032 | 0.0030 | 0.0018 | 0.0037 | 0.0007 | 0.1267 | n/a | n/a | 0.6408 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0164 | 0.0162 | 0.0137 | 0.0195 | 0.0019 | 0.1267 | n/a | n/a | 0.6408 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0176 | 0.0176 | 0.0172 | 0.0180 | 0.0003 | 0.0830 | n/a | n/a | 0.7637 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.0830 | n/a | n/a | 0.7637 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0134 | 0.0138 | 0.0128 | 0.0150 | 0.0009 | 0.0830 | n/a | n/a | 0.7637 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0174 | 0.0178 | 0.0170 | 0.0195 | 0.0009 | 0.0909 | n/a | n/a | 0.7735 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.0909 | n/a | n/a | 0.7735 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0135 | 0.0137 | 0.0124 | 0.0154 | 0.0011 | 0.0909 | n/a | n/a | 0.7735 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0176 | 0.0175 | 0.0169 | 0.0180 | 0.0004 | 0.0931 | n/a | n/a | 0.6159 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.0931 | n/a | n/a | 0.6159 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0108 | 0.0107 | 0.0102 | 0.0111 | 0.0003 | 0.0931 | n/a | n/a | 0.6159 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0903 | 0.0920 | 0.0894 | 0.0989 | 0.0036 | 0.1132 | n/a | n/a | 1.4458 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0102 | 0.0102 | 0.0099 | 0.0106 | 0.0003 | 0.1132 | n/a | n/a | 1.4458 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1305 | 0.1285 | 0.1179 | 0.1364 | 0.0063 | 0.1132 | n/a | n/a | 1.4458 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0232 | 0.0232 | 0.0216 | 0.0249 | 0.0011 | 0.2427 | n/a | n/a | 0.6177 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0056 | 0.0051 | 0.0039 | 0.0062 | 0.0010 | 0.2427 | n/a | n/a | 0.6177 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0143 | 0.0141 | 0.0124 | 0.0152 | 0.0009 | 0.2427 | n/a | n/a | 0.6177 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0210 | 0.0215 | 0.0208 | 0.0233 | 0.0009 | 0.1849 | n/a | n/a | 0.6000 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0040 | 0.0000 | 0.1849 | n/a | n/a | 0.6000 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0126 | 0.0127 | 0.0125 | 0.0128 | 0.0001 | 0.1849 | n/a | n/a | 0.6000 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0207 | 0.0210 | 0.0207 | 0.0217 | 0.0004 | 0.1759 | n/a | n/a | 0.6551 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0037 | 0.0000 | 0.1759 | n/a | n/a | 0.6551 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0136 | 0.0139 | 0.0127 | 0.0154 | 0.0011 | 0.1759 | n/a | n/a | 0.6551 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0206 | 0.0207 | 0.0206 | 0.0210 | 0.0002 | 0.2303 | n/a | n/a | 0.5668 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0047 | 0.0048 | 0.0043 | 0.0059 | 0.0006 | 0.2303 | n/a | n/a | 0.5668 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0117 | 0.0123 | 0.0111 | 0.0144 | 0.0012 | 0.2303 | n/a | n/a | 0.5668 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1044 | 0.1038 | 0.1022 | 0.1047 | 0.0011 | 0.2096 | n/a | n/a | 1.3669 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0219 | 0.0222 | 0.0219 | 0.0229 | 0.0004 | 0.2096 | n/a | n/a | 1.3669 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1428 | 0.1422 | 0.1332 | 0.1487 | 0.0051 | 0.2096 | n/a | n/a | 1.3669 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0150 | 0.0151 | 0.0148 | 0.0153 | 0.0002 | 0.1170 | n/a | n/a | 2.9872 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0018 | 0.0018 | 0.0016 | 0.0022 | 0.0002 | 0.1170 | n/a | n/a | 2.9872 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0448 | 0.0451 | 0.0436 | 0.0473 | 0.0013 | 0.1170 | n/a | n/a | 2.9872 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0156 | 0.0160 | 0.0150 | 0.0173 | 0.0009 | 0.2879 | n/a | n/a | 4.8579 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0045 | 0.0046 | 0.0043 | 0.0051 | 0.0003 | 0.2879 | n/a | n/a | 4.8579 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0758 | 0.0791 | 0.0709 | 0.0978 | 0.0097 | 0.2879 | n/a | n/a | 4.8579 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0158 | 0.0182 | 0.0152 | 0.0225 | 0.0032 | 0.0812 | n/a | n/a | 1.8544 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0812 | n/a | n/a | 1.8544 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0294 | 0.0299 | 0.0291 | 0.0316 | 0.0010 | 0.0812 | n/a | n/a | 1.8544 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0152 | 0.0152 | 0.0151 | 0.0153 | 0.0001 | 0.0797 | n/a | n/a | 1.9678 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0797 | n/a | n/a | 1.9678 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0299 | 0.0297 | 0.0290 | 0.0303 | 0.0005 | 0.0797 | n/a | n/a | 1.9678 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0153 | 0.0155 | 0.0152 | 0.0161 | 0.0003 | 0.0877 | n/a | n/a | 1.9328 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.0877 | n/a | n/a | 1.9328 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0296 | 0.0296 | 0.0295 | 0.0299 | 0.0002 | 0.0877 | n/a | n/a | 1.9328 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0152 | 0.0157 | 0.0150 | 0.0175 | 0.0009 | 0.1109 | n/a | n/a | 2.9784 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0019 | 0.0001 | 0.1109 | n/a | n/a | 2.9784 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0452 | 0.0451 | 0.0441 | 0.0456 | 0.0005 | 0.1109 | n/a | n/a | 2.9784 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0153 | 0.0154 | 0.0151 | 0.0158 | 0.0003 | 0.2826 | n/a | n/a | 4.7854 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0043 | 0.0044 | 0.0042 | 0.0046 | 0.0001 | 0.2826 | n/a | n/a | 4.7854 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0733 | 0.0764 | 0.0717 | 0.0912 | 0.0074 | 0.2826 | n/a | n/a | 4.7854 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0149 | 0.0150 | 0.0149 | 0.0152 | 0.0001 | 0.0814 | n/a | n/a | 1.9459 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0814 | n/a | n/a | 1.9459 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0290 | 0.0291 | 0.0287 | 0.0299 | 0.0004 | 0.0814 | n/a | n/a | 1.9459 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0150 | 0.0151 | 0.0150 | 0.0153 | 0.0001 | 0.0808 | n/a | n/a | 1.9498 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0001 | 0.0808 | n/a | n/a | 1.9498 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0292 | 0.0297 | 0.0287 | 0.0314 | 0.0010 | 0.0808 | n/a | n/a | 1.9498 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0152 | 0.0001 | 0.0862 | n/a | n/a | 1.9672 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 0.0862 | n/a | n/a | 1.9672 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0296 | 0.0297 | 0.0292 | 0.0301 | 0.0003 | 0.0862 | n/a | n/a | 1.9672 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0150 | 0.0001 | 0.1132 | n/a | n/a | 2.9425 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.1132 | n/a | n/a | 2.9425 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0437 | 0.0439 | 0.0433 | 0.0448 | 0.0005 | 0.1132 | n/a | n/a | 2.9425 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0153 | 0.0157 | 0.0149 | 0.0178 | 0.0010 | 0.4756 | n/a | n/a | 4.8586 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0073 | 0.0075 | 0.0051 | 0.0108 | 0.0020 | 0.4756 | n/a | n/a | 4.8586 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0745 | 0.0760 | 0.0712 | 0.0830 | 0.0044 | 0.4756 | n/a | n/a | 4.8586 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0156 | 0.0168 | 0.0154 | 0.0207 | 0.0020 | 0.0927 | n/a | n/a | 1.9204 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.0927 | n/a | n/a | 1.9204 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0300 | 0.0305 | 0.0294 | 0.0331 | 0.0013 | 0.0927 | n/a | n/a | 1.9204 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0151 | 0.0152 | 0.0151 | 0.0155 | 0.0002 | 0.0846 | n/a | n/a | 1.9719 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0846 | n/a | n/a | 1.9719 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0299 | 0.0304 | 0.0293 | 0.0331 | 0.0014 | 0.0846 | n/a | n/a | 1.9719 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0150 | 0.0151 | 0.0150 | 0.0152 | 0.0001 | 0.0911 | n/a | n/a | 1.9372 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 0.0911 | n/a | n/a | 1.9372 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0291 | 0.0293 | 0.0289 | 0.0300 | 0.0004 | 0.0911 | n/a | n/a | 1.9372 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0148 | 0.0149 | 0.0148 | 0.0151 | 0.0001 | 0.1191 | n/a | n/a | 3.0172 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0018 | 0.0019 | 0.0017 | 0.0023 | 0.0002 | 0.1191 | n/a | n/a | 3.0172 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0448 | 0.0502 | 0.0441 | 0.0660 | 0.0084 | 0.1191 | n/a | n/a | 3.0172 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0153 | 0.0154 | 0.0153 | 0.0156 | 0.0001 | 0.2894 | n/a | n/a | 4.6654 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0044 | 0.0045 | 0.0043 | 0.0047 | 0.0002 | 0.2894 | n/a | n/a | 4.6654 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0716 | 0.0719 | 0.0704 | 0.0737 | 0.0011 | 0.2894 | n/a | n/a | 4.6654 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0163 | 0.0164 | 0.0160 | 0.0174 | 0.0005 | 0.0921 | n/a | n/a | 1.7923 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.0921 | n/a | n/a | 1.7923 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0292 | 0.0292 | 0.0290 | 0.0294 | 0.0002 | 0.0921 | n/a | n/a | 1.7923 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0156 | 0.0156 | 0.0155 | 0.0158 | 0.0001 | 0.0882 | n/a | n/a | 1.8789 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.0882 | n/a | n/a | 1.8789 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0293 | 0.0296 | 0.0288 | 0.0314 | 0.0009 | 0.0882 | n/a | n/a | 1.8789 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0194 | 0.0200 | 0.0158 | 0.0239 | 0.0030 | 0.0752 | n/a | n/a | 1.5358 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.0752 | n/a | n/a | 1.5358 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0298 | 0.0301 | 0.0294 | 0.0314 | 0.0007 | 0.0752 | n/a | n/a | 1.5358 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0154 | 0.0158 | 0.0151 | 0.0173 | 0.0008 | 0.1501 | n/a | n/a | 3.9672 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.1501 | n/a | n/a | 3.9672 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0611 | 0.0607 | 0.0543 | 0.0688 | 0.0057 | 0.1501 | n/a | n/a | 3.9672 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0197 | 0.0200 | 0.0195 | 0.0214 | 0.0007 | 0.3485 | n/a | n/a | 4.6604 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0069 | 0.0068 | 0.0066 | 0.0070 | 0.0001 | 0.3485 | n/a | n/a | 4.6604 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0916 | 0.0917 | 0.0892 | 0.0939 | 0.0018 | 0.3485 | n/a | n/a | 4.6604 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0257 | 0.0256 | 0.0254 | 0.0259 | 0.0002 | 0.1440 | n/a | n/a | 1.1250 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0037 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.1440 | n/a | n/a | 1.1250 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0289 | 0.0292 | 0.0288 | 0.0302 | 0.0006 | 0.1440 | n/a | n/a | 1.1250 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0192 | 0.0193 | 0.0189 | 0.0199 | 0.0003 | 0.1366 | n/a | n/a | 1.5464 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0025 | 0.0026 | 0.0000 | 0.1366 | n/a | n/a | 1.5464 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0297 | 0.0302 | 0.0292 | 0.0327 | 0.0013 | 0.1366 | n/a | n/a | 1.5464 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0205 | 0.0213 | 0.0203 | 0.0231 | 0.0012 | 0.1413 | n/a | n/a | 1.5045 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0029 | 0.0031 | 0.0001 | 0.1413 | n/a | n/a | 1.5045 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0308 | 0.0310 | 0.0305 | 0.0316 | 0.0004 | 0.1413 | n/a | n/a | 1.5045 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0168 | 0.0168 | 0.0164 | 0.0175 | 0.0004 | 0.2264 | n/a | n/a | 3.8290 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0038 | 0.0039 | 0.0037 | 0.0041 | 0.0001 | 0.2264 | n/a | n/a | 3.8290 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0643 | 0.0644 | 0.0582 | 0.0731 | 0.0056 | 0.2264 | n/a | n/a | 3.8290 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0342 | 0.0341 | 0.0332 | 0.0347 | 0.0006 | 0.3399 | n/a | n/a | 2.6955 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0116 | 0.0122 | 0.0113 | 0.0148 | 0.0013 | 0.3399 | n/a | n/a | 2.6955 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.0922 | 0.0944 | 0.0886 | 0.1038 | 0.0053 | 0.3399 | n/a | n/a | 2.6955 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0621 | 0.0679 | 0.0618 | 0.0843 | 0.0087 | 0.1755 | n/a | n/a | 0.6011 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0109 | 0.0109 | 0.0108 | 0.0110 | 0.0001 | 0.1755 | n/a | n/a | 0.6011 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0373 | 0.0383 | 0.0368 | 0.0424 | 0.0020 | 0.1755 | n/a | n/a | 0.6011 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0386 | 0.0405 | 0.0348 | 0.0511 | 0.0058 | 0.1982 | n/a | n/a | 0.9505 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0077 | 0.0078 | 0.0068 | 0.0094 | 0.0009 | 0.1982 | n/a | n/a | 0.9505 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0367 | 0.0368 | 0.0359 | 0.0379 | 0.0008 | 0.1982 | n/a | n/a | 0.9505 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0401 | 0.0402 | 0.0373 | 0.0429 | 0.0018 | 0.1843 | n/a | n/a | 0.9214 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0074 | 0.0075 | 0.0073 | 0.0079 | 0.0002 | 0.1843 | n/a | n/a | 0.9214 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0369 | 0.0372 | 0.0342 | 0.0412 | 0.0022 | 0.1843 | n/a | n/a | 0.9214 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0239 | 0.0240 | 0.0229 | 0.0253 | 0.0008 | 0.4149 | n/a | n/a | 3.3319 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0099 | 0.0107 | 0.0095 | 0.0135 | 0.0015 | 0.4149 | n/a | n/a | 3.3319 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0798 | 0.0828 | 0.0764 | 0.0980 | 0.0078 | 0.4149 | n/a | n/a | 3.3319 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0916 | 0.0910 | 0.0871 | 0.0942 | 0.0028 | 0.3480 | n/a | n/a | 1.2415 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0319 | 0.0338 | 0.0310 | 0.0376 | 0.0029 | 0.3480 | n/a | n/a | 1.2415 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1137 | 0.1119 | 0.1041 | 0.1158 | 0.0043 | 0.3480 | n/a | n/a | 1.2415 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2478 | 0.2624 | 0.2234 | 0.3148 | 0.0323 | 0.1625 | n/a | n/a | 0.1524 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0403 | 0.0398 | 0.0368 | 0.0412 | 0.0016 | 0.1625 | n/a | n/a | 0.1524 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0378 | 0.0384 | 0.0374 | 0.0402 | 0.0011 | 0.1625 | n/a | n/a | 0.1524 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0845 | 0.0849 | 0.0822 | 0.0878 | 0.0018 | 0.2330 | n/a | n/a | 0.4870 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0197 | 0.0198 | 0.0193 | 0.0207 | 0.0005 | 0.2330 | n/a | n/a | 0.4870 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0412 | 0.0409 | 0.0395 | 0.0424 | 0.0011 | 0.2330 | n/a | n/a | 0.4870 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1009 | 0.1007 | 0.0990 | 0.1022 | 0.0013 | 0.2437 | n/a | n/a | 0.4205 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0246 | 0.0250 | 0.0246 | 0.0266 | 0.0008 | 0.2437 | n/a | n/a | 0.4205 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0424 | 0.0451 | 0.0415 | 0.0569 | 0.0060 | 0.2437 | n/a | n/a | 0.4205 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0151 | 0.0152 | 0.0150 | 0.0156 | 0.0002 | 0.1411 | n/a | n/a | 0.7730 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0021 | 0.0021 | 0.0021 | 0.0022 | 0.0000 | 0.1411 | n/a | n/a | 0.7730 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0117 | 0.0117 | 0.0117 | 0.0117 | 0.0000 | 0.1411 | n/a | n/a | 0.7730 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0153 | 0.0153 | 0.0151 | 0.0155 | 0.0001 | 0.5141 | n/a | n/a | 0.7115 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0079 | 0.0079 | 0.0077 | 0.0080 | 0.0001 | 0.5141 | n/a | n/a | 0.7115 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0109 | 0.0109 | 0.0106 | 0.0112 | 0.0002 | 0.5141 | n/a | n/a | 0.7115 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0150 | 0.0150 | 0.0150 | 0.0151 | 0.0000 | 0.1402 | n/a | n/a | 0.7922 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0021 | 0.0021 | 0.0021 | 0.0022 | 0.0000 | 0.1402 | n/a | n/a | 0.7922 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0119 | 0.0117 | 0.0120 | 0.0001 | 0.1402 | n/a | n/a | 0.7922 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0153 | 0.0153 | 0.0152 | 0.0156 | 0.0001 | 0.5314 | n/a | n/a | 0.7307 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0081 | 0.0082 | 0.0077 | 0.0092 | 0.0005 | 0.5314 | n/a | n/a | 0.7307 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0112 | 0.0114 | 0.0111 | 0.0120 | 0.0003 | 0.5314 | n/a | n/a | 0.7307 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0154 | 0.0155 | 0.0152 | 0.0158 | 0.0002 | 0.1387 | n/a | n/a | 0.7956 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0021 | 0.0021 | 0.0021 | 0.0023 | 0.0001 | 0.1387 | n/a | n/a | 0.7956 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0123 | 0.0122 | 0.0119 | 0.0124 | 0.0002 | 0.1387 | n/a | n/a | 0.7956 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0156 | 0.0168 | 0.0151 | 0.0211 | 0.0023 | 0.4989 | n/a | n/a | 0.7318 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0078 | 0.0078 | 0.0077 | 0.0080 | 0.0001 | 0.4989 | n/a | n/a | 0.7318 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0114 | 0.0114 | 0.0110 | 0.0118 | 0.0003 | 0.4989 | n/a | n/a | 0.7318 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0152 | 0.0152 | 0.0151 | 0.0155 | 0.0002 | 0.1445 | n/a | n/a | 0.8540 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0023 | 0.0000 | 0.1445 | n/a | n/a | 0.8540 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0129 | 0.0130 | 0.0127 | 0.0137 | 0.0004 | 0.1445 | n/a | n/a | 0.8540 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0153 | 0.0156 | 0.0152 | 0.0162 | 0.0004 | 0.5176 | n/a | n/a | 0.8124 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0079 | 0.0079 | 0.0077 | 0.0082 | 0.0002 | 0.5176 | n/a | n/a | 0.8124 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0125 | 0.0124 | 0.0122 | 0.0127 | 0.0002 | 0.5176 | n/a | n/a | 0.8124 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0167 | 0.0178 | 0.0161 | 0.0224 | 0.0023 | 0.1534 | n/a | n/a | 0.9424 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0025 | 0.0027 | 0.0001 | 0.1534 | n/a | n/a | 0.9424 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0158 | 0.0155 | 0.0136 | 0.0164 | 0.0010 | 0.1534 | n/a | n/a | 0.9424 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0161 | 0.0162 | 0.0161 | 0.0164 | 0.0001 | 0.5159 | n/a | n/a | 0.8170 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0083 | 0.0086 | 0.0082 | 0.0094 | 0.0004 | 0.5159 | n/a | n/a | 0.8170 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0132 | 0.0139 | 0.0126 | 0.0162 | 0.0013 | 0.5159 | n/a | n/a | 0.8170 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0194 | 0.0195 | 0.0192 | 0.0198 | 0.0002 | 0.1921 | n/a | n/a | 0.8177 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0037 | 0.0042 | 0.0034 | 0.0063 | 0.0011 | 0.1921 | n/a | n/a | 0.8177 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0159 | 0.0159 | 0.0137 | 0.0181 | 0.0015 | 0.1921 | n/a | n/a | 0.8177 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0199 | 0.0200 | 0.0194 | 0.0205 | 0.0003 | 0.4721 | n/a | n/a | 0.7208 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0094 | 0.0095 | 0.0092 | 0.0102 | 0.0004 | 0.4721 | n/a | n/a | 0.7208 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0144 | 0.0147 | 0.0135 | 0.0167 | 0.0012 | 0.4721 | n/a | n/a | 0.7208 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0325 | 0.0325 | 0.0321 | 0.0329 | 0.0003 | 0.2019 | n/a | n/a | 0.9243 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0000 | 0.2019 | n/a | n/a | 0.9243 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0301 | 0.0297 | 0.0199 | 0.0379 | 0.0065 | 0.2019 | n/a | n/a | 0.9243 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0344 | 0.0339 | 0.0323 | 0.0351 | 0.0010 | 0.3805 | n/a | n/a | 0.6559 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0131 | 0.0135 | 0.0130 | 0.0142 | 0.0005 | 0.3805 | n/a | n/a | 0.6559 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0226 | 0.0246 | 0.0194 | 0.0335 | 0.0053 | 0.3805 | n/a | n/a | 0.6559 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0158 | 0.0158 | 0.0156 | 0.0160 | 0.0002 | 0.1664 | n/a | n/a | 0.7628 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0026 | 0.0027 | 0.0026 | 0.0027 | 0.0000 | 0.1664 | n/a | n/a | 0.7628 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0121 | 0.0122 | 0.0117 | 0.0131 | 0.0005 | 0.1664 | n/a | n/a | 0.7628 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0103 | 0.0102 | 0.0100 | 0.0104 | 0.0001 | 0.9694 | n/a | n/a | 1.1106 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0099 | 0.0100 | 0.0096 | 0.0110 | 0.0005 | 0.9694 | n/a | n/a | 1.1106 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0114 | 0.0116 | 0.0112 | 0.0122 | 0.0004 | 0.9694 | n/a | n/a | 1.1106 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0189 | 0.0189 | 0.0186 | 0.0193 | 0.0003 | 0.2196 | n/a | n/a | 0.8181 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0041 | 0.0042 | 0.0040 | 0.0044 | 0.0002 | 0.2196 | n/a | n/a | 0.8181 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0154 | 0.0155 | 0.0149 | 0.0161 | 0.0005 | 0.2196 | n/a | n/a | 0.8181 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0129 | 0.0131 | 0.0128 | 0.0133 | 0.0002 | 0.9255 | n/a | n/a | 1.0870 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0120 | 0.0121 | 0.0117 | 0.0125 | 0.0004 | 0.9255 | n/a | n/a | 1.0870 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0141 | 0.0145 | 0.0128 | 0.0165 | 0.0013 | 0.9255 | n/a | n/a | 1.0870 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0294 | 0.0297 | 0.0293 | 0.0305 | 0.0005 | 0.2496 | n/a | n/a | 0.9350 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0073 | 0.0073 | 0.0073 | 0.0074 | 0.0000 | 0.2496 | n/a | n/a | 0.9350 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0275 | 0.0269 | 0.0254 | 0.0286 | 0.0012 | 0.2496 | n/a | n/a | 0.9350 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0227 | 0.0229 | 0.0226 | 0.0235 | 0.0003 | 0.6968 | n/a | n/a | 0.6029 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0158 | 0.0159 | 0.0150 | 0.0175 | 0.0009 | 0.6968 | n/a | n/a | 0.6029 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0137 | 0.0138 | 0.0136 | 0.0140 | 0.0002 | 0.6968 | n/a | n/a | 0.6029 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0793 | 0.0797 | 0.0789 | 0.0811 | 0.0008 | 0.2722 | n/a | n/a | 0.8325 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0216 | 0.0216 | 0.0213 | 0.0221 | 0.0003 | 0.2722 | n/a | n/a | 0.8325 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0660 | 0.0663 | 0.0638 | 0.0687 | 0.0017 | 0.2722 | n/a | n/a | 0.8325 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0574 | 0.0581 | 0.0567 | 0.0617 | 0.0018 | 0.4181 | n/a | n/a | 0.2929 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0240 | 0.0240 | 0.0235 | 0.0249 | 0.0005 | 0.4181 | n/a | n/a | 0.2929 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0168 | 0.0201 | 0.0164 | 0.0257 | 0.0044 | 0.4181 | n/a | n/a | 0.2929 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0180 | 0.0181 | 0.0179 | 0.0184 | 0.0002 | 0.1107 | n/a | n/a | 0.5380 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0020 | 0.0000 | 0.1107 | n/a | n/a | 0.5380 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0097 | 0.0105 | 0.0095 | 0.0140 | 0.0017 | 0.1107 | n/a | n/a | 0.5380 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0957 | 0.1047 | 0.0936 | 0.1315 | 0.0143 | 0.1096 | n/a | n/a | 0.4081 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0105 | 0.0105 | 0.0105 | 0.0106 | 0.0000 | 0.1096 | n/a | n/a | 0.4081 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0391 | 0.0340 | 0.0165 | 0.0529 | 0.0146 | 0.1096 | n/a | n/a | 0.4081 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3866 | 0.3864 | 0.3760 | 0.3949 | 0.0074 | 0.9340 | n/a | n/a | 0.2952 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3611 | 0.3554 | 0.3169 | 0.3806 | 0.0222 | 0.9340 | n/a | n/a | 0.2952 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1141 | 0.1137 | 0.0950 | 0.1306 | 0.0124 | 0.9340 | n/a | n/a | 0.2952 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.0576 | 2.1371 | 1.9173 | 2.5486 | 0.2158 | 0.2247 | n/a | n/a | 0.1365 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4624 | 0.4665 | 0.4056 | 0.5353 | 0.0553 | 0.2247 | n/a | n/a | 0.1365 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2808 | 0.3076 | 0.2667 | 0.3627 | 0.0424 | 0.2247 | n/a | n/a | 0.1365 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1832 | 0.1844 | 0.1803 | 0.1918 | 0.0042 | 7.0413 | n/a | n/a | 0.5449 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2897 | 1.3941 | 1.2526 | 1.7658 | 0.1905 | 7.0413 | n/a | n/a | 0.5449 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.0998 | 0.1066 | 0.0963 | 0.1374 | 0.0155 | 7.0413 | n/a | n/a | 0.5449 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 5.8037 | 5.8352 | 5.6717 | 5.9826 | 0.1191 | 10.5508 | n/a | n/a | 0.0550 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 61.2331 | 61.1396 | 60.1840 | 61.6365 | 0.5080 | 10.5508 | n/a | n/a | 0.0550 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3194 | 0.3293 | 0.3155 | 0.3682 | 0.0197 | 10.5508 | n/a | n/a | 0.0550 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0276 | 0.0277 | 0.0273 | 0.0285 | 0.0004 | 5.6964 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1571 | 0.1573 | 0.1546 | 0.1617 | 0.0024 | 5.6964 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0273 | 0.0289 | 0.0269 | 0.0360 | 0.0035 | 19.9107 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5435 | 0.5446 | 0.5410 | 0.5507 | 0.0034 | 19.9107 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7020 | 0.7018 | 0.6924 | 0.7108 | 0.0069 | 10.7494 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.5462 | 7.5903 | 7.3992 | 7.8303 | 0.1456 | 10.7494 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6872 | 0.6893 | 0.6778 | 0.7059 | 0.0092 | 37.8776 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 26.0300 | 26.1376 | 25.8286 | 26.5418 | 0.2439 | 37.8776 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1170 | 0.1171 | 0.1167 | 0.1177 | 0.0004 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2935 | 0.2934 | 0.2925 | 0.2945 | 0.0007 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2467 | 0.2503 | 0.2456 | 0.2658 | 0.0078 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.5994 | 0.5995 | 0.5933 | 0.6080 | 0.0052 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1454 | 1.1439 | 1.1401 | 1.1468 | 0.0028 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7089 | 2.7054 | 2.6910 | 2.7179 | 0.0112 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 4.7408 | 4.7423 | 4.6576 | 4.7954 | 0.0477 | 0.0328 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1556 | 0.1591 | 0.1535 | 0.1715 | 0.0067 | 0.0328 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |
| backends | backend_info | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | available_devices | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cpu_transfer | `(1,)` | TensorStudio | 0.0014 | 0.0015 | 0.0014 | 0.0017 | 0.0001 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cuda_availability_check | `(1,)` | TensorStudio | 0.0021 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
