# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.10.0`
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
- TensorStudio wins versus JAX CPU dispatch: `38`
- TensorStudio losses versus JAX CPU dispatch: `60`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0163 | 0.0168 | 0.0159 | 0.0195 | 0.0013 | 0.0387 | n/a | n/a | 0.6935 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0006 | 0.0000 | 0.0387 | n/a | n/a | 0.6935 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0113 | 0.0114 | 0.0110 | 0.0117 | 0.0003 | 0.0387 | n/a | n/a | 0.6935 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0164 | 0.0189 | 0.0159 | 0.0251 | 0.0036 | 0.0413 | n/a | n/a | 0.7256 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0413 | n/a | n/a | 0.7256 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0119 | 0.0121 | 0.0113 | 0.0130 | 0.0006 | 0.0413 | n/a | n/a | 0.7256 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0159 | 0.0159 | 0.0157 | 0.0159 | 0.0001 | 0.0424 | n/a | n/a | 0.7133 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0424 | n/a | n/a | 0.7133 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0113 | 0.0119 | 0.0111 | 0.0145 | 0.0013 | 0.0424 | n/a | n/a | 0.7133 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0159 | 0.0163 | 0.0158 | 0.0173 | 0.0006 | 0.0443 | n/a | n/a | 0.5518 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0443 | n/a | n/a | 0.5518 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0088 | 0.0088 | 0.0086 | 0.0089 | 0.0001 | 0.0443 | n/a | n/a | 0.5518 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0857 | 0.0856 | 0.0843 | 0.0867 | 0.0008 | 0.0620 | n/a | n/a | 1.0549 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0053 | 0.0054 | 0.0052 | 0.0057 | 0.0002 | 0.0620 | n/a | n/a | 1.0549 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0904 | 0.0899 | 0.0872 | 0.0920 | 0.0016 | 0.0620 | n/a | n/a | 1.0549 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0161 | 0.0168 | 0.0158 | 0.0191 | 0.0012 | 0.0399 | n/a | n/a | 0.7203 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0399 | n/a | n/a | 0.7203 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0116 | 0.0118 | 0.0111 | 0.0131 | 0.0007 | 0.0399 | n/a | n/a | 0.7203 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0161 | 0.0161 | 0.0160 | 0.0162 | 0.0001 | 0.0434 | n/a | n/a | 0.7501 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0434 | n/a | n/a | 0.7501 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0121 | 0.0120 | 0.0114 | 0.0128 | 0.0005 | 0.0434 | n/a | n/a | 0.7501 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0159 | 0.0159 | 0.0158 | 0.0160 | 0.0001 | 0.0426 | n/a | n/a | 0.7331 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0426 | n/a | n/a | 0.7331 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0117 | 0.0116 | 0.0115 | 0.0117 | 0.0001 | 0.0426 | n/a | n/a | 0.7331 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0159 | 0.0161 | 0.0159 | 0.0166 | 0.0003 | 0.0444 | n/a | n/a | 0.5562 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0444 | n/a | n/a | 0.5562 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0089 | 0.0089 | 0.0086 | 0.0096 | 0.0004 | 0.0444 | n/a | n/a | 0.5562 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0845 | 0.0848 | 0.0839 | 0.0865 | 0.0009 | 0.0641 | n/a | n/a | 1.0489 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0054 | 0.0055 | 0.0054 | 0.0056 | 0.0001 | 0.0641 | n/a | n/a | 1.0489 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0886 | 0.0880 | 0.0865 | 0.0893 | 0.0011 | 0.0641 | n/a | n/a | 1.0489 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0161 | 0.0163 | 0.0161 | 0.0169 | 0.0003 | 0.0405 | n/a | n/a | 0.7457 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0008 | 0.0001 | 0.0405 | n/a | n/a | 0.7457 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0120 | 0.0143 | 0.0112 | 0.0201 | 0.0036 | 0.0405 | n/a | n/a | 0.7457 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0161 | 0.0168 | 0.0160 | 0.0196 | 0.0014 | 0.0436 | n/a | n/a | 0.7013 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0436 | n/a | n/a | 0.7013 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0113 | 0.0115 | 0.0113 | 0.0118 | 0.0002 | 0.0436 | n/a | n/a | 0.7013 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0161 | 0.0161 | 0.0158 | 0.0163 | 0.0002 | 0.0455 | n/a | n/a | 0.7242 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0455 | n/a | n/a | 0.7242 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0117 | 0.0117 | 0.0113 | 0.0122 | 0.0003 | 0.0455 | n/a | n/a | 0.7242 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0161 | 0.0161 | 0.0159 | 0.0164 | 0.0002 | 0.0444 | n/a | n/a | 0.5576 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0444 | n/a | n/a | 0.5576 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0090 | 0.0091 | 0.0089 | 0.0097 | 0.0003 | 0.0444 | n/a | n/a | 0.5576 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0857 | 0.0856 | 0.0845 | 0.0867 | 0.0009 | 0.0636 | n/a | n/a | 1.0295 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0055 | 0.0055 | 0.0054 | 0.0058 | 0.0001 | 0.0636 | n/a | n/a | 1.0295 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0882 | 0.0884 | 0.0877 | 0.0892 | 0.0006 | 0.0636 | n/a | n/a | 1.0295 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0161 | 0.0164 | 0.0161 | 0.0178 | 0.0007 | 0.0408 | n/a | n/a | 0.7150 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0010 | 0.0001 | 0.0408 | n/a | n/a | 0.7150 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0115 | 0.0116 | 0.0114 | 0.0117 | 0.0001 | 0.0408 | n/a | n/a | 0.7150 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0162 | 0.0165 | 0.0159 | 0.0181 | 0.0008 | 0.0436 | n/a | n/a | 0.7465 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.0436 | n/a | n/a | 0.7465 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0121 | 0.0121 | 0.0117 | 0.0126 | 0.0003 | 0.0436 | n/a | n/a | 0.7465 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0162 | 0.0161 | 0.0158 | 0.0163 | 0.0002 | 0.0438 | n/a | n/a | 0.7398 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0438 | n/a | n/a | 0.7398 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0120 | 0.0120 | 0.0115 | 0.0125 | 0.0004 | 0.0438 | n/a | n/a | 0.7398 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0160 | 0.0161 | 0.0159 | 0.0167 | 0.0003 | 0.0450 | n/a | n/a | 0.5528 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0450 | n/a | n/a | 0.5528 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0088 | 0.0089 | 0.0086 | 0.0094 | 0.0003 | 0.0450 | n/a | n/a | 0.5528 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0859 | 0.0860 | 0.0845 | 0.0883 | 0.0013 | 0.0647 | n/a | n/a | 1.0337 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0056 | 0.0055 | 0.0054 | 0.0057 | 0.0001 | 0.0647 | n/a | n/a | 1.0337 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0888 | 0.0884 | 0.0873 | 0.0890 | 0.0006 | 0.0647 | n/a | n/a | 1.0337 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0162 | 0.0161 | 0.0160 | 0.0162 | 0.0001 | 0.0569 | n/a | n/a | 0.8730 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0009 | 0.0009 | 0.0009 | 0.0010 | 0.0000 | 0.0569 | n/a | n/a | 0.8730 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0141 | 0.0145 | 0.0134 | 0.0160 | 0.0010 | 0.0569 | n/a | n/a | 0.8730 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0159 | 0.0163 | 0.0158 | 0.0170 | 0.0005 | 0.0767 | n/a | n/a | 0.8663 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0012 | 0.0012 | 0.0010 | 0.0014 | 0.0001 | 0.0767 | n/a | n/a | 0.8663 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0138 | 0.0141 | 0.0125 | 0.0162 | 0.0013 | 0.0767 | n/a | n/a | 0.8663 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0162 | 0.0163 | 0.0162 | 0.0167 | 0.0002 | 0.0616 | n/a | n/a | 0.7904 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0616 | n/a | n/a | 0.7904 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0128 | 0.0128 | 0.0126 | 0.0131 | 0.0002 | 0.0616 | n/a | n/a | 0.7904 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0166 | 0.0168 | 0.0161 | 0.0175 | 0.0005 | 0.0645 | n/a | n/a | 0.6713 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0012 | 0.0000 | 0.0645 | n/a | n/a | 0.6713 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0112 | 0.0112 | 0.0101 | 0.0129 | 0.0010 | 0.0645 | n/a | n/a | 0.6713 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0868 | 0.0873 | 0.0854 | 0.0906 | 0.0019 | 0.0827 | n/a | n/a | 1.3777 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0072 | 0.0073 | 0.0070 | 0.0080 | 0.0004 | 0.0827 | n/a | n/a | 1.3777 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1195 | 0.1413 | 0.1037 | 0.2533 | 0.0565 | 0.0827 | n/a | n/a | 1.3777 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0180 | 0.0181 | 0.0172 | 0.0199 | 0.0010 | 0.0906 | n/a | n/a | 0.7581 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0016 | 0.0017 | 0.0015 | 0.0022 | 0.0002 | 0.0906 | n/a | n/a | 0.7581 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0136 | 0.0135 | 0.0127 | 0.0141 | 0.0006 | 0.0906 | n/a | n/a | 0.7581 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0181 | 0.0186 | 0.0174 | 0.0198 | 0.0010 | 0.0827 | n/a | n/a | 0.7579 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0018 | 0.0015 | 0.0026 | 0.0005 | 0.0827 | n/a | n/a | 0.7579 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0138 | 0.0139 | 0.0130 | 0.0147 | 0.0006 | 0.0827 | n/a | n/a | 0.7579 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0172 | 0.0180 | 0.0169 | 0.0210 | 0.0016 | 0.1005 | n/a | n/a | 0.8134 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.1005 | n/a | n/a | 0.8134 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0140 | 0.0140 | 0.0130 | 0.0150 | 0.0007 | 0.1005 | n/a | n/a | 0.8134 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0173 | 0.0175 | 0.0172 | 0.0179 | 0.0003 | 0.0965 | n/a | n/a | 0.6591 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0000 | 0.0965 | n/a | n/a | 0.6591 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0114 | 0.0113 | 0.0103 | 0.0117 | 0.0005 | 0.0965 | n/a | n/a | 0.6591 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0951 | 0.0957 | 0.0893 | 0.1019 | 0.0051 | 0.1071 | n/a | n/a | 1.2234 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0102 | 0.0101 | 0.0098 | 0.0103 | 0.0002 | 0.1071 | n/a | n/a | 1.2234 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1163 | 0.1185 | 0.1129 | 0.1279 | 0.0058 | 0.1071 | n/a | n/a | 1.2234 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0209 | 0.0211 | 0.0207 | 0.0215 | 0.0003 | 0.1800 | n/a | n/a | 0.7098 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.1800 | n/a | n/a | 0.7098 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0148 | 0.0159 | 0.0133 | 0.0219 | 0.0030 | 0.1800 | n/a | n/a | 0.7098 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0207 | 0.0216 | 0.0207 | 0.0250 | 0.0017 | 0.2728 | n/a | n/a | 0.6048 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0057 | 0.0057 | 0.0055 | 0.0061 | 0.0002 | 0.2728 | n/a | n/a | 0.6048 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0125 | 0.0128 | 0.0123 | 0.0141 | 0.0007 | 0.2728 | n/a | n/a | 0.6048 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0221 | 0.0222 | 0.0218 | 0.0231 | 0.0005 | 0.1829 | n/a | n/a | 0.5973 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0040 | 0.0040 | 0.0040 | 0.0041 | 0.0000 | 0.1829 | n/a | n/a | 0.5973 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0132 | 0.0139 | 0.0125 | 0.0172 | 0.0017 | 0.1829 | n/a | n/a | 0.5973 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0211 | 0.0214 | 0.0210 | 0.0220 | 0.0004 | 0.3578 | n/a | n/a | 0.5890 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0076 | 0.0085 | 0.0062 | 0.0145 | 0.0030 | 0.3578 | n/a | n/a | 0.5890 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0125 | 0.0127 | 0.0118 | 0.0137 | 0.0008 | 0.3578 | n/a | n/a | 0.5890 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1031 | 0.1037 | 0.1024 | 0.1057 | 0.0012 | 0.2123 | n/a | n/a | 1.3455 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0219 | 0.0222 | 0.0212 | 0.0234 | 0.0009 | 0.2123 | n/a | n/a | 1.3455 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1387 | 0.1407 | 0.1355 | 0.1512 | 0.0056 | 0.2123 | n/a | n/a | 1.3455 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0149 | 0.0151 | 0.0148 | 0.0161 | 0.0005 | 0.1148 | n/a | n/a | 2.9145 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.1148 | n/a | n/a | 2.9145 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0434 | 0.0435 | 0.0433 | 0.0438 | 0.0002 | 0.1148 | n/a | n/a | 2.9145 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0149 | 0.0154 | 0.0149 | 0.0172 | 0.0009 | 0.2959 | n/a | n/a | 4.9272 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0046 | 0.0001 | 0.2959 | n/a | n/a | 4.9272 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0735 | 0.0799 | 0.0717 | 0.1051 | 0.0127 | 0.2959 | n/a | n/a | 4.9272 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0150 | 0.0152 | 0.0149 | 0.0162 | 0.0005 | 0.0834 | n/a | n/a | 2.0441 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0013 | 0.0015 | 0.0012 | 0.0023 | 0.0004 | 0.0834 | n/a | n/a | 2.0441 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0307 | 0.0308 | 0.0297 | 0.0330 | 0.0012 | 0.0834 | n/a | n/a | 2.0441 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0150 | 0.0153 | 0.0149 | 0.0160 | 0.0004 | 0.0817 | n/a | n/a | 1.9616 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0817 | n/a | n/a | 1.9616 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0295 | 0.0294 | 0.0292 | 0.0296 | 0.0001 | 0.0817 | n/a | n/a | 1.9616 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0151 | 0.0150 | 0.0148 | 0.0152 | 0.0001 | 0.0867 | n/a | n/a | 1.9534 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0001 | 0.0867 | n/a | n/a | 1.9534 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0295 | 0.0295 | 0.0291 | 0.0299 | 0.0003 | 0.0867 | n/a | n/a | 1.9534 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0149 | 0.0151 | 0.0148 | 0.0160 | 0.0005 | 0.1134 | n/a | n/a | 2.9879 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.1134 | n/a | n/a | 2.9879 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0445 | 0.0444 | 0.0440 | 0.0448 | 0.0003 | 0.1134 | n/a | n/a | 2.9879 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0150 | 0.0154 | 0.0149 | 0.0170 | 0.0008 | 0.2889 | n/a | n/a | 5.1010 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0043 | 0.0044 | 0.0043 | 0.0047 | 0.0002 | 0.2889 | n/a | n/a | 5.1010 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0763 | 0.0756 | 0.0736 | 0.0772 | 0.0015 | 0.2889 | n/a | n/a | 5.1010 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0151 | 0.0151 | 0.0148 | 0.0158 | 0.0003 | 0.0824 | n/a | n/a | 1.9683 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0824 | n/a | n/a | 1.9683 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0297 | 0.0298 | 0.0293 | 0.0303 | 0.0004 | 0.0824 | n/a | n/a | 1.9683 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0148 | 0.0151 | 0.0148 | 0.0163 | 0.0006 | 0.1146 | n/a | n/a | 2.0579 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0017 | 0.0018 | 0.0012 | 0.0027 | 0.0006 | 0.1146 | n/a | n/a | 2.0579 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0305 | 0.0315 | 0.0302 | 0.0340 | 0.0015 | 0.1146 | n/a | n/a | 2.0579 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0151 | 0.0151 | 0.0149 | 0.0153 | 0.0001 | 0.0844 | n/a | n/a | 1.9884 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0844 | n/a | n/a | 1.9884 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0299 | 0.0307 | 0.0292 | 0.0329 | 0.0014 | 0.0844 | n/a | n/a | 1.9884 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0150 | 0.0001 | 0.1132 | n/a | n/a | 2.9827 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.1132 | n/a | n/a | 2.9827 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0443 | 0.0444 | 0.0436 | 0.0458 | 0.0007 | 0.1132 | n/a | n/a | 2.9827 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0150 | 0.0151 | 0.0149 | 0.0154 | 0.0002 | 0.3056 | n/a | n/a | 4.8092 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0046 | 0.0045 | 0.0043 | 0.0048 | 0.0002 | 0.3056 | n/a | n/a | 4.8092 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0723 | 0.0793 | 0.0717 | 0.1058 | 0.0133 | 0.3056 | n/a | n/a | 4.8092 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0152 | 0.0152 | 0.0151 | 0.0153 | 0.0001 | 0.0854 | n/a | n/a | 1.9787 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0854 | n/a | n/a | 1.9787 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0300 | 0.0301 | 0.0292 | 0.0309 | 0.0007 | 0.0854 | n/a | n/a | 1.9787 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0149 | 0.0150 | 0.0148 | 0.0154 | 0.0002 | 0.0850 | n/a | n/a | 2.0360 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0850 | n/a | n/a | 2.0360 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0304 | 0.0305 | 0.0301 | 0.0311 | 0.0004 | 0.0850 | n/a | n/a | 2.0360 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0152 | 0.0153 | 0.0149 | 0.0158 | 0.0004 | 0.0868 | n/a | n/a | 1.9807 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.0868 | n/a | n/a | 1.9807 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0301 | 0.0305 | 0.0296 | 0.0320 | 0.0009 | 0.0868 | n/a | n/a | 1.9807 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0149 | 0.0150 | 0.0147 | 0.0156 | 0.0003 | 0.1145 | n/a | n/a | 2.9777 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0019 | 0.0001 | 0.1145 | n/a | n/a | 2.9777 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0445 | 0.0446 | 0.0441 | 0.0458 | 0.0006 | 0.1145 | n/a | n/a | 2.9777 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0154 | 0.0156 | 0.0153 | 0.0163 | 0.0004 | 0.2919 | n/a | n/a | 4.7047 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0045 | 0.0045 | 0.0045 | 0.0046 | 0.0001 | 0.2919 | n/a | n/a | 4.7047 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0725 | 0.0736 | 0.0717 | 0.0771 | 0.0021 | 0.2919 | n/a | n/a | 4.7047 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0163 | 0.0172 | 0.0162 | 0.0208 | 0.0018 | 0.0938 | n/a | n/a | 1.8451 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.0938 | n/a | n/a | 1.8451 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0301 | 0.0307 | 0.0294 | 0.0332 | 0.0014 | 0.0938 | n/a | n/a | 1.8451 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0155 | 0.0155 | 0.0153 | 0.0159 | 0.0002 | 0.0908 | n/a | n/a | 1.9539 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0016 | 0.0013 | 0.0022 | 0.0004 | 0.0908 | n/a | n/a | 1.9539 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0303 | 0.0304 | 0.0300 | 0.0308 | 0.0002 | 0.0908 | n/a | n/a | 1.9539 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0158 | 0.0158 | 0.0155 | 0.0162 | 0.0003 | 0.0900 | n/a | n/a | 1.9270 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0018 | 0.0001 | 0.0900 | n/a | n/a | 1.9270 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0304 | 0.0303 | 0.0293 | 0.0312 | 0.0008 | 0.0900 | n/a | n/a | 1.9270 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0153 | 0.0154 | 0.0152 | 0.0157 | 0.0002 | 0.1509 | n/a | n/a | 3.8470 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0025 | 0.0001 | 0.1509 | n/a | n/a | 3.8470 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0588 | 0.0615 | 0.0561 | 0.0698 | 0.0051 | 0.1509 | n/a | n/a | 3.8470 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0196 | 0.0206 | 0.0193 | 0.0245 | 0.0020 | 0.3541 | n/a | n/a | 4.6444 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0070 | 0.0069 | 0.0066 | 0.0074 | 0.0003 | 0.3541 | n/a | n/a | 4.6444 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0912 | 0.0906 | 0.0875 | 0.0940 | 0.0025 | 0.3541 | n/a | n/a | 4.6444 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0255 | 0.0255 | 0.0254 | 0.0258 | 0.0001 | 0.1406 | n/a | n/a | 1.1787 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0038 | 0.0001 | 0.1406 | n/a | n/a | 1.1787 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0300 | 0.0316 | 0.0299 | 0.0368 | 0.0026 | 0.1406 | n/a | n/a | 1.1787 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0189 | 0.0197 | 0.0184 | 0.0235 | 0.0019 | 0.1436 | n/a | n/a | 1.5702 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0030 | 0.0001 | 0.1436 | n/a | n/a | 1.5702 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0297 | 0.0298 | 0.0296 | 0.0300 | 0.0002 | 0.1436 | n/a | n/a | 1.5702 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0202 | 0.0202 | 0.0200 | 0.0205 | 0.0002 | 0.1520 | n/a | n/a | 1.5645 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0031 | 0.0031 | 0.0029 | 0.0036 | 0.0003 | 0.1520 | n/a | n/a | 1.5645 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0316 | 0.0318 | 0.0301 | 0.0338 | 0.0014 | 0.1520 | n/a | n/a | 1.5645 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0173 | 0.0174 | 0.0165 | 0.0192 | 0.0009 | 0.2105 | n/a | n/a | 3.3996 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0037 | 0.0000 | 0.2105 | n/a | n/a | 3.3996 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0587 | 0.0617 | 0.0563 | 0.0696 | 0.0052 | 0.2105 | n/a | n/a | 3.3996 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0332 | 0.0333 | 0.0330 | 0.0340 | 0.0004 | 0.3674 | n/a | n/a | 2.9581 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0122 | 0.0123 | 0.0120 | 0.0126 | 0.0002 | 0.3674 | n/a | n/a | 2.9581 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.0982 | 0.0977 | 0.0935 | 0.1012 | 0.0031 | 0.3674 | n/a | n/a | 2.9581 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0621 | 0.0629 | 0.0620 | 0.0660 | 0.0016 | 0.1776 | n/a | n/a | 0.5608 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0110 | 0.0110 | 0.0108 | 0.0112 | 0.0001 | 0.1776 | n/a | n/a | 0.5608 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0348 | 0.0367 | 0.0331 | 0.0419 | 0.0033 | 0.1776 | n/a | n/a | 0.5608 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0305 | 0.0307 | 0.0303 | 0.0313 | 0.0003 | 0.1919 | n/a | n/a | 1.1070 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0059 | 0.0059 | 0.0057 | 0.0062 | 0.0002 | 0.1919 | n/a | n/a | 1.1070 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0338 | 0.0337 | 0.0330 | 0.0340 | 0.0003 | 0.1919 | n/a | n/a | 1.1070 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0402 | 0.0426 | 0.0370 | 0.0555 | 0.0068 | 0.1920 | n/a | n/a | 0.8788 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0077 | 0.0076 | 0.0073 | 0.0078 | 0.0002 | 0.1920 | n/a | n/a | 0.8788 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0353 | 0.0357 | 0.0337 | 0.0392 | 0.0019 | 0.1920 | n/a | n/a | 0.8788 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0218 | 0.0231 | 0.0212 | 0.0288 | 0.0029 | 0.3903 | n/a | n/a | 3.6631 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0085 | 0.0087 | 0.0085 | 0.0094 | 0.0003 | 0.3903 | n/a | n/a | 3.6631 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0798 | 0.0871 | 0.0724 | 0.1116 | 0.0148 | 0.3903 | n/a | n/a | 3.6631 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0875 | 0.0873 | 0.0857 | 0.0886 | 0.0009 | 0.3867 | n/a | n/a | 1.2360 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0338 | 0.0354 | 0.0334 | 0.0422 | 0.0034 | 0.3867 | n/a | n/a | 1.2360 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1081 | 0.1064 | 0.0982 | 0.1119 | 0.0048 | 0.3867 | n/a | n/a | 1.2360 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2117 | 0.2113 | 0.2100 | 0.2119 | 0.0007 | 0.1911 | n/a | n/a | 0.1949 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0405 | 0.0404 | 0.0397 | 0.0413 | 0.0005 | 0.1911 | n/a | n/a | 0.1949 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0413 | 0.0414 | 0.0381 | 0.0447 | 0.0022 | 0.1911 | n/a | n/a | 0.1949 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0788 | 0.0791 | 0.0776 | 0.0815 | 0.0013 | 0.2432 | n/a | n/a | 0.5373 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0192 | 0.0190 | 0.0186 | 0.0196 | 0.0004 | 0.2432 | n/a | n/a | 0.5373 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0424 | 0.0429 | 0.0400 | 0.0482 | 0.0029 | 0.2432 | n/a | n/a | 0.5373 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1013 | 0.1012 | 0.1003 | 0.1018 | 0.0005 | 0.2558 | n/a | n/a | 0.4415 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0259 | 0.0321 | 0.0257 | 0.0416 | 0.0076 | 0.2558 | n/a | n/a | 0.4415 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0447 | 0.0483 | 0.0418 | 0.0643 | 0.0081 | 0.2558 | n/a | n/a | 0.4415 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0207 | 0.0204 | 0.0152 | 0.0294 | 0.0052 | 0.1053 | n/a | n/a | 0.6043 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0025 | 0.0001 | 0.1053 | n/a | n/a | 0.6043 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0125 | 0.0123 | 0.0113 | 0.0131 | 0.0007 | 0.1053 | n/a | n/a | 0.6043 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0153 | 0.0155 | 0.0152 | 0.0162 | 0.0004 | 0.5208 | n/a | n/a | 0.6998 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0080 | 0.0086 | 0.0079 | 0.0111 | 0.0013 | 0.5208 | n/a | n/a | 0.6998 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0107 | 0.0110 | 0.0106 | 0.0119 | 0.0005 | 0.5208 | n/a | n/a | 0.6998 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0153 | 0.0154 | 0.0153 | 0.0156 | 0.0001 | 0.1436 | n/a | n/a | 0.7997 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0022 | 0.0023 | 0.0021 | 0.0026 | 0.0002 | 0.1436 | n/a | n/a | 0.7997 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0118 | 0.0125 | 0.0003 | 0.1436 | n/a | n/a | 0.7997 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0153 | 0.0153 | 0.0153 | 0.0155 | 0.0001 | 0.5206 | n/a | n/a | 0.7160 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0080 | 0.0079 | 0.0078 | 0.0080 | 0.0001 | 0.5206 | n/a | n/a | 0.7160 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0110 | 0.0111 | 0.0109 | 0.0115 | 0.0002 | 0.5206 | n/a | n/a | 0.7160 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0153 | 0.0154 | 0.0151 | 0.0160 | 0.0003 | 0.1474 | n/a | n/a | 0.7750 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.1474 | n/a | n/a | 0.7750 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0119 | 0.0118 | 0.0115 | 0.0121 | 0.0002 | 0.1474 | n/a | n/a | 0.7750 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0154 | 0.0154 | 0.0152 | 0.0155 | 0.0001 | 0.5097 | n/a | n/a | 0.7198 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0078 | 0.0079 | 0.0078 | 0.0080 | 0.0001 | 0.5097 | n/a | n/a | 0.7198 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0111 | 0.0115 | 0.0108 | 0.0136 | 0.0010 | 0.5097 | n/a | n/a | 0.7198 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0154 | 0.0156 | 0.0153 | 0.0161 | 0.0003 | 0.1541 | n/a | n/a | 0.8437 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0024 | 0.0024 | 0.0021 | 0.0026 | 0.0002 | 0.1541 | n/a | n/a | 0.8437 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0130 | 0.0134 | 0.0129 | 0.0150 | 0.0008 | 0.1541 | n/a | n/a | 0.8437 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0157 | 0.0160 | 0.0154 | 0.0169 | 0.0006 | 0.5183 | n/a | n/a | 0.8220 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0081 | 0.0084 | 0.0078 | 0.0097 | 0.0007 | 0.5183 | n/a | n/a | 0.8220 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0129 | 0.0131 | 0.0121 | 0.0144 | 0.0007 | 0.5183 | n/a | n/a | 0.8220 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0165 | 0.0165 | 0.0163 | 0.0169 | 0.0002 | 0.1572 | n/a | n/a | 0.9535 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0025 | 0.0028 | 0.0001 | 0.1572 | n/a | n/a | 0.9535 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0157 | 0.0157 | 0.0139 | 0.0171 | 0.0012 | 0.1572 | n/a | n/a | 0.9535 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0180 | 0.0187 | 0.0165 | 0.0221 | 0.0022 | 0.4791 | n/a | n/a | 0.7505 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0086 | 0.0088 | 0.0082 | 0.0095 | 0.0004 | 0.4791 | n/a | n/a | 0.7505 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0135 | 0.0140 | 0.0126 | 0.0161 | 0.0013 | 0.4791 | n/a | n/a | 0.7505 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0206 | 0.0209 | 0.0194 | 0.0224 | 0.0011 | 0.1695 | n/a | n/a | 0.7735 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0035 | 0.0035 | 0.0035 | 0.0038 | 0.0001 | 0.1695 | n/a | n/a | 0.7735 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0160 | 0.0177 | 0.0144 | 0.0225 | 0.0031 | 0.1695 | n/a | n/a | 0.7735 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0205 | 0.0204 | 0.0198 | 0.0212 | 0.0005 | 0.5022 | n/a | n/a | 0.8445 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0103 | 0.0108 | 0.0094 | 0.0124 | 0.0011 | 0.5022 | n/a | n/a | 0.8445 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0173 | 0.0184 | 0.0164 | 0.0220 | 0.0020 | 0.5022 | n/a | n/a | 0.8445 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0435 | 0.0432 | 0.0372 | 0.0482 | 0.0040 | 0.1750 | n/a | n/a | 0.5090 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0076 | 0.0078 | 0.0074 | 0.0085 | 0.0004 | 0.1750 | n/a | n/a | 0.5090 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0222 | 0.0239 | 0.0204 | 0.0298 | 0.0035 | 0.1750 | n/a | n/a | 0.5090 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0348 | 0.0344 | 0.0322 | 0.0377 | 0.0020 | 0.4652 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0162 | 0.0165 | 0.0153 | 0.0193 | 0.0014 | 0.4652 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0308 | 0.0293 | 0.0189 | 0.0341 | 0.0055 | 0.4652 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0159 | 0.0159 | 0.0157 | 0.0161 | 0.0002 | 0.1904 | n/a | n/a | 0.8133 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0030 | 0.0031 | 0.0029 | 0.0033 | 0.0001 | 0.1904 | n/a | n/a | 0.8133 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0129 | 0.0130 | 0.0128 | 0.0134 | 0.0002 | 0.1904 | n/a | n/a | 0.8133 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0102 | 0.0103 | 0.0100 | 0.0108 | 0.0003 | 1.0396 | n/a | n/a | 1.2341 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0106 | 0.0105 | 0.0098 | 0.0110 | 0.0004 | 1.0396 | n/a | n/a | 1.2341 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0126 | 0.0131 | 0.0120 | 0.0144 | 0.0011 | 1.0396 | n/a | n/a | 1.2341 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0193 | 0.0192 | 0.0184 | 0.0195 | 0.0004 | 0.2150 | n/a | n/a | 0.8988 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0043 | 0.0041 | 0.0048 | 0.0003 | 0.2150 | n/a | n/a | 0.8988 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0174 | 0.0185 | 0.0164 | 0.0218 | 0.0020 | 0.2150 | n/a | n/a | 0.8988 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0150 | 0.0149 | 0.0133 | 0.0164 | 0.0013 | 0.8339 | n/a | n/a | 0.9340 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0125 | 0.0133 | 0.0122 | 0.0169 | 0.0018 | 0.8339 | n/a | n/a | 0.9340 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0140 | 0.0138 | 0.0129 | 0.0148 | 0.0006 | 0.8339 | n/a | n/a | 0.9340 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0297 | 0.0302 | 0.0291 | 0.0314 | 0.0010 | 0.2639 | n/a | n/a | 0.9581 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0078 | 0.0080 | 0.0077 | 0.0088 | 0.0004 | 0.2639 | n/a | n/a | 0.9581 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0284 | 0.0268 | 0.0184 | 0.0387 | 0.0075 | 0.2639 | n/a | n/a | 0.9581 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0223 | 0.0235 | 0.0222 | 0.0274 | 0.0020 | 0.6817 | n/a | n/a | 0.5917 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0152 | 0.0155 | 0.0150 | 0.0161 | 0.0005 | 0.6817 | n/a | n/a | 0.5917 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0132 | 0.0142 | 0.0127 | 0.0175 | 0.0018 | 0.6817 | n/a | n/a | 0.5917 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0790 | 0.0812 | 0.0781 | 0.0904 | 0.0046 | 0.2625 | n/a | n/a | 0.8673 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0207 | 0.0213 | 0.0207 | 0.0234 | 0.0011 | 0.2625 | n/a | n/a | 0.8673 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0685 | 0.0685 | 0.0628 | 0.0747 | 0.0043 | 0.2625 | n/a | n/a | 0.8673 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0570 | 0.0573 | 0.0565 | 0.0589 | 0.0008 | 0.3790 | n/a | n/a | 0.4188 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0216 | 0.0226 | 0.0213 | 0.0253 | 0.0015 | 0.3790 | n/a | n/a | 0.4188 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0239 | 0.0239 | 0.0235 | 0.0245 | 0.0003 | 0.3790 | n/a | n/a | 0.4188 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0187 | 0.0197 | 0.0181 | 0.0238 | 0.0021 | 0.1058 | n/a | n/a | 0.9826 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0020 | 0.0000 | 0.1058 | n/a | n/a | 0.9826 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0184 | 0.0177 | 0.0123 | 0.0208 | 0.0029 | 0.1058 | n/a | n/a | 0.9826 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1185 | 0.1240 | 0.1147 | 0.1365 | 0.0091 | 0.0889 | n/a | n/a | 0.2661 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0105 | 0.0105 | 0.0105 | 0.0106 | 0.0000 | 0.0889 | n/a | n/a | 0.2661 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0315 | 0.0370 | 0.0175 | 0.0774 | 0.0213 | 0.0889 | n/a | n/a | 0.2661 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3874 | 0.3871 | 0.3746 | 0.4044 | 0.0104 | 0.8430 | n/a | n/a | 0.3409 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3266 | 0.3307 | 0.3050 | 0.3620 | 0.0216 | 0.8430 | n/a | n/a | 0.3409 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1321 | 0.1180 | 0.0762 | 0.1335 | 0.0219 | 0.8430 | n/a | n/a | 0.3409 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.0731 | 2.1048 | 2.0153 | 2.2768 | 0.0969 | 0.1938 | n/a | n/a | 0.1204 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4017 | 0.4254 | 0.3893 | 0.4789 | 0.0362 | 0.1938 | n/a | n/a | 0.1204 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2497 | 0.2529 | 0.2330 | 0.2843 | 0.0181 | 0.1938 | n/a | n/a | 0.1204 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2028 | 0.2033 | 0.1908 | 0.2141 | 0.0076 | 6.7128 | n/a | n/a | 0.6012 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.3615 | 1.3923 | 1.2244 | 1.6188 | 0.1358 | 6.7128 | n/a | n/a | 0.6012 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1219 | 0.1144 | 0.0952 | 0.1314 | 0.0152 | 6.7128 | n/a | n/a | 0.6012 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 6.6071 | 6.6995 | 6.4560 | 7.0825 | 0.2350 | 9.4038 | n/a | n/a | 0.0519 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 62.1320 | 62.5393 | 61.8989 | 63.8061 | 0.7189 | 9.4038 | n/a | n/a | 0.0519 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3431 | 0.3401 | 0.3286 | 0.3509 | 0.0079 | 9.4038 | n/a | n/a | 0.0519 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0279 | 0.0279 | 0.0273 | 0.0286 | 0.0004 | 6.2464 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1740 | 0.1719 | 0.1676 | 0.1743 | 0.0028 | 6.2464 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0275 | 0.0277 | 0.0266 | 0.0287 | 0.0007 | 21.4893 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5906 | 0.5903 | 0.5697 | 0.6054 | 0.0117 | 21.4893 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7123 | 0.7092 | 0.6789 | 0.7293 | 0.0177 | 11.5985 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.2617 | 8.2440 | 8.1379 | 8.3181 | 0.0705 | 11.5985 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7308 | 0.7285 | 0.7121 | 0.7406 | 0.0105 | 37.9785 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 27.7535 | 28.4120 | 27.2748 | 29.9413 | 1.0943 | 37.9785 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1148 | 0.1154 | 0.1142 | 0.1167 | 0.0011 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2942 | 0.2958 | 0.2912 | 0.3039 | 0.0046 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2456 | 0.2479 | 0.2432 | 0.2585 | 0.0055 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.5991 | 0.6001 | 0.5950 | 0.6054 | 0.0038 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1506 | 1.1562 | 1.1429 | 1.1895 | 0.0170 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7546 | 2.7596 | 2.7154 | 2.8115 | 0.0388 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.3211 | 5.3278 | 5.1900 | 5.4988 | 0.0987 | 0.0298 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1585 | 0.1617 | 0.1559 | 0.1747 | 0.0068 | 0.0298 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
