# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.16.0`
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
- TensorStudio wins versus JAX CPU dispatch: `48`
- TensorStudio losses versus JAX CPU dispatch: `50`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0166 | 0.0003 | 0.0410 | n/a | n/a | 0.7528 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0410 | n/a | n/a | 0.7528 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0119 | 0.0118 | 0.0115 | 0.0121 | 0.0002 | 0.0410 | n/a | n/a | 0.7528 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0161 | 0.0161 | 0.0161 | 0.0162 | 0.0000 | 0.0438 | n/a | n/a | 0.7385 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0438 | n/a | n/a | 0.7385 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0119 | 0.0120 | 0.0118 | 0.0126 | 0.0003 | 0.0438 | n/a | n/a | 0.7385 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0159 | 0.0160 | 0.0159 | 0.0162 | 0.0001 | 0.0429 | n/a | n/a | 0.7699 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0429 | n/a | n/a | 0.7699 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0123 | 0.0124 | 0.0122 | 0.0133 | 0.0004 | 0.0429 | n/a | n/a | 0.7699 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0162 | 0.0162 | 0.0158 | 0.0167 | 0.0003 | 0.0459 | n/a | n/a | 0.5446 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.0459 | n/a | n/a | 0.5446 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0088 | 0.0090 | 0.0086 | 0.0101 | 0.0006 | 0.0459 | n/a | n/a | 0.5446 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0855 | 0.0855 | 0.0851 | 0.0857 | 0.0002 | 0.0637 | n/a | n/a | 1.0665 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0054 | 0.0055 | 0.0053 | 0.0056 | 0.0001 | 0.0637 | n/a | n/a | 1.0665 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0912 | 0.0918 | 0.0899 | 0.0942 | 0.0017 | 0.0637 | n/a | n/a | 1.0665 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0160 | 0.0162 | 0.0158 | 0.0169 | 0.0005 | 0.0402 | n/a | n/a | 0.7470 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0007 | 0.0000 | 0.0402 | n/a | n/a | 0.7470 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0119 | 0.0116 | 0.0123 | 0.0003 | 0.0402 | n/a | n/a | 0.7470 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0161 | 0.0162 | 0.0159 | 0.0166 | 0.0003 | 0.0437 | n/a | n/a | 0.7416 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0437 | n/a | n/a | 0.7416 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0121 | 0.0118 | 0.0128 | 0.0004 | 0.0437 | n/a | n/a | 0.7416 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0159 | 0.0159 | 0.0158 | 0.0160 | 0.0001 | 0.0435 | n/a | n/a | 0.7539 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0435 | n/a | n/a | 0.7539 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0121 | 0.0116 | 0.0129 | 0.0004 | 0.0435 | n/a | n/a | 0.7539 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0161 | 0.0161 | 0.0159 | 0.0164 | 0.0002 | 0.0447 | n/a | n/a | 0.5586 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.0447 | n/a | n/a | 0.5586 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0090 | 0.0091 | 0.0087 | 0.0100 | 0.0005 | 0.0447 | n/a | n/a | 0.5586 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0847 | 0.0852 | 0.0844 | 0.0873 | 0.0011 | 0.0637 | n/a | n/a | 1.0782 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0054 | 0.0055 | 0.0054 | 0.0057 | 0.0001 | 0.0637 | n/a | n/a | 1.0782 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0914 | 0.0914 | 0.0908 | 0.0922 | 0.0005 | 0.0637 | n/a | n/a | 1.0782 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0158 | 0.0158 | 0.0158 | 0.0159 | 0.0001 | 0.0410 | n/a | n/a | 0.7530 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0007 | 0.0000 | 0.0410 | n/a | n/a | 0.7530 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0119 | 0.0120 | 0.0117 | 0.0123 | 0.0002 | 0.0410 | n/a | n/a | 0.7530 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0163 | 0.0166 | 0.0159 | 0.0174 | 0.0006 | 0.0412 | n/a | n/a | 0.7198 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0412 | n/a | n/a | 0.7198 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0117 | 0.0117 | 0.0114 | 0.0118 | 0.0001 | 0.0412 | n/a | n/a | 0.7198 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0162 | 0.0162 | 0.0160 | 0.0166 | 0.0002 | 0.0474 | n/a | n/a | 0.7284 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.0474 | n/a | n/a | 0.7284 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0118 | 0.0118 | 0.0117 | 0.0119 | 0.0001 | 0.0474 | n/a | n/a | 0.7284 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0159 | 0.0159 | 0.0159 | 0.0160 | 0.0000 | 0.0451 | n/a | n/a | 0.5601 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0451 | n/a | n/a | 0.5601 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0089 | 0.0090 | 0.0088 | 0.0094 | 0.0002 | 0.0451 | n/a | n/a | 0.5601 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0855 | 0.0855 | 0.0850 | 0.0864 | 0.0005 | 0.0644 | n/a | n/a | 1.0876 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0055 | 0.0055 | 0.0054 | 0.0057 | 0.0001 | 0.0644 | n/a | n/a | 1.0876 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0930 | 0.0934 | 0.0914 | 0.0951 | 0.0013 | 0.0644 | n/a | n/a | 1.0876 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0160 | 0.0161 | 0.0158 | 0.0167 | 0.0003 | 0.0412 | n/a | n/a | 0.7311 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0412 | n/a | n/a | 0.7311 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0117 | 0.0118 | 0.0115 | 0.0120 | 0.0002 | 0.0412 | n/a | n/a | 0.7311 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0161 | 0.0177 | 0.0160 | 0.0243 | 0.0033 | 0.0553 | n/a | n/a | 0.7545 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0009 | 0.0012 | 0.0008 | 0.0017 | 0.0004 | 0.0553 | n/a | n/a | 0.7545 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0121 | 0.0130 | 0.0118 | 0.0150 | 0.0013 | 0.0553 | n/a | n/a | 0.7545 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0159 | 0.0159 | 0.0158 | 0.0163 | 0.0002 | 0.0440 | n/a | n/a | 0.7459 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0440 | n/a | n/a | 0.7459 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0119 | 0.0119 | 0.0118 | 0.0121 | 0.0001 | 0.0440 | n/a | n/a | 0.7459 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0160 | 0.0160 | 0.0159 | 0.0161 | 0.0001 | 0.0454 | n/a | n/a | 0.6746 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0454 | n/a | n/a | 0.6746 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0108 | 0.0132 | 0.0088 | 0.0203 | 0.0042 | 0.0454 | n/a | n/a | 0.6746 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0852 | 0.0858 | 0.0851 | 0.0876 | 0.0010 | 0.0662 | n/a | n/a | 1.0874 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0056 | 0.0056 | 0.0055 | 0.0058 | 0.0001 | 0.0662 | n/a | n/a | 1.0874 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0926 | 0.0927 | 0.0907 | 0.0949 | 0.0014 | 0.0662 | n/a | n/a | 1.0874 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0159 | 0.0161 | 0.0158 | 0.0165 | 0.0003 | 0.0767 | n/a | n/a | 1.1164 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0012 | 0.0013 | 0.0010 | 0.0019 | 0.0003 | 0.0767 | n/a | n/a | 1.1164 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0177 | 0.0182 | 0.0141 | 0.0212 | 0.0026 | 0.0767 | n/a | n/a | 1.1164 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0161 | 0.0163 | 0.0159 | 0.0169 | 0.0004 | 0.0631 | n/a | n/a | 1.1786 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0015 | 0.0002 | 0.0631 | n/a | n/a | 1.1786 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0190 | 0.0187 | 0.0168 | 0.0201 | 0.0012 | 0.0631 | n/a | n/a | 1.1786 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0170 | 0.0166 | 0.0158 | 0.0171 | 0.0006 | 0.0625 | n/a | n/a | 0.9432 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0011 | 0.0000 | 0.0625 | n/a | n/a | 0.9432 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0160 | 0.0173 | 0.0134 | 0.0220 | 0.0030 | 0.0625 | n/a | n/a | 0.9432 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0164 | 0.0172 | 0.0159 | 0.0210 | 0.0019 | 0.0724 | n/a | n/a | 0.8081 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0012 | 0.0013 | 0.0011 | 0.0017 | 0.0003 | 0.0724 | n/a | n/a | 0.8081 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0133 | 0.0124 | 0.0102 | 0.0142 | 0.0015 | 0.0724 | n/a | n/a | 0.8081 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0861 | 0.0864 | 0.0858 | 0.0873 | 0.0006 | 0.0856 | n/a | n/a | 1.4036 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0074 | 0.0075 | 0.0074 | 0.0081 | 0.0003 | 0.0856 | n/a | n/a | 1.4036 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1209 | 0.1203 | 0.1099 | 0.1282 | 0.0059 | 0.0856 | n/a | n/a | 1.4036 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0174 | 0.0173 | 0.0169 | 0.0179 | 0.0004 | 0.0824 | n/a | n/a | 0.9720 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.0824 | n/a | n/a | 0.9720 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0169 | 0.0172 | 0.0156 | 0.0188 | 0.0013 | 0.0824 | n/a | n/a | 0.9720 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0183 | 0.0181 | 0.0168 | 0.0201 | 0.0012 | 0.0790 | n/a | n/a | 1.0193 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.0790 | n/a | n/a | 1.0193 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0186 | 0.0186 | 0.0144 | 0.0224 | 0.0025 | 0.0790 | n/a | n/a | 1.0193 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0172 | 0.0172 | 0.0169 | 0.0175 | 0.0002 | 0.0922 | n/a | n/a | 1.0891 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.0922 | n/a | n/a | 1.0891 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0187 | 0.0190 | 0.0156 | 0.0220 | 0.0024 | 0.0922 | n/a | n/a | 1.0891 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0176 | 0.0177 | 0.0172 | 0.0184 | 0.0005 | 0.0944 | n/a | n/a | 0.6871 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.0944 | n/a | n/a | 0.6871 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0121 | 0.0120 | 0.0105 | 0.0133 | 0.0010 | 0.0944 | n/a | n/a | 0.6871 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0906 | 0.0906 | 0.0899 | 0.0912 | 0.0005 | 0.1147 | n/a | n/a | 1.3196 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0104 | 0.0106 | 0.0103 | 0.0113 | 0.0004 | 0.1147 | n/a | n/a | 1.3196 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1195 | 0.1225 | 0.1144 | 0.1347 | 0.0072 | 0.1147 | n/a | n/a | 1.3196 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0212 | 0.0214 | 0.0211 | 0.0223 | 0.0004 | 0.2093 | n/a | n/a | 1.2821 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0044 | 0.0046 | 0.0038 | 0.0054 | 0.0006 | 0.2093 | n/a | n/a | 1.2821 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0272 | 0.0259 | 0.0150 | 0.0374 | 0.0073 | 0.2093 | n/a | n/a | 1.2821 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0213 | 0.0222 | 0.0211 | 0.0259 | 0.0019 | 0.1814 | n/a | n/a | 0.7165 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0040 | 0.0038 | 0.0042 | 0.0002 | 0.1814 | n/a | n/a | 0.7165 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0153 | 0.0184 | 0.0142 | 0.0311 | 0.0064 | 0.1814 | n/a | n/a | 0.7165 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0217 | 0.0215 | 0.0212 | 0.0218 | 0.0002 | 0.1724 | n/a | n/a | 1.0305 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0038 | 0.0037 | 0.0039 | 0.0001 | 0.1724 | n/a | n/a | 1.0305 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0223 | 0.0218 | 0.0149 | 0.0288 | 0.0059 | 0.1724 | n/a | n/a | 1.0305 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0211 | 0.0213 | 0.0209 | 0.0225 | 0.0006 | 0.2097 | n/a | n/a | 0.5811 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0001 | 0.2097 | n/a | n/a | 0.5811 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0122 | 0.0132 | 0.0121 | 0.0170 | 0.0019 | 0.2097 | n/a | n/a | 0.5811 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1067 | 0.1064 | 0.1044 | 0.1085 | 0.0013 | 0.2195 | n/a | n/a | 1.3857 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0234 | 0.0233 | 0.0226 | 0.0240 | 0.0005 | 0.2195 | n/a | n/a | 1.3857 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1479 | 0.1483 | 0.1421 | 0.1551 | 0.0054 | 0.2195 | n/a | n/a | 1.3857 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0149 | 0.0151 | 0.0149 | 0.0157 | 0.0003 | 0.1172 | n/a | n/a | 3.0349 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 0.1172 | n/a | n/a | 3.0349 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0452 | 0.0452 | 0.0445 | 0.0461 | 0.0005 | 0.1172 | n/a | n/a | 3.0349 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0151 | 0.0151 | 0.0150 | 0.0152 | 0.0001 | 0.2920 | n/a | n/a | 4.9372 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0044 | 0.0000 | 0.2920 | n/a | n/a | 4.9372 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0747 | 0.0748 | 0.0739 | 0.0762 | 0.0008 | 0.2920 | n/a | n/a | 4.9372 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0151 | 0.0151 | 0.0149 | 0.0152 | 0.0001 | 0.0823 | n/a | n/a | 2.0279 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0823 | n/a | n/a | 2.0279 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0305 | 0.0310 | 0.0293 | 0.0344 | 0.0018 | 0.0823 | n/a | n/a | 2.0279 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0150 | 0.0150 | 0.0150 | 0.0151 | 0.0000 | 0.0815 | n/a | n/a | 2.0055 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0014 | 0.0001 | 0.0815 | n/a | n/a | 2.0055 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0301 | 0.0302 | 0.0297 | 0.0310 | 0.0005 | 0.0815 | n/a | n/a | 2.0055 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0151 | 0.0151 | 0.0150 | 0.0152 | 0.0001 | 0.0861 | n/a | n/a | 1.9972 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0014 | 0.0000 | 0.0861 | n/a | n/a | 1.9972 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0302 | 0.0301 | 0.0296 | 0.0306 | 0.0003 | 0.0861 | n/a | n/a | 1.9972 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0151 | 0.0151 | 0.0148 | 0.0156 | 0.0003 | 0.1151 | n/a | n/a | 2.9570 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 0.1151 | n/a | n/a | 2.9570 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0446 | 0.0447 | 0.0435 | 0.0465 | 0.0010 | 0.1151 | n/a | n/a | 2.9570 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0150 | 0.0151 | 0.0150 | 0.0155 | 0.0002 | 0.2914 | n/a | n/a | 4.8863 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0044 | 0.0044 | 0.0044 | 0.0046 | 0.0001 | 0.2914 | n/a | n/a | 4.8863 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0735 | 0.0739 | 0.0728 | 0.0769 | 0.0015 | 0.2914 | n/a | n/a | 4.8863 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0151 | 0.0151 | 0.0150 | 0.0151 | 0.0000 | 0.0821 | n/a | n/a | 2.0231 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0821 | n/a | n/a | 2.0231 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0305 | 0.0304 | 0.0296 | 0.0310 | 0.0005 | 0.0821 | n/a | n/a | 2.0231 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0151 | 0.0152 | 0.0151 | 0.0152 | 0.0001 | 0.0781 | n/a | n/a | 1.9953 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.0781 | n/a | n/a | 1.9953 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0302 | 0.0303 | 0.0299 | 0.0306 | 0.0003 | 0.0781 | n/a | n/a | 1.9953 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0152 | 0.0154 | 0.0151 | 0.0161 | 0.0004 | 0.0824 | n/a | n/a | 1.9910 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0824 | n/a | n/a | 1.9910 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0303 | 0.0304 | 0.0296 | 0.0316 | 0.0007 | 0.0824 | n/a | n/a | 1.9910 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0149 | 0.0149 | 0.0149 | 0.0150 | 0.0000 | 0.1161 | n/a | n/a | 3.0298 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 0.1161 | n/a | n/a | 3.0298 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0453 | 0.0452 | 0.0448 | 0.0458 | 0.0004 | 0.1161 | n/a | n/a | 3.0298 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0154 | 0.0154 | 0.0152 | 0.0156 | 0.0001 | 0.2895 | n/a | n/a | 4.8297 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0044 | 0.0045 | 0.0044 | 0.0048 | 0.0002 | 0.2895 | n/a | n/a | 4.8297 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0742 | 0.0745 | 0.0725 | 0.0759 | 0.0012 | 0.2895 | n/a | n/a | 4.8297 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0153 | 0.0155 | 0.0153 | 0.0161 | 0.0003 | 0.0831 | n/a | n/a | 2.0040 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.0831 | n/a | n/a | 2.0040 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0307 | 0.0305 | 0.0297 | 0.0314 | 0.0006 | 0.0831 | n/a | n/a | 2.0040 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0155 | 0.0157 | 0.0150 | 0.0164 | 0.0005 | 0.0843 | n/a | n/a | 1.9501 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0843 | n/a | n/a | 1.9501 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0302 | 0.0305 | 0.0296 | 0.0315 | 0.0007 | 0.0843 | n/a | n/a | 1.9501 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0154 | 0.0154 | 0.0152 | 0.0156 | 0.0002 | 0.0836 | n/a | n/a | 2.0666 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0015 | 0.0001 | 0.0836 | n/a | n/a | 2.0666 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0318 | 0.0319 | 0.0298 | 0.0340 | 0.0014 | 0.0836 | n/a | n/a | 2.0666 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0151 | 0.0154 | 0.0149 | 0.0166 | 0.0006 | 0.1186 | n/a | n/a | 3.0494 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0021 | 0.0001 | 0.1186 | n/a | n/a | 3.0494 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0461 | 0.0467 | 0.0445 | 0.0502 | 0.0019 | 0.1186 | n/a | n/a | 3.0494 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0157 | 0.0162 | 0.0155 | 0.0175 | 0.0008 | 0.2994 | n/a | n/a | 4.6598 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0047 | 0.0048 | 0.0045 | 0.0052 | 0.0003 | 0.2994 | n/a | n/a | 4.6598 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0730 | 0.0733 | 0.0719 | 0.0754 | 0.0012 | 0.2994 | n/a | n/a | 4.6598 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0164 | 0.0166 | 0.0163 | 0.0173 | 0.0004 | 0.0940 | n/a | n/a | 1.8753 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0021 | 0.0002 | 0.0940 | n/a | n/a | 1.8753 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0308 | 0.0305 | 0.0298 | 0.0311 | 0.0005 | 0.0940 | n/a | n/a | 1.8753 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0161 | 0.0001 | 0.0875 | n/a | n/a | 1.9153 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0015 | 0.0000 | 0.0875 | n/a | n/a | 1.9153 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0303 | 0.0304 | 0.0301 | 0.0308 | 0.0002 | 0.0875 | n/a | n/a | 1.9153 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0158 | 0.0158 | 0.0157 | 0.0158 | 0.0000 | 0.0921 | n/a | n/a | 1.9185 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.0921 | n/a | n/a | 1.9185 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0302 | 0.0306 | 0.0299 | 0.0321 | 0.0008 | 0.0921 | n/a | n/a | 1.9185 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0156 | 0.0156 | 0.0154 | 0.0158 | 0.0002 | 0.1531 | n/a | n/a | 3.8421 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0001 | 0.1531 | n/a | n/a | 3.8421 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0598 | 0.0616 | 0.0582 | 0.0671 | 0.0033 | 0.1531 | n/a | n/a | 3.8421 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0197 | 0.0198 | 0.0193 | 0.0205 | 0.0005 | 0.3633 | n/a | n/a | 4.5207 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0072 | 0.0072 | 0.0067 | 0.0080 | 0.0004 | 0.3633 | n/a | n/a | 4.5207 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0891 | 0.0907 | 0.0865 | 0.0980 | 0.0040 | 0.3633 | n/a | n/a | 4.5207 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0264 | 0.0270 | 0.0255 | 0.0309 | 0.0020 | 0.1485 | n/a | n/a | 1.1584 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0039 | 0.0041 | 0.0037 | 0.0047 | 0.0004 | 0.1485 | n/a | n/a | 1.1584 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0306 | 0.0307 | 0.0304 | 0.0311 | 0.0003 | 0.1485 | n/a | n/a | 1.1584 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0193 | 0.0194 | 0.0189 | 0.0203 | 0.0005 | 0.1354 | n/a | n/a | 1.6157 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0001 | 0.1354 | n/a | n/a | 1.6157 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0313 | 0.0313 | 0.0307 | 0.0318 | 0.0004 | 0.1354 | n/a | n/a | 1.6157 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0210 | 0.0209 | 0.0204 | 0.0213 | 0.0003 | 0.1370 | n/a | n/a | 1.4856 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0029 | 0.0029 | 0.0000 | 0.1370 | n/a | n/a | 1.4856 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0312 | 0.0313 | 0.0307 | 0.0322 | 0.0005 | 0.1370 | n/a | n/a | 1.4856 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0169 | 0.0174 | 0.0166 | 0.0198 | 0.0012 | 0.2195 | n/a | n/a | 3.8155 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0037 | 0.0037 | 0.0036 | 0.0040 | 0.0002 | 0.2195 | n/a | n/a | 3.8155 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0645 | 0.0684 | 0.0621 | 0.0806 | 0.0070 | 0.2195 | n/a | n/a | 3.8155 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0340 | 0.0339 | 0.0335 | 0.0343 | 0.0004 | 0.3567 | n/a | n/a | 3.0779 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0121 | 0.0124 | 0.0119 | 0.0131 | 0.0005 | 0.3567 | n/a | n/a | 3.0779 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1047 | 0.1068 | 0.1017 | 0.1143 | 0.0046 | 0.3567 | n/a | n/a | 3.0779 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0644 | 0.0649 | 0.0629 | 0.0672 | 0.0017 | 0.1720 | n/a | n/a | 0.5180 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0111 | 0.0112 | 0.0109 | 0.0116 | 0.0003 | 0.1720 | n/a | n/a | 0.5180 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0333 | 0.0337 | 0.0332 | 0.0352 | 0.0008 | 0.1720 | n/a | n/a | 0.5180 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0329 | 0.0328 | 0.0324 | 0.0332 | 0.0003 | 0.1902 | n/a | n/a | 1.0387 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0063 | 0.0064 | 0.0062 | 0.0068 | 0.0002 | 0.1902 | n/a | n/a | 1.0387 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0342 | 0.0341 | 0.0334 | 0.0344 | 0.0004 | 0.1902 | n/a | n/a | 1.0387 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0373 | 0.0373 | 0.0370 | 0.0376 | 0.0002 | 0.1969 | n/a | n/a | 0.9745 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0073 | 0.0074 | 0.0073 | 0.0076 | 0.0001 | 0.1969 | n/a | n/a | 0.9745 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0364 | 0.0362 | 0.0352 | 0.0366 | 0.0005 | 0.1969 | n/a | n/a | 0.9745 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0223 | 0.0223 | 0.0220 | 0.0225 | 0.0002 | 0.3936 | n/a | n/a | 4.3166 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0088 | 0.0096 | 0.0086 | 0.0131 | 0.0018 | 0.3936 | n/a | n/a | 4.3166 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0961 | 0.0927 | 0.0739 | 0.1086 | 0.0129 | 0.3936 | n/a | n/a | 4.3166 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0877 | 0.0880 | 0.0872 | 0.0891 | 0.0007 | 0.3801 | n/a | n/a | 1.3719 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0333 | 0.0332 | 0.0327 | 0.0336 | 0.0004 | 0.3801 | n/a | n/a | 1.3719 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1203 | 0.1227 | 0.1094 | 0.1411 | 0.0105 | 0.3801 | n/a | n/a | 1.3719 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2149 | 0.2156 | 0.2147 | 0.2177 | 0.0012 | 0.1782 | n/a | n/a | 0.1822 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0383 | 0.0392 | 0.0380 | 0.0426 | 0.0017 | 0.1782 | n/a | n/a | 0.1822 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0392 | 0.0395 | 0.0386 | 0.0417 | 0.0011 | 0.1782 | n/a | n/a | 0.1822 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0825 | 0.0829 | 0.0803 | 0.0860 | 0.0021 | 0.2391 | n/a | n/a | 0.4919 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0197 | 0.0200 | 0.0194 | 0.0209 | 0.0006 | 0.2391 | n/a | n/a | 0.4919 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0406 | 0.0409 | 0.0399 | 0.0425 | 0.0009 | 0.2391 | n/a | n/a | 0.4919 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1013 | 0.1025 | 0.1009 | 0.1064 | 0.0021 | 0.2495 | n/a | n/a | 0.4395 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0253 | 0.0254 | 0.0250 | 0.0263 | 0.0005 | 0.2495 | n/a | n/a | 0.4395 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0445 | 0.0448 | 0.0435 | 0.0464 | 0.0010 | 0.2495 | n/a | n/a | 0.4395 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0155 | 0.0158 | 0.0153 | 0.0173 | 0.0007 | 0.1490 | n/a | n/a | 0.7376 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0023 | 0.0001 | 0.1490 | n/a | n/a | 0.7376 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0114 | 0.0114 | 0.0113 | 0.0115 | 0.0001 | 0.1490 | n/a | n/a | 0.7376 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0154 | 0.0155 | 0.0154 | 0.0155 | 0.0001 | 0.5127 | n/a | n/a | 0.7259 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0079 | 0.0079 | 0.0077 | 0.0082 | 0.0002 | 0.5127 | n/a | n/a | 0.7259 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0112 | 0.0115 | 0.0111 | 0.0124 | 0.0005 | 0.5127 | n/a | n/a | 0.7259 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0156 | 0.0156 | 0.0154 | 0.0158 | 0.0002 | 0.1525 | n/a | n/a | 0.7803 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0024 | 0.0026 | 0.0022 | 0.0031 | 0.0004 | 0.1525 | n/a | n/a | 0.7803 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0122 | 0.0126 | 0.0120 | 0.0142 | 0.0008 | 0.1525 | n/a | n/a | 0.7803 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0155 | 0.0155 | 0.0154 | 0.0157 | 0.0001 | 0.4913 | n/a | n/a | 0.7280 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0076 | 0.0077 | 0.0076 | 0.0081 | 0.0002 | 0.4913 | n/a | n/a | 0.7280 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0113 | 0.0113 | 0.0112 | 0.0114 | 0.0001 | 0.4913 | n/a | n/a | 0.7280 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0155 | 0.0188 | 0.0154 | 0.0291 | 0.0053 | 0.1481 | n/a | n/a | 0.8084 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0025 | 0.0001 | 0.1481 | n/a | n/a | 0.8084 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0126 | 0.0129 | 0.0121 | 0.0138 | 0.0007 | 0.1481 | n/a | n/a | 0.8084 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0155 | 0.0155 | 0.0155 | 0.0157 | 0.0001 | 0.5273 | n/a | n/a | 0.7478 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0082 | 0.0080 | 0.0077 | 0.0083 | 0.0003 | 0.5273 | n/a | n/a | 0.7478 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0116 | 0.0126 | 0.0111 | 0.0171 | 0.0022 | 0.5273 | n/a | n/a | 0.7478 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0157 | 0.0158 | 0.0157 | 0.0159 | 0.0001 | 0.1470 | n/a | n/a | 0.8937 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0000 | 0.1470 | n/a | n/a | 0.8937 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0141 | 0.0140 | 0.0137 | 0.0143 | 0.0002 | 0.1470 | n/a | n/a | 0.8937 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0157 | 0.0158 | 0.0156 | 0.0161 | 0.0002 | 0.5029 | n/a | n/a | 0.7743 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0079 | 0.0081 | 0.0078 | 0.0092 | 0.0005 | 0.5029 | n/a | n/a | 0.7743 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0121 | 0.0129 | 0.0120 | 0.0160 | 0.0016 | 0.5029 | n/a | n/a | 0.7743 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0166 | 0.0167 | 0.0163 | 0.0174 | 0.0004 | 0.1620 | n/a | n/a | 1.0405 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0027 | 0.0028 | 0.0026 | 0.0030 | 0.0002 | 0.1620 | n/a | n/a | 1.0405 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0172 | 0.0182 | 0.0166 | 0.0230 | 0.0024 | 0.1620 | n/a | n/a | 1.0405 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0170 | 0.0169 | 0.0162 | 0.0176 | 0.0006 | 0.4923 | n/a | n/a | 1.1980 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0083 | 0.0088 | 0.0082 | 0.0107 | 0.0010 | 0.4923 | n/a | n/a | 1.1980 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0203 | 0.0193 | 0.0143 | 0.0215 | 0.0026 | 0.4923 | n/a | n/a | 1.1980 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0205 | 0.0203 | 0.0196 | 0.0208 | 0.0005 | 0.1764 | n/a | n/a | 0.9780 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0039 | 0.0001 | 0.1764 | n/a | n/a | 0.9780 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0200 | 0.0189 | 0.0151 | 0.0206 | 0.0020 | 0.1764 | n/a | n/a | 0.9780 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0201 | 0.0201 | 0.0197 | 0.0206 | 0.0003 | 0.4588 | n/a | n/a | 0.8359 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0092 | 0.0093 | 0.0090 | 0.0097 | 0.0002 | 0.4588 | n/a | n/a | 0.8359 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0168 | 0.0165 | 0.0142 | 0.0188 | 0.0016 | 0.4588 | n/a | n/a | 0.8359 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0337 | 0.0335 | 0.0326 | 0.0342 | 0.0006 | 0.1980 | n/a | n/a | 0.7442 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0067 | 0.0067 | 0.0065 | 0.0069 | 0.0001 | 0.1980 | n/a | n/a | 0.7442 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0251 | 0.0260 | 0.0224 | 0.0289 | 0.0024 | 0.1980 | n/a | n/a | 0.7442 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0334 | 0.0336 | 0.0330 | 0.0344 | 0.0005 | 0.3892 | n/a | n/a | 0.8234 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0130 | 0.0134 | 0.0128 | 0.0151 | 0.0008 | 0.3892 | n/a | n/a | 0.8234 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0275 | 0.0317 | 0.0242 | 0.0410 | 0.0072 | 0.3892 | n/a | n/a | 0.8234 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0158 | 0.0160 | 0.0157 | 0.0168 | 0.0004 | 0.1928 | n/a | n/a | 0.7955 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0030 | 0.0033 | 0.0029 | 0.0046 | 0.0007 | 0.1928 | n/a | n/a | 0.7955 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0126 | 0.0133 | 0.0122 | 0.0166 | 0.0017 | 0.1928 | n/a | n/a | 0.7955 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0104 | 0.0111 | 0.0101 | 0.0140 | 0.0015 | 0.9665 | n/a | n/a | 1.1091 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0100 | 0.0100 | 0.0098 | 0.0103 | 0.0002 | 0.9665 | n/a | n/a | 1.1091 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0115 | 0.0119 | 0.0113 | 0.0133 | 0.0008 | 0.9665 | n/a | n/a | 1.1091 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0196 | 0.0194 | 0.0187 | 0.0199 | 0.0004 | 0.2092 | n/a | n/a | 1.0543 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0041 | 0.0045 | 0.0041 | 0.0053 | 0.0005 | 0.2092 | n/a | n/a | 1.0543 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0206 | 0.0196 | 0.0150 | 0.0237 | 0.0034 | 0.2092 | n/a | n/a | 1.0543 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0133 | 0.0135 | 0.0132 | 0.0141 | 0.0003 | 0.8746 | n/a | n/a | 1.4959 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0116 | 0.0126 | 0.0114 | 0.0164 | 0.0019 | 0.8746 | n/a | n/a | 1.4959 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0199 | 0.0200 | 0.0162 | 0.0237 | 0.0026 | 0.8746 | n/a | n/a | 1.4959 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0303 | 0.0305 | 0.0298 | 0.0319 | 0.0008 | 0.2548 | n/a | n/a | 0.8067 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0077 | 0.0080 | 0.0076 | 0.0089 | 0.0005 | 0.2548 | n/a | n/a | 0.8067 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0244 | 0.0248 | 0.0202 | 0.0301 | 0.0033 | 0.2548 | n/a | n/a | 0.8067 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0230 | 0.0232 | 0.0229 | 0.0240 | 0.0004 | 0.6623 | n/a | n/a | 0.7153 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0152 | 0.0153 | 0.0148 | 0.0162 | 0.0005 | 0.6623 | n/a | n/a | 0.7153 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0164 | 0.0230 | 0.0134 | 0.0377 | 0.0104 | 0.6623 | n/a | n/a | 0.7153 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0820 | 0.0818 | 0.0809 | 0.0827 | 0.0007 | 0.2639 | n/a | n/a | 0.8548 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0216 | 0.0218 | 0.0216 | 0.0221 | 0.0002 | 0.2639 | n/a | n/a | 0.8548 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0701 | 0.0728 | 0.0670 | 0.0786 | 0.0048 | 0.2639 | n/a | n/a | 0.8548 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0585 | 0.0588 | 0.0580 | 0.0600 | 0.0007 | 0.3785 | n/a | n/a | 0.3385 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0222 | 0.0225 | 0.0217 | 0.0234 | 0.0006 | 0.3785 | n/a | n/a | 0.3385 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0198 | 0.0241 | 0.0183 | 0.0399 | 0.0081 | 0.3785 | n/a | n/a | 0.3385 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0187 | 0.0193 | 0.0183 | 0.0217 | 0.0012 | 0.1021 | n/a | n/a | 0.8941 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0021 | 0.0001 | 0.1021 | n/a | n/a | 0.8941 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0167 | 0.0168 | 0.0112 | 0.0278 | 0.0061 | 0.1021 | n/a | n/a | 0.8941 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0991 | 0.1080 | 0.0969 | 0.1329 | 0.0136 | 0.1091 | n/a | n/a | 0.5174 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0108 | 0.0110 | 0.0104 | 0.0120 | 0.0006 | 0.1091 | n/a | n/a | 0.5174 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0513 | 0.0432 | 0.0175 | 0.0566 | 0.0147 | 0.1091 | n/a | n/a | 0.5174 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3907 | 0.3951 | 0.3870 | 0.4108 | 0.0089 | 1.0716 | n/a | n/a | 0.3095 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.4186 | 0.4014 | 0.3082 | 0.4604 | 0.0508 | 1.0716 | n/a | n/a | 0.3095 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1209 | 0.1211 | 0.1067 | 0.1361 | 0.0119 | 1.0716 | n/a | n/a | 0.3095 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.7970 | 2.6829 | 2.1008 | 3.3534 | 0.4486 | 0.1669 | n/a | n/a | 0.0908 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4668 | 0.4652 | 0.4396 | 0.4982 | 0.0198 | 0.1669 | n/a | n/a | 0.0908 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2540 | 0.2548 | 0.2315 | 0.2996 | 0.0245 | 0.1669 | n/a | n/a | 0.0908 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1938 | 0.1982 | 0.1928 | 0.2141 | 0.0081 | 6.5522 | n/a | n/a | 0.5171 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2695 | 1.2952 | 1.2136 | 1.4738 | 0.0927 | 6.5522 | n/a | n/a | 0.5171 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1002 | 0.1070 | 0.0990 | 0.1279 | 0.0109 | 6.5522 | n/a | n/a | 0.5171 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 5.7194 | 5.7316 | 5.6945 | 5.7900 | 0.0333 | 10.5775 | n/a | n/a | 0.0606 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 60.4972 | 60.9421 | 59.5117 | 62.4298 | 1.1254 | 10.5775 | n/a | n/a | 0.0606 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3467 | 0.3385 | 0.3052 | 0.3706 | 0.0233 | 10.5775 | n/a | n/a | 0.0606 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0282 | 0.0284 | 0.0275 | 0.0297 | 0.0007 | 5.7866 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1632 | 0.1654 | 0.1614 | 0.1758 | 0.0053 | 5.7866 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0277 | 0.0280 | 0.0271 | 0.0300 | 0.0010 | 19.7400 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5474 | 0.5484 | 0.5422 | 0.5590 | 0.0060 | 19.7400 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6982 | 0.6967 | 0.6901 | 0.7030 | 0.0045 | 10.9643 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.6549 | 7.6290 | 7.4842 | 7.7202 | 0.0852 | 10.9643 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6888 | 0.6914 | 0.6819 | 0.7013 | 0.0082 | 38.1835 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 26.3019 | 26.3364 | 26.1829 | 26.5880 | 0.1418 | 38.1835 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1206 | 0.1247 | 0.1193 | 0.1429 | 0.0091 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2985 | 0.2983 | 0.2969 | 0.2990 | 0.0007 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2503 | 0.2506 | 0.2494 | 0.2520 | 0.0009 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.6127 | 0.6143 | 0.6042 | 0.6260 | 0.0078 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1459 | 1.1520 | 1.1426 | 1.1789 | 0.0137 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7217 | 2.7239 | 2.7132 | 2.7417 | 0.0108 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.1662 | 5.1672 | 4.9766 | 5.3213 | 0.1212 | 0.0309 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1596 | 0.1612 | 0.1561 | 0.1679 | 0.0043 | 0.0309 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |
| backends | backend_info | `(1,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0024 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | available_devices | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cpu_transfer | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0014 | 0.0020 | 0.0002 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cuda_availability_check | `(1,)` | TensorStudio | 0.0021 | 0.0021 | 0.0021 | 0.0022 | 0.0001 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
