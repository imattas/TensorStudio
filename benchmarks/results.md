# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.6.0`
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
- TensorStudio wins versus JAX CPU dispatch: `39`
- TensorStudio losses versus JAX CPU dispatch: `59`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0169 | 0.0176 | 0.0165 | 0.0208 | 0.0016 | 0.0397 | n/a | n/a | 0.7255 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0397 | n/a | n/a | 0.7255 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0122 | 0.0121 | 0.0116 | 0.0126 | 0.0005 | 0.0397 | n/a | n/a | 0.7255 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0168 | 0.0168 | 0.0165 | 0.0172 | 0.0003 | 0.0408 | n/a | n/a | 0.7791 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0408 | n/a | n/a | 0.7791 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0131 | 0.0132 | 0.0121 | 0.0149 | 0.0010 | 0.0408 | n/a | n/a | 0.7791 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0165 | 0.0166 | 0.0164 | 0.0169 | 0.0002 | 0.0409 | n/a | n/a | 0.7335 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0409 | n/a | n/a | 0.7335 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0121 | 0.0126 | 0.0120 | 0.0145 | 0.0010 | 0.0409 | n/a | n/a | 0.7335 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0168 | 0.0168 | 0.0164 | 0.0170 | 0.0002 | 0.0419 | n/a | n/a | 0.5577 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0012 | 0.0002 | 0.0419 | n/a | n/a | 0.5577 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0094 | 0.0095 | 0.0090 | 0.0103 | 0.0004 | 0.0419 | n/a | n/a | 0.5577 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0914 | 0.0929 | 0.0883 | 0.1002 | 0.0043 | 0.0635 | n/a | n/a | 1.0708 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0058 | 0.0059 | 0.0054 | 0.0066 | 0.0004 | 0.0635 | n/a | n/a | 1.0708 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0978 | 0.0973 | 0.0882 | 0.1028 | 0.0051 | 0.0635 | n/a | n/a | 1.0708 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0166 | 0.0169 | 0.0164 | 0.0184 | 0.0007 | 0.0391 | n/a | n/a | 0.7774 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0391 | n/a | n/a | 0.7774 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0129 | 0.0138 | 0.0121 | 0.0160 | 0.0017 | 0.0391 | n/a | n/a | 0.7774 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0178 | 0.0178 | 0.0164 | 0.0195 | 0.0012 | 0.0399 | n/a | n/a | 0.6734 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0399 | n/a | n/a | 0.6734 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0120 | 0.0114 | 0.0128 | 0.0005 | 0.0399 | n/a | n/a | 0.6734 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0166 | 0.0169 | 0.0163 | 0.0184 | 0.0008 | 0.0419 | n/a | n/a | 0.7185 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0419 | n/a | n/a | 0.7185 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0124 | 0.0117 | 0.0146 | 0.0011 | 0.0419 | n/a | n/a | 0.7185 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0175 | 0.0182 | 0.0164 | 0.0209 | 0.0018 | 0.0426 | n/a | n/a | 0.5393 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0426 | n/a | n/a | 0.5393 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0095 | 0.0097 | 0.0093 | 0.0107 | 0.0005 | 0.0426 | n/a | n/a | 0.5393 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0922 | 0.0919 | 0.0906 | 0.0927 | 0.0008 | 0.0602 | n/a | n/a | 1.0323 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0056 | 0.0056 | 0.0054 | 0.0058 | 0.0002 | 0.0602 | n/a | n/a | 1.0323 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0952 | 0.0935 | 0.0897 | 0.0962 | 0.0027 | 0.0602 | n/a | n/a | 1.0323 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0166 | 0.0167 | 0.0164 | 0.0171 | 0.0003 | 0.0419 | n/a | n/a | 0.7752 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0419 | n/a | n/a | 0.7752 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0128 | 0.0135 | 0.0119 | 0.0169 | 0.0018 | 0.0419 | n/a | n/a | 0.7752 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0170 | 0.0172 | 0.0166 | 0.0190 | 0.0009 | 0.0443 | n/a | n/a | 0.7339 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0443 | n/a | n/a | 0.7339 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0125 | 0.0121 | 0.0111 | 0.0128 | 0.0006 | 0.0443 | n/a | n/a | 0.7339 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0168 | 0.0174 | 0.0164 | 0.0187 | 0.0011 | 0.0408 | n/a | n/a | 0.7345 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0408 | n/a | n/a | 0.7345 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0123 | 0.0129 | 0.0119 | 0.0153 | 0.0012 | 0.0408 | n/a | n/a | 0.7345 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0165 | 0.0168 | 0.0163 | 0.0182 | 0.0007 | 0.0473 | n/a | n/a | 0.5800 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0473 | n/a | n/a | 0.5800 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0096 | 0.0096 | 0.0091 | 0.0104 | 0.0004 | 0.0473 | n/a | n/a | 0.5800 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0951 | 0.0960 | 0.0887 | 0.1043 | 0.0066 | 0.0621 | n/a | n/a | 1.1597 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0059 | 0.0060 | 0.0055 | 0.0064 | 0.0004 | 0.0621 | n/a | n/a | 1.1597 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.1103 | 0.1065 | 0.0896 | 0.1207 | 0.0137 | 0.0621 | n/a | n/a | 1.1597 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0204 | 0.0226 | 0.0184 | 0.0315 | 0.0048 | 0.0368 | n/a | n/a | 0.6186 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0368 | n/a | n/a | 0.6186 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0126 | 0.0128 | 0.0115 | 0.0145 | 0.0010 | 0.0368 | n/a | n/a | 0.6186 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0174 | 0.0173 | 0.0166 | 0.0183 | 0.0006 | 0.0450 | n/a | n/a | 0.7554 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0450 | n/a | n/a | 0.7554 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0131 | 0.0131 | 0.0119 | 0.0139 | 0.0007 | 0.0450 | n/a | n/a | 0.7554 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0200 | 0.0198 | 0.0166 | 0.0230 | 0.0024 | 0.0366 | n/a | n/a | 0.6412 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0366 | n/a | n/a | 0.6412 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0128 | 0.0133 | 0.0123 | 0.0161 | 0.0014 | 0.0366 | n/a | n/a | 0.6412 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0168 | 0.0173 | 0.0164 | 0.0191 | 0.0010 | 0.0444 | n/a | n/a | 0.5728 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.0444 | n/a | n/a | 0.5728 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0096 | 0.0098 | 0.0095 | 0.0107 | 0.0005 | 0.0444 | n/a | n/a | 0.5728 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0992 | 0.0987 | 0.0919 | 0.1048 | 0.0053 | 0.0636 | n/a | n/a | 1.1219 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0063 | 0.0064 | 0.0059 | 0.0069 | 0.0004 | 0.0636 | n/a | n/a | 1.1219 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.1113 | 0.1144 | 0.0892 | 0.1687 | 0.0290 | 0.0636 | n/a | n/a | 1.1219 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0179 | 0.0179 | 0.0165 | 0.0195 | 0.0010 | 0.0573 | n/a | n/a | 0.9024 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0012 | 0.0010 | 0.0018 | 0.0003 | 0.0573 | n/a | n/a | 0.9024 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0162 | 0.0157 | 0.0132 | 0.0182 | 0.0020 | 0.0573 | n/a | n/a | 0.9024 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0187 | 0.0191 | 0.0160 | 0.0230 | 0.0024 | 0.0539 | n/a | n/a | 0.8359 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0539 | n/a | n/a | 0.8359 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0157 | 0.0153 | 0.0126 | 0.0181 | 0.0019 | 0.0539 | n/a | n/a | 0.8359 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0163 | 0.0169 | 0.0162 | 0.0182 | 0.0008 | 0.0849 | n/a | n/a | 0.9020 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0014 | 0.0014 | 0.0010 | 0.0019 | 0.0003 | 0.0849 | n/a | n/a | 0.9020 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0147 | 0.0145 | 0.0122 | 0.0168 | 0.0015 | 0.0849 | n/a | n/a | 0.9020 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0170 | 0.0183 | 0.0160 | 0.0209 | 0.0021 | 0.1602 | n/a | n/a | 0.7767 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0024 | 0.0029 | 0.0002 | 0.1602 | n/a | n/a | 0.7767 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0132 | 0.0130 | 0.0101 | 0.0148 | 0.0017 | 0.1602 | n/a | n/a | 0.7767 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0928 | 0.0944 | 0.0910 | 0.1022 | 0.0040 | 0.0853 | n/a | n/a | 1.5280 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0079 | 0.0079 | 0.0072 | 0.0085 | 0.0005 | 0.0853 | n/a | n/a | 1.5280 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1419 | 0.1398 | 0.1265 | 0.1520 | 0.0094 | 0.0853 | n/a | n/a | 1.5280 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0202 | 0.0202 | 0.0178 | 0.0228 | 0.0018 | 0.0698 | n/a | n/a | 0.7394 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.0698 | n/a | n/a | 0.7394 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0149 | 0.0153 | 0.0136 | 0.0177 | 0.0014 | 0.0698 | n/a | n/a | 0.7394 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0180 | 0.0184 | 0.0175 | 0.0206 | 0.0011 | 0.0797 | n/a | n/a | 0.8183 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0019 | 0.0002 | 0.0797 | n/a | n/a | 0.8183 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0147 | 0.0160 | 0.0127 | 0.0217 | 0.0034 | 0.0797 | n/a | n/a | 0.8183 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0188 | 0.0187 | 0.0174 | 0.0195 | 0.0007 | 0.0785 | n/a | n/a | 0.7264 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0018 | 0.0014 | 0.0026 | 0.0004 | 0.0785 | n/a | n/a | 0.7264 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0137 | 0.0140 | 0.0131 | 0.0155 | 0.0009 | 0.0785 | n/a | n/a | 0.7264 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0231 | 0.0218 | 0.0170 | 0.0260 | 0.0033 | 0.0754 | n/a | n/a | 0.5194 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0018 | 0.0001 | 0.0754 | n/a | n/a | 0.5194 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0120 | 0.0123 | 0.0115 | 0.0137 | 0.0008 | 0.0754 | n/a | n/a | 0.5194 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0948 | 0.0952 | 0.0925 | 0.0995 | 0.0023 | 0.1494 | n/a | n/a | 2.1269 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0142 | 0.0142 | 0.0097 | 0.0183 | 0.0036 | 0.1494 | n/a | n/a | 2.1269 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.2017 | 0.1921 | 0.1434 | 0.2530 | 0.0417 | 0.1494 | n/a | n/a | 2.1269 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0256 | 0.0297 | 0.0242 | 0.0384 | 0.0060 | 0.2015 | n/a | n/a | 0.9505 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0052 | 0.0055 | 0.0044 | 0.0079 | 0.0013 | 0.2015 | n/a | n/a | 0.9505 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0243 | 0.0277 | 0.0240 | 0.0362 | 0.0048 | 0.2015 | n/a | n/a | 0.9505 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0394 | 0.0402 | 0.0371 | 0.0438 | 0.0028 | 0.1477 | n/a | n/a | 0.6054 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0058 | 0.0061 | 0.0045 | 0.0080 | 0.0014 | 0.1477 | n/a | n/a | 0.6054 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0239 | 0.0276 | 0.0141 | 0.0501 | 0.0130 | 0.1477 | n/a | n/a | 0.6054 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0287 | 0.0304 | 0.0255 | 0.0359 | 0.0043 | 0.1550 | n/a | n/a | 0.5821 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0044 | 0.0047 | 0.0044 | 0.0055 | 0.0004 | 0.1550 | n/a | n/a | 0.5821 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0167 | 0.0177 | 0.0146 | 0.0247 | 0.0036 | 0.1550 | n/a | n/a | 0.5821 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0225 | 0.0250 | 0.0218 | 0.0337 | 0.0045 | 0.2627 | n/a | n/a | 0.6500 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0059 | 0.0061 | 0.0049 | 0.0088 | 0.0014 | 0.2627 | n/a | n/a | 0.6500 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0146 | 0.0168 | 0.0136 | 0.0259 | 0.0046 | 0.2627 | n/a | n/a | 0.6500 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1215 | 0.1310 | 0.1155 | 0.1600 | 0.0174 | 0.1955 | n/a | n/a | 1.1058 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0237 | 0.0246 | 0.0224 | 0.0300 | 0.0028 | 0.1955 | n/a | n/a | 1.1058 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1343 | 0.1427 | 0.1301 | 0.1689 | 0.0142 | 0.1955 | n/a | n/a | 1.1058 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0172 | 0.0190 | 0.0169 | 0.0232 | 0.0025 | 0.1516 | n/a | n/a | 3.2646 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0026 | 0.0025 | 0.0018 | 0.0033 | 0.0006 | 0.1516 | n/a | n/a | 3.2646 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0561 | 0.0555 | 0.0475 | 0.0615 | 0.0047 | 0.1516 | n/a | n/a | 3.2646 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0184 | 0.0187 | 0.0157 | 0.0222 | 0.0023 | 0.2609 | n/a | n/a | 4.2102 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0048 | 0.0049 | 0.0043 | 0.0060 | 0.0006 | 0.2609 | n/a | n/a | 4.2102 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0776 | 0.0767 | 0.0727 | 0.0801 | 0.0028 | 0.2609 | n/a | n/a | 4.2102 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0164 | 0.0170 | 0.0161 | 0.0185 | 0.0010 | 0.0735 | n/a | n/a | 1.8388 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0002 | 0.0735 | n/a | n/a | 1.8388 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0302 | 0.0314 | 0.0298 | 0.0341 | 0.0018 | 0.0735 | n/a | n/a | 1.8388 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0162 | 0.0166 | 0.0155 | 0.0183 | 0.0010 | 0.0763 | n/a | n/a | 2.3853 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0014 | 0.0012 | 0.0017 | 0.0002 | 0.0763 | n/a | n/a | 2.3853 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0387 | 0.0369 | 0.0316 | 0.0418 | 0.0040 | 0.0763 | n/a | n/a | 2.3853 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0164 | 0.0167 | 0.0160 | 0.0180 | 0.0007 | 0.0866 | n/a | n/a | 2.0398 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0015 | 0.0013 | 0.0021 | 0.0003 | 0.0866 | n/a | n/a | 2.0398 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0334 | 0.0346 | 0.0322 | 0.0399 | 0.0029 | 0.0866 | n/a | n/a | 2.0398 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0162 | 0.0168 | 0.0161 | 0.0190 | 0.0011 | 0.3417 | n/a | n/a | 2.9348 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0055 | 0.0060 | 0.0026 | 0.0112 | 0.0029 | 0.3417 | n/a | n/a | 2.9348 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0475 | 0.0496 | 0.0465 | 0.0573 | 0.0040 | 0.3417 | n/a | n/a | 2.9348 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0159 | 0.0162 | 0.0158 | 0.0172 | 0.0005 | 0.2939 | n/a | n/a | 4.6695 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0047 | 0.0046 | 0.0045 | 0.0047 | 0.0001 | 0.2939 | n/a | n/a | 4.6695 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0742 | 0.0770 | 0.0731 | 0.0827 | 0.0040 | 0.2939 | n/a | n/a | 4.6695 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0164 | 0.0169 | 0.0162 | 0.0185 | 0.0009 | 0.0796 | n/a | n/a | 1.8723 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0017 | 0.0002 | 0.0796 | n/a | n/a | 1.8723 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0306 | 0.0303 | 0.0293 | 0.0308 | 0.0006 | 0.0796 | n/a | n/a | 1.8723 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0153 | 0.0155 | 0.0151 | 0.0165 | 0.0005 | 0.0841 | n/a | n/a | 1.9227 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0001 | 0.0841 | n/a | n/a | 1.9227 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0294 | 0.0298 | 0.0289 | 0.0320 | 0.0011 | 0.0841 | n/a | n/a | 1.9227 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0154 | 0.0155 | 0.0151 | 0.0159 | 0.0003 | 0.0819 | n/a | n/a | 1.9078 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0819 | n/a | n/a | 1.9078 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0294 | 0.0297 | 0.0293 | 0.0310 | 0.0007 | 0.0819 | n/a | n/a | 1.9078 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0150 | 0.0150 | 0.0148 | 0.0152 | 0.0001 | 0.1245 | n/a | n/a | 2.9814 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0019 | 0.0019 | 0.0017 | 0.0020 | 0.0001 | 0.1245 | n/a | n/a | 2.9814 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0448 | 0.0478 | 0.0434 | 0.0552 | 0.0047 | 0.1245 | n/a | n/a | 2.9814 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0173 | 0.0169 | 0.0157 | 0.0178 | 0.0008 | 0.2724 | n/a | n/a | 4.1608 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0047 | 0.0047 | 0.0043 | 0.0050 | 0.0002 | 0.2724 | n/a | n/a | 4.1608 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0719 | 0.0724 | 0.0699 | 0.0750 | 0.0021 | 0.2724 | n/a | n/a | 4.1608 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0154 | 0.0154 | 0.0153 | 0.0157 | 0.0002 | 0.0828 | n/a | n/a | 1.8979 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0017 | 0.0002 | 0.0828 | n/a | n/a | 1.8979 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0292 | 0.0295 | 0.0290 | 0.0306 | 0.0006 | 0.0828 | n/a | n/a | 1.8979 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0152 | 0.0153 | 0.0152 | 0.0154 | 0.0001 | 0.0863 | n/a | n/a | 2.0475 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.0863 | n/a | n/a | 2.0475 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0311 | 0.0325 | 0.0297 | 0.0371 | 0.0028 | 0.0863 | n/a | n/a | 2.0475 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0154 | 0.0156 | 0.0153 | 0.0166 | 0.0005 | 0.0854 | n/a | n/a | 2.0147 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 0.0854 | n/a | n/a | 2.0147 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0310 | 0.0319 | 0.0303 | 0.0346 | 0.0016 | 0.0854 | n/a | n/a | 2.0147 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0152 | 0.0152 | 0.0150 | 0.0154 | 0.0001 | 0.1231 | n/a | n/a | 2.9110 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0019 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.1231 | n/a | n/a | 2.9110 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0442 | 0.0449 | 0.0440 | 0.0479 | 0.0015 | 0.1231 | n/a | n/a | 2.9110 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0161 | 0.0001 | 0.2850 | n/a | n/a | 4.8947 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0045 | 0.0045 | 0.0044 | 0.0047 | 0.0001 | 0.2850 | n/a | n/a | 4.8947 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0775 | 0.0763 | 0.0724 | 0.0793 | 0.0028 | 0.2850 | n/a | n/a | 4.8947 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0170 | 0.0172 | 0.0166 | 0.0179 | 0.0005 | 0.1063 | n/a | n/a | 1.8023 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0018 | 0.0017 | 0.0015 | 0.0020 | 0.0002 | 0.1063 | n/a | n/a | 1.8023 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0307 | 0.0334 | 0.0292 | 0.0395 | 0.0044 | 0.1063 | n/a | n/a | 1.8023 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0159 | 0.0160 | 0.0158 | 0.0167 | 0.0004 | 0.0850 | n/a | n/a | 1.8822 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0018 | 0.0002 | 0.0850 | n/a | n/a | 1.8822 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0299 | 0.0335 | 0.0293 | 0.0480 | 0.0073 | 0.0850 | n/a | n/a | 1.8822 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0160 | 0.0160 | 0.0158 | 0.0161 | 0.0001 | 0.0906 | n/a | n/a | 1.8178 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | 0.0906 | n/a | n/a | 1.8178 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0291 | 0.0292 | 0.0289 | 0.0296 | 0.0003 | 0.0906 | n/a | n/a | 1.8178 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0166 | 0.0160 | 0.0149 | 0.0167 | 0.0008 | 0.1405 | n/a | n/a | 3.5991 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.1405 | n/a | n/a | 3.5991 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0598 | 0.0643 | 0.0574 | 0.0820 | 0.0091 | 0.1405 | n/a | n/a | 3.5991 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0198 | 0.0201 | 0.0192 | 0.0208 | 0.0007 | 0.3597 | n/a | n/a | 4.7647 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0071 | 0.0071 | 0.0066 | 0.0078 | 0.0004 | 0.3597 | n/a | n/a | 4.7647 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0944 | 0.0940 | 0.0883 | 0.1013 | 0.0043 | 0.3597 | n/a | n/a | 4.7647 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0260 | 0.0260 | 0.0258 | 0.0264 | 0.0002 | 0.1460 | n/a | n/a | 1.1312 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0041 | 0.0001 | 0.1460 | n/a | n/a | 1.1312 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0294 | 0.0297 | 0.0291 | 0.0307 | 0.0006 | 0.1460 | n/a | n/a | 1.1312 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0191 | 0.0191 | 0.0189 | 0.0194 | 0.0002 | 0.1344 | n/a | n/a | 1.5697 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0025 | 0.0026 | 0.0001 | 0.1344 | n/a | n/a | 1.5697 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0300 | 0.0301 | 0.0294 | 0.0309 | 0.0005 | 0.1344 | n/a | n/a | 1.5697 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0210 | 0.0208 | 0.0198 | 0.0221 | 0.0008 | 0.1598 | n/a | n/a | 1.4889 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0034 | 0.0032 | 0.0029 | 0.0036 | 0.0003 | 0.1598 | n/a | n/a | 1.4889 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0312 | 0.0311 | 0.0300 | 0.0324 | 0.0008 | 0.1598 | n/a | n/a | 1.4889 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0164 | 0.0168 | 0.0163 | 0.0175 | 0.0006 | 0.2333 | n/a | n/a | 3.8755 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0038 | 0.0038 | 0.0036 | 0.0040 | 0.0002 | 0.2333 | n/a | n/a | 3.8755 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0635 | 0.0652 | 0.0580 | 0.0734 | 0.0053 | 0.2333 | n/a | n/a | 3.8755 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0341 | 0.0342 | 0.0337 | 0.0347 | 0.0003 | 0.3515 | n/a | n/a | 3.5371 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0120 | 0.0122 | 0.0114 | 0.0134 | 0.0007 | 0.3515 | n/a | n/a | 3.5371 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1207 | 0.1277 | 0.1024 | 0.1560 | 0.0225 | 0.3515 | n/a | n/a | 3.5371 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0858 | 0.0877 | 0.0794 | 0.0947 | 0.0055 | 0.1381 | n/a | n/a | 0.5187 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0118 | 0.0127 | 0.0115 | 0.0146 | 0.0014 | 0.1381 | n/a | n/a | 0.5187 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0445 | 0.0459 | 0.0349 | 0.0562 | 0.0075 | 0.1381 | n/a | n/a | 0.5187 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0340 | 0.0354 | 0.0331 | 0.0392 | 0.0023 | 0.1991 | n/a | n/a | 1.0945 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0068 | 0.0069 | 0.0065 | 0.0072 | 0.0003 | 0.1991 | n/a | n/a | 1.0945 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0372 | 0.0390 | 0.0333 | 0.0514 | 0.0066 | 0.1991 | n/a | n/a | 1.0945 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0381 | 0.0398 | 0.0377 | 0.0435 | 0.0023 | 0.1951 | n/a | n/a | 0.9137 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0074 | 0.0076 | 0.0071 | 0.0083 | 0.0004 | 0.1951 | n/a | n/a | 0.9137 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0348 | 0.0344 | 0.0337 | 0.0350 | 0.0005 | 0.1951 | n/a | n/a | 0.9137 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0228 | 0.0243 | 0.0215 | 0.0281 | 0.0027 | 0.4113 | n/a | n/a | 4.2580 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0094 | 0.0105 | 0.0087 | 0.0162 | 0.0029 | 0.4113 | n/a | n/a | 4.2580 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0969 | 0.0993 | 0.0824 | 0.1263 | 0.0162 | 0.4113 | n/a | n/a | 4.2580 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0947 | 0.0930 | 0.0862 | 0.0985 | 0.0045 | 0.3401 | n/a | n/a | 1.1904 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0322 | 0.0343 | 0.0321 | 0.0396 | 0.0030 | 0.3401 | n/a | n/a | 1.1904 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1127 | 0.1138 | 0.1076 | 0.1244 | 0.0058 | 0.3401 | n/a | n/a | 1.1904 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2216 | 0.2290 | 0.2159 | 0.2607 | 0.0166 | 0.1782 | n/a | n/a | 0.2183 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0395 | 0.0399 | 0.0375 | 0.0431 | 0.0022 | 0.1782 | n/a | n/a | 0.2183 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0484 | 0.0502 | 0.0386 | 0.0667 | 0.0109 | 0.1782 | n/a | n/a | 0.2183 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0757 | 0.0787 | 0.0756 | 0.0840 | 0.0038 | 0.2604 | n/a | n/a | 0.5281 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0197 | 0.0198 | 0.0189 | 0.0207 | 0.0006 | 0.2604 | n/a | n/a | 0.5281 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0400 | 0.0399 | 0.0387 | 0.0412 | 0.0010 | 0.2604 | n/a | n/a | 0.5281 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1016 | 0.1027 | 0.1013 | 0.1055 | 0.0016 | 0.2556 | n/a | n/a | 0.4148 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0260 | 0.0258 | 0.0242 | 0.0279 | 0.0013 | 0.2556 | n/a | n/a | 0.4148 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0421 | 0.0448 | 0.0413 | 0.0520 | 0.0042 | 0.2556 | n/a | n/a | 0.4148 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0156 | 0.0155 | 0.0154 | 0.0157 | 0.0001 | 0.1391 | n/a | n/a | 0.7569 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0022 | 0.0023 | 0.0021 | 0.0028 | 0.0003 | 0.1391 | n/a | n/a | 0.7569 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0118 | 0.0119 | 0.0116 | 0.0128 | 0.0004 | 0.1391 | n/a | n/a | 0.7569 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0155 | 0.0156 | 0.0155 | 0.0157 | 0.0001 | 0.5248 | n/a | n/a | 0.7261 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0082 | 0.0082 | 0.0080 | 0.0083 | 0.0001 | 0.5248 | n/a | n/a | 0.7261 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0113 | 0.0114 | 0.0108 | 0.0119 | 0.0004 | 0.5248 | n/a | n/a | 0.7261 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0156 | 0.0157 | 0.0154 | 0.0162 | 0.0003 | 0.1465 | n/a | n/a | 0.7913 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0023 | 0.0023 | 0.0021 | 0.0026 | 0.0002 | 0.1465 | n/a | n/a | 0.7913 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0123 | 0.0124 | 0.0121 | 0.0131 | 0.0004 | 0.1465 | n/a | n/a | 0.7913 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0161 | 0.0162 | 0.0159 | 0.0166 | 0.0003 | 0.5112 | n/a | n/a | 0.7503 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0082 | 0.0081 | 0.0078 | 0.0083 | 0.0002 | 0.5112 | n/a | n/a | 0.7503 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0121 | 0.0123 | 0.0116 | 0.0135 | 0.0007 | 0.5112 | n/a | n/a | 0.7503 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0156 | 0.0157 | 0.0155 | 0.0161 | 0.0002 | 0.1384 | n/a | n/a | 0.7993 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 0.1384 | n/a | n/a | 0.7993 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0125 | 0.0126 | 0.0121 | 0.0137 | 0.0006 | 0.1384 | n/a | n/a | 0.7993 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0157 | 0.0158 | 0.0155 | 0.0164 | 0.0003 | 0.5208 | n/a | n/a | 0.7587 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0082 | 0.0082 | 0.0082 | 0.0085 | 0.0001 | 0.5208 | n/a | n/a | 0.7587 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0119 | 0.0121 | 0.0113 | 0.0129 | 0.0005 | 0.5208 | n/a | n/a | 0.7587 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0157 | 0.0157 | 0.0156 | 0.0159 | 0.0001 | 0.1390 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0024 | 0.0001 | 0.1390 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0133 | 0.0132 | 0.0126 | 0.0139 | 0.0004 | 0.1390 | n/a | n/a | 0.8517 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0162 | 0.0176 | 0.0159 | 0.0201 | 0.0019 | 0.5132 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0083 | 0.0083 | 0.0080 | 0.0086 | 0.0002 | 0.5132 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0144 | 0.0143 | 0.0128 | 0.0158 | 0.0011 | 0.5132 | n/a | n/a | 0.8869 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0167 | 0.0167 | 0.0166 | 0.0168 | 0.0000 | 0.1567 | n/a | n/a | 0.9619 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0026 | 0.0027 | 0.0026 | 0.0032 | 0.0002 | 0.1567 | n/a | n/a | 0.9619 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0161 | 0.0161 | 0.0138 | 0.0177 | 0.0013 | 0.1567 | n/a | n/a | 0.9619 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0170 | 0.0168 | 0.0161 | 0.0175 | 0.0006 | 0.5293 | n/a | n/a | 0.9530 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0090 | 0.0097 | 0.0083 | 0.0117 | 0.0012 | 0.5293 | n/a | n/a | 0.9530 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0162 | 0.0168 | 0.0139 | 0.0195 | 0.0021 | 0.5293 | n/a | n/a | 0.9530 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0201 | 0.0201 | 0.0195 | 0.0207 | 0.0004 | 0.1782 | n/a | n/a | 0.7486 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0036 | 0.0037 | 0.0034 | 0.0041 | 0.0002 | 0.1782 | n/a | n/a | 0.7486 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0150 | 0.0161 | 0.0143 | 0.0202 | 0.0021 | 0.1782 | n/a | n/a | 0.7486 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0202 | 0.0203 | 0.0197 | 0.0211 | 0.0005 | 0.4766 | n/a | n/a | 0.7550 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0096 | 0.0101 | 0.0092 | 0.0124 | 0.0012 | 0.4766 | n/a | n/a | 0.7550 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0152 | 0.0154 | 0.0140 | 0.0178 | 0.0014 | 0.4766 | n/a | n/a | 0.7550 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0330 | 0.0331 | 0.0322 | 0.0341 | 0.0006 | 0.2243 | n/a | n/a | 0.8992 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0074 | 0.0078 | 0.0070 | 0.0095 | 0.0009 | 0.2243 | n/a | n/a | 0.8992 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0297 | 0.0262 | 0.0198 | 0.0317 | 0.0052 | 0.2243 | n/a | n/a | 0.8992 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0338 | 0.0335 | 0.0323 | 0.0351 | 0.0011 | 0.3827 | n/a | n/a | 0.8555 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0129 | 0.0132 | 0.0129 | 0.0145 | 0.0006 | 0.3827 | n/a | n/a | 0.8555 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0289 | 0.0268 | 0.0187 | 0.0337 | 0.0057 | 0.3827 | n/a | n/a | 0.8555 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0159 | 0.0164 | 0.0156 | 0.0177 | 0.0008 | 0.1924 | n/a | n/a | 0.8231 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0031 | 0.0031 | 0.0026 | 0.0042 | 0.0006 | 0.1924 | n/a | n/a | 0.8231 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0131 | 0.0134 | 0.0127 | 0.0149 | 0.0008 | 0.1924 | n/a | n/a | 0.8231 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0105 | 0.0107 | 0.0102 | 0.0113 | 0.0005 | 1.0622 | n/a | n/a | 1.1111 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0112 | 0.0110 | 0.0097 | 0.0131 | 0.0012 | 1.0622 | n/a | n/a | 1.1111 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0117 | 0.0119 | 0.0114 | 0.0132 | 0.0006 | 1.0622 | n/a | n/a | 1.1111 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0188 | 0.0194 | 0.0186 | 0.0211 | 0.0009 | 0.2189 | n/a | n/a | 0.8419 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0041 | 0.0042 | 0.0040 | 0.0047 | 0.0003 | 0.2189 | n/a | n/a | 0.8419 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0159 | 0.0164 | 0.0137 | 0.0211 | 0.0025 | 0.2189 | n/a | n/a | 0.8419 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0134 | 0.0136 | 0.0131 | 0.0145 | 0.0005 | 0.8962 | n/a | n/a | 1.4997 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0121 | 0.0121 | 0.0109 | 0.0132 | 0.0008 | 0.8962 | n/a | n/a | 1.4997 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0202 | 0.0193 | 0.0152 | 0.0228 | 0.0026 | 0.8962 | n/a | n/a | 1.4997 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0455 | 0.0427 | 0.0309 | 0.0491 | 0.0064 | 0.2234 | n/a | n/a | 0.7323 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0102 | 0.0101 | 0.0087 | 0.0119 | 0.0011 | 0.2234 | n/a | n/a | 0.7323 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0333 | 0.0356 | 0.0324 | 0.0433 | 0.0040 | 0.2234 | n/a | n/a | 0.7323 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0287 | 0.0286 | 0.0260 | 0.0317 | 0.0022 | 0.5716 | n/a | n/a | 0.4914 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0164 | 0.0165 | 0.0160 | 0.0173 | 0.0005 | 0.5716 | n/a | n/a | 0.4914 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0141 | 0.0145 | 0.0140 | 0.0154 | 0.0006 | 0.5716 | n/a | n/a | 0.4914 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0990 | 0.0995 | 0.0895 | 0.1164 | 0.0094 | 0.2293 | n/a | n/a | 0.6972 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0227 | 0.0228 | 0.0215 | 0.0248 | 0.0011 | 0.2293 | n/a | n/a | 0.6972 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0690 | 0.0716 | 0.0646 | 0.0806 | 0.0064 | 0.2293 | n/a | n/a | 0.6972 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0725 | 0.0707 | 0.0591 | 0.0883 | 0.0105 | 0.3323 | n/a | n/a | 0.2298 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0241 | 0.0246 | 0.0231 | 0.0276 | 0.0016 | 0.3323 | n/a | n/a | 0.2298 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0167 | 0.0230 | 0.0162 | 0.0403 | 0.0093 | 0.3323 | n/a | n/a | 0.2298 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0209 | 0.0221 | 0.0192 | 0.0254 | 0.0025 | 0.0985 | n/a | n/a | 0.4507 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0021 | 0.0021 | 0.0020 | 0.0022 | 0.0001 | 0.0985 | n/a | n/a | 0.4507 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0094 | 0.0096 | 0.0093 | 0.0104 | 0.0004 | 0.0985 | n/a | n/a | 0.4507 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1017 | 0.1053 | 0.0960 | 0.1245 | 0.0101 | 0.1025 | n/a | n/a | 0.1767 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0104 | 0.0104 | 0.0104 | 0.0105 | 0.0000 | 0.1025 | n/a | n/a | 0.1767 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0180 | 0.0277 | 0.0179 | 0.0479 | 0.0125 | 0.1025 | n/a | n/a | 0.1767 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3994 | 0.4012 | 0.3765 | 0.4400 | 0.0219 | 0.7132 | n/a | n/a | 0.3841 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2848 | 0.2955 | 0.2581 | 0.3403 | 0.0340 | 0.7132 | n/a | n/a | 0.3841 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1534 | 0.1497 | 0.1231 | 0.1781 | 0.0201 | 0.7132 | n/a | n/a | 0.3841 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.5261 | 2.7506 | 2.0672 | 3.7313 | 0.5596 | 0.1524 | n/a | n/a | 0.0906 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.3848 | 0.3921 | 0.3396 | 0.4467 | 0.0391 | 0.1524 | n/a | n/a | 0.0906 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2288 | 0.2249 | 0.2183 | 0.2291 | 0.0049 | 0.1524 | n/a | n/a | 0.0906 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2120 | 0.2096 | 0.1995 | 0.2200 | 0.0070 | 6.1191 | n/a | n/a | 0.4524 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2975 | 1.5899 | 1.2614 | 2.2292 | 0.3957 | 6.1191 | n/a | n/a | 0.4524 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.0959 | 0.1045 | 0.0940 | 0.1229 | 0.0119 | 6.1191 | n/a | n/a | 0.4524 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 6.0889 | 6.0885 | 6.0309 | 6.1400 | 0.0348 | 10.2894 | n/a | n/a | 0.0559 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 62.6518 | 62.9329 | 61.8360 | 64.9169 | 1.0977 | 10.2894 | n/a | n/a | 0.0559 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3402 | 0.3575 | 0.3281 | 0.3964 | 0.0301 | 10.2894 | n/a | n/a | 0.0559 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0286 | 0.0296 | 0.0271 | 0.0339 | 0.0023 | 5.6321 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1613 | 0.1635 | 0.1576 | 0.1733 | 0.0057 | 5.6321 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0304 | 0.0311 | 0.0274 | 0.0373 | 0.0035 | 18.4457 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5601 | 0.5591 | 0.5484 | 0.5670 | 0.0062 | 18.4457 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6229 | 0.6286 | 0.6138 | 0.6478 | 0.0119 | 12.2138 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.6075 | 7.8191 | 7.4806 | 8.2552 | 0.3387 | 12.2138 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6881 | 0.6893 | 0.6526 | 0.7376 | 0.0337 | 39.9640 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 27.4983 | 28.4737 | 27.1957 | 30.9504 | 1.5009 | 39.9640 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1201 | 0.1223 | 0.1191 | 0.1307 | 0.0043 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2984 | 0.2983 | 0.2961 | 0.3000 | 0.0016 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2481 | 0.2469 | 0.2444 | 0.2492 | 0.0020 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.6288 | 0.6371 | 0.5967 | 0.7282 | 0.0480 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1373 | 1.1446 | 1.1229 | 1.1676 | 0.0164 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7150 | 2.7640 | 2.7034 | 2.9651 | 0.1008 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.1632 | 5.2214 | 5.0996 | 5.3585 | 0.1044 | 0.0298 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1540 | 0.1618 | 0.1527 | 0.1899 | 0.0143 | 0.0298 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
