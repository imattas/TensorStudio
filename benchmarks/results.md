# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.13.0`
- TensorStudio threads: `12`
- TensorStudio BLAS enabled: `False`
- TensorStudio SIMD level: `sse2`
- TensorStudio storage pool enabled: `True`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `8`
- TensorStudio losses versus NumPy: `95`
- TensorStudio wins versus JAX CPU dispatch: `51`
- TensorStudio losses versus JAX CPU dispatch: `47`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0196 | 0.0207 | 0.0160 | 0.0270 | 0.0042 | 0.0330 | n/a | n/a | 0.6685 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0330 | n/a | n/a | 0.6685 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0131 | 0.0143 | 0.0116 | 0.0193 | 0.0028 | 0.0330 | n/a | n/a | 0.6685 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0168 | 0.0167 | 0.0161 | 0.0172 | 0.0004 | 0.0413 | n/a | n/a | 0.8567 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0413 | n/a | n/a | 0.8567 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0144 | 0.0154 | 0.0118 | 0.0217 | 0.0035 | 0.0413 | n/a | n/a | 0.8567 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0200 | 0.0205 | 0.0177 | 0.0240 | 0.0021 | 0.0337 | n/a | n/a | 0.6271 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0337 | n/a | n/a | 0.6271 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0125 | 0.0138 | 0.0124 | 0.0159 | 0.0016 | 0.0337 | n/a | n/a | 0.6271 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0171 | 0.0175 | 0.0163 | 0.0190 | 0.0010 | 0.0470 | n/a | n/a | 0.6725 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0009 | 0.0007 | 0.0012 | 0.0002 | 0.0470 | n/a | n/a | 0.6725 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0115 | 0.0116 | 0.0092 | 0.0140 | 0.0015 | 0.0470 | n/a | n/a | 0.6725 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0995 | 0.1103 | 0.0940 | 0.1533 | 0.0219 | 0.0664 | n/a | n/a | 1.0322 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0066 | 0.0077 | 0.0058 | 0.0116 | 0.0022 | 0.0664 | n/a | n/a | 1.0322 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.1027 | 0.1030 | 0.0929 | 0.1100 | 0.0061 | 0.0664 | n/a | n/a | 1.0322 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0163 | 0.0168 | 0.0160 | 0.0187 | 0.0010 | 0.0407 | n/a | n/a | 0.7641 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0407 | n/a | n/a | 0.7641 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0125 | 0.0124 | 0.0122 | 0.0126 | 0.0002 | 0.0407 | n/a | n/a | 0.7641 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0167 | 0.0170 | 0.0161 | 0.0179 | 0.0007 | 0.0421 | n/a | n/a | 0.9067 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0421 | n/a | n/a | 0.9067 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0152 | 0.0149 | 0.0127 | 0.0169 | 0.0015 | 0.0421 | n/a | n/a | 0.9067 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0168 | 0.0168 | 0.0162 | 0.0176 | 0.0006 | 0.0412 | n/a | n/a | 0.7788 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.0412 | n/a | n/a | 0.7788 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0131 | 0.0134 | 0.0128 | 0.0149 | 0.0008 | 0.0412 | n/a | n/a | 0.7788 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0176 | 0.0179 | 0.0167 | 0.0195 | 0.0009 | 0.0408 | n/a | n/a | 0.5798 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0408 | n/a | n/a | 0.5798 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0102 | 0.0102 | 0.0096 | 0.0107 | 0.0003 | 0.0408 | n/a | n/a | 0.5798 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0882 | 0.0896 | 0.0872 | 0.0925 | 0.0023 | 0.0748 | n/a | n/a | 1.1606 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0066 | 0.0066 | 0.0058 | 0.0074 | 0.0005 | 0.0748 | n/a | n/a | 1.1606 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1024 | 0.1026 | 0.1001 | 0.1050 | 0.0017 | 0.0748 | n/a | n/a | 1.1606 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0171 | 0.0177 | 0.0164 | 0.0202 | 0.0013 | 0.0374 | n/a | n/a | 0.8363 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0008 | 0.0001 | 0.0374 | n/a | n/a | 0.8363 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0143 | 0.0138 | 0.0126 | 0.0148 | 0.0010 | 0.0374 | n/a | n/a | 0.8363 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0176 | 0.0180 | 0.0172 | 0.0189 | 0.0007 | 0.0499 | n/a | n/a | 0.7518 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0009 | 0.0009 | 0.0007 | 0.0014 | 0.0003 | 0.0499 | n/a | n/a | 0.7518 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0132 | 0.0129 | 0.0119 | 0.0135 | 0.0006 | 0.0499 | n/a | n/a | 0.7518 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0163 | 0.0169 | 0.0161 | 0.0190 | 0.0011 | 0.0420 | n/a | n/a | 0.8113 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0420 | n/a | n/a | 0.8113 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0132 | 0.0136 | 0.0121 | 0.0162 | 0.0015 | 0.0420 | n/a | n/a | 0.8113 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0164 | 0.0164 | 0.0162 | 0.0169 | 0.0002 | 0.0437 | n/a | n/a | 0.6023 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0437 | n/a | n/a | 0.6023 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0099 | 0.0100 | 0.0094 | 0.0114 | 0.0007 | 0.0437 | n/a | n/a | 0.6023 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0871 | 0.0880 | 0.0858 | 0.0906 | 0.0019 | 0.0659 | n/a | n/a | 1.1431 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0057 | 0.0058 | 0.0056 | 0.0060 | 0.0002 | 0.0659 | n/a | n/a | 1.1431 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0996 | 0.1003 | 0.0983 | 0.1022 | 0.0016 | 0.0659 | n/a | n/a | 1.1431 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0170 | 0.0173 | 0.0165 | 0.0182 | 0.0007 | 0.0388 | n/a | n/a | 0.8228 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0388 | n/a | n/a | 0.8228 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0140 | 0.0146 | 0.0127 | 0.0180 | 0.0018 | 0.0388 | n/a | n/a | 0.8228 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0178 | 0.0197 | 0.0164 | 0.0261 | 0.0035 | 0.0833 | n/a | n/a | 0.7119 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0012 | 0.0017 | 0.0002 | 0.0833 | n/a | n/a | 0.7119 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0127 | 0.0127 | 0.0122 | 0.0135 | 0.0004 | 0.0833 | n/a | n/a | 0.7119 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0162 | 0.0162 | 0.0161 | 0.0164 | 0.0001 | 0.0434 | n/a | n/a | 0.7854 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0434 | n/a | n/a | 0.7854 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0127 | 0.0130 | 0.0123 | 0.0145 | 0.0008 | 0.0434 | n/a | n/a | 0.7854 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0179 | 0.0175 | 0.0162 | 0.0194 | 0.0012 | 0.0403 | n/a | n/a | 0.5248 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0403 | n/a | n/a | 0.5248 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0094 | 0.0098 | 0.0091 | 0.0114 | 0.0008 | 0.0403 | n/a | n/a | 0.5248 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0873 | 0.0877 | 0.0857 | 0.0906 | 0.0017 | 0.0663 | n/a | n/a | 1.1173 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0058 | 0.0057 | 0.0053 | 0.0059 | 0.0002 | 0.0663 | n/a | n/a | 1.1173 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0976 | 0.0982 | 0.0928 | 0.1077 | 0.0053 | 0.0663 | n/a | n/a | 1.1173 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0179 | 0.0188 | 0.0160 | 0.0232 | 0.0027 | 0.0559 | n/a | n/a | 1.0662 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.0559 | n/a | n/a | 1.0662 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0191 | 0.0195 | 0.0189 | 0.0211 | 0.0008 | 0.0559 | n/a | n/a | 1.0662 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0170 | 0.0171 | 0.0161 | 0.0181 | 0.0007 | 0.0619 | n/a | n/a | 1.0876 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0011 | 0.0012 | 0.0010 | 0.0018 | 0.0003 | 0.0619 | n/a | n/a | 1.0876 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0185 | 0.0172 | 0.0132 | 0.0202 | 0.0029 | 0.0619 | n/a | n/a | 1.0876 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0184 | 0.0176 | 0.0158 | 0.0189 | 0.0012 | 0.0548 | n/a | n/a | 1.0019 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.0548 | n/a | n/a | 1.0019 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0185 | 0.0185 | 0.0164 | 0.0203 | 0.0012 | 0.0548 | n/a | n/a | 1.0019 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0172 | 0.0176 | 0.0159 | 0.0205 | 0.0015 | 0.0622 | n/a | n/a | 0.8757 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.0622 | n/a | n/a | 0.8757 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0150 | 0.0139 | 0.0104 | 0.0155 | 0.0019 | 0.0622 | n/a | n/a | 0.8757 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0895 | 0.0916 | 0.0874 | 0.0998 | 0.0047 | 0.1172 | n/a | n/a | 1.8103 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0105 | 0.0114 | 0.0084 | 0.0145 | 0.0023 | 0.1172 | n/a | n/a | 1.8103 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1621 | 0.1627 | 0.1557 | 0.1718 | 0.0052 | 0.1172 | n/a | n/a | 1.8103 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0173 | 0.0176 | 0.0171 | 0.0182 | 0.0004 | 0.0975 | n/a | n/a | 1.1614 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.0975 | n/a | n/a | 1.1614 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0201 | 0.0199 | 0.0177 | 0.0215 | 0.0013 | 0.0975 | n/a | n/a | 1.1614 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0176 | 0.0179 | 0.0172 | 0.0185 | 0.0005 | 0.0862 | n/a | n/a | 1.1855 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0019 | 0.0002 | 0.0862 | n/a | n/a | 1.1855 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0208 | 0.0197 | 0.0155 | 0.0214 | 0.0022 | 0.0862 | n/a | n/a | 1.1855 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0178 | 0.0180 | 0.0166 | 0.0192 | 0.0010 | 0.0834 | n/a | n/a | 1.1281 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.0834 | n/a | n/a | 1.1281 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0201 | 0.0182 | 0.0140 | 0.0209 | 0.0027 | 0.0834 | n/a | n/a | 1.1281 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0183 | 0.0190 | 0.0174 | 0.0210 | 0.0013 | 0.0904 | n/a | n/a | 0.8839 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.0904 | n/a | n/a | 0.8839 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0162 | 0.0160 | 0.0148 | 0.0166 | 0.0006 | 0.0904 | n/a | n/a | 0.8839 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0936 | 0.0935 | 0.0923 | 0.0945 | 0.0009 | 0.1132 | n/a | n/a | 1.7883 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0106 | 0.0109 | 0.0099 | 0.0130 | 0.0011 | 0.1132 | n/a | n/a | 1.7883 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1673 | 0.1663 | 0.1631 | 0.1676 | 0.0017 | 0.1132 | n/a | n/a | 1.7883 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0207 | 0.0214 | 0.0205 | 0.0232 | 0.0010 | 0.2066 | n/a | n/a | 0.8969 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0043 | 0.0049 | 0.0035 | 0.0064 | 0.0013 | 0.2066 | n/a | n/a | 0.8969 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0186 | 0.0184 | 0.0132 | 0.0226 | 0.0036 | 0.2066 | n/a | n/a | 0.8969 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0208 | 0.0211 | 0.0206 | 0.0217 | 0.0005 | 0.2287 | n/a | n/a | 0.6293 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0048 | 0.0052 | 0.0045 | 0.0064 | 0.0008 | 0.2287 | n/a | n/a | 0.6293 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0131 | 0.0158 | 0.0128 | 0.0225 | 0.0038 | 0.2287 | n/a | n/a | 0.6293 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0208 | 0.0223 | 0.0205 | 0.0255 | 0.0020 | 0.1879 | n/a | n/a | 1.0906 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0039 | 0.0040 | 0.0038 | 0.0044 | 0.0002 | 0.1879 | n/a | n/a | 1.0906 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0227 | 0.0221 | 0.0195 | 0.0233 | 0.0013 | 0.1879 | n/a | n/a | 1.0906 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0209 | 0.0213 | 0.0206 | 0.0227 | 0.0008 | 0.2601 | n/a | n/a | 0.5765 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0054 | 0.0055 | 0.0043 | 0.0068 | 0.0011 | 0.2601 | n/a | n/a | 0.5765 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0120 | 0.0121 | 0.0117 | 0.0131 | 0.0005 | 0.2601 | n/a | n/a | 0.5765 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1050 | 0.1053 | 0.1041 | 0.1074 | 0.0012 | 0.2078 | n/a | n/a | 1.5932 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0218 | 0.0232 | 0.0212 | 0.0278 | 0.0024 | 0.2078 | n/a | n/a | 1.5932 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1672 | 0.1637 | 0.1393 | 0.1756 | 0.0128 | 0.2078 | n/a | n/a | 1.5932 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0153 | 0.0001 | 0.1134 | n/a | n/a | 3.2229 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0018 | 0.0016 | 0.0020 | 0.0001 | 0.1134 | n/a | n/a | 3.2229 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0484 | 0.0488 | 0.0474 | 0.0511 | 0.0013 | 0.1134 | n/a | n/a | 3.2229 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0151 | 0.0153 | 0.0150 | 0.0161 | 0.0004 | 0.3093 | n/a | n/a | 5.1637 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0047 | 0.0051 | 0.0045 | 0.0061 | 0.0006 | 0.3093 | n/a | n/a | 5.1637 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0781 | 0.0805 | 0.0764 | 0.0878 | 0.0042 | 0.3093 | n/a | n/a | 5.1637 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0161 | 0.0163 | 0.0150 | 0.0179 | 0.0011 | 0.0867 | n/a | n/a | 2.0628 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0014 | 0.0015 | 0.0012 | 0.0018 | 0.0002 | 0.0867 | n/a | n/a | 2.0628 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0333 | 0.0334 | 0.0312 | 0.0356 | 0.0014 | 0.0867 | n/a | n/a | 2.0628 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0152 | 0.0152 | 0.0149 | 0.0153 | 0.0001 | 0.0798 | n/a | n/a | 2.2402 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0014 | 0.0012 | 0.0018 | 0.0002 | 0.0798 | n/a | n/a | 2.2402 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0340 | 0.0338 | 0.0327 | 0.0353 | 0.0009 | 0.0798 | n/a | n/a | 2.2402 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0156 | 0.0157 | 0.0152 | 0.0166 | 0.0005 | 0.0881 | n/a | n/a | 2.1257 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0015 | 0.0013 | 0.0020 | 0.0003 | 0.0881 | n/a | n/a | 2.1257 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0331 | 0.0331 | 0.0319 | 0.0341 | 0.0008 | 0.0881 | n/a | n/a | 2.1257 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0156 | 0.0158 | 0.0152 | 0.0169 | 0.0007 | 0.1213 | n/a | n/a | 3.0564 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0019 | 0.0018 | 0.0016 | 0.0019 | 0.0001 | 0.1213 | n/a | n/a | 3.0564 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0476 | 0.0478 | 0.0462 | 0.0493 | 0.0010 | 0.1213 | n/a | n/a | 3.0564 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0158 | 0.0157 | 0.0150 | 0.0163 | 0.0005 | 0.2856 | n/a | n/a | 4.9356 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0045 | 0.0045 | 0.0042 | 0.0049 | 0.0003 | 0.2856 | n/a | n/a | 4.9356 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0779 | 0.0818 | 0.0774 | 0.0902 | 0.0053 | 0.2856 | n/a | n/a | 4.9356 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0166 | 0.0163 | 0.0153 | 0.0173 | 0.0009 | 0.0939 | n/a | n/a | 1.9879 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0016 | 0.0017 | 0.0013 | 0.0026 | 0.0005 | 0.0939 | n/a | n/a | 1.9879 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0329 | 0.0326 | 0.0304 | 0.0344 | 0.0016 | 0.0939 | n/a | n/a | 1.9879 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0166 | 0.0167 | 0.0154 | 0.0179 | 0.0009 | 0.0919 | n/a | n/a | 2.0011 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0015 | 0.0015 | 0.0012 | 0.0017 | 0.0002 | 0.0919 | n/a | n/a | 2.0011 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0332 | 0.0333 | 0.0319 | 0.0347 | 0.0010 | 0.0919 | n/a | n/a | 2.0011 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0166 | 0.0166 | 0.0151 | 0.0176 | 0.0009 | 0.0759 | n/a | n/a | 2.0185 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0016 | 0.0002 | 0.0759 | n/a | n/a | 2.0185 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0335 | 0.0334 | 0.0319 | 0.0345 | 0.0008 | 0.0759 | n/a | n/a | 2.0185 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0153 | 0.0155 | 0.0151 | 0.0167 | 0.0006 | 0.1215 | n/a | n/a | 3.2783 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0019 | 0.0020 | 0.0016 | 0.0024 | 0.0003 | 0.1215 | n/a | n/a | 3.2783 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0501 | 0.0509 | 0.0489 | 0.0535 | 0.0018 | 0.1215 | n/a | n/a | 3.2783 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0156 | 0.0159 | 0.0151 | 0.0178 | 0.0009 | 0.3067 | n/a | n/a | 5.1476 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0048 | 0.0048 | 0.0045 | 0.0051 | 0.0003 | 0.3067 | n/a | n/a | 5.1476 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0803 | 0.0800 | 0.0780 | 0.0814 | 0.0012 | 0.3067 | n/a | n/a | 5.1476 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0208 | 0.0209 | 0.0172 | 0.0255 | 0.0028 | 0.0924 | n/a | n/a | 1.6535 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0019 | 0.0017 | 0.0013 | 0.0022 | 0.0004 | 0.0924 | n/a | n/a | 1.6535 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0343 | 0.0355 | 0.0330 | 0.0406 | 0.0027 | 0.0924 | n/a | n/a | 1.6535 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0158 | 0.0168 | 0.0156 | 0.0194 | 0.0015 | 0.1141 | n/a | n/a | 2.1895 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0018 | 0.0020 | 0.0015 | 0.0027 | 0.0004 | 0.1141 | n/a | n/a | 2.1895 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0346 | 0.0348 | 0.0314 | 0.0380 | 0.0023 | 0.1141 | n/a | n/a | 2.1895 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0171 | 0.0172 | 0.0158 | 0.0187 | 0.0011 | 0.0789 | n/a | n/a | 2.2384 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0016 | 0.0013 | 0.0025 | 0.0005 | 0.0789 | n/a | n/a | 2.2384 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0382 | 0.0375 | 0.0319 | 0.0436 | 0.0039 | 0.0789 | n/a | n/a | 2.2384 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0165 | 0.0166 | 0.0156 | 0.0182 | 0.0009 | 0.1253 | n/a | n/a | 3.5876 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0021 | 0.0023 | 0.0017 | 0.0028 | 0.0004 | 0.1253 | n/a | n/a | 3.5876 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0593 | 0.0598 | 0.0552 | 0.0653 | 0.0042 | 0.1253 | n/a | n/a | 3.5876 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0163 | 0.0163 | 0.0158 | 0.0174 | 0.0006 | 0.3022 | n/a | n/a | 6.0106 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0049 | 0.0053 | 0.0043 | 0.0072 | 0.0010 | 0.3022 | n/a | n/a | 6.0106 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0979 | 0.1002 | 0.0928 | 0.1131 | 0.0070 | 0.3022 | n/a | n/a | 6.0106 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0205 | 0.0200 | 0.0179 | 0.0223 | 0.0017 | 0.0804 | n/a | n/a | 2.3011 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0016 | 0.0018 | 0.0015 | 0.0026 | 0.0004 | 0.0804 | n/a | n/a | 2.3011 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0472 | 0.0446 | 0.0347 | 0.0509 | 0.0057 | 0.0804 | n/a | n/a | 2.3011 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0179 | 0.0176 | 0.0164 | 0.0188 | 0.0009 | 0.1146 | n/a | n/a | 2.0471 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0021 | 0.0021 | 0.0016 | 0.0027 | 0.0004 | 0.1146 | n/a | n/a | 2.0471 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0367 | 0.0365 | 0.0346 | 0.0386 | 0.0015 | 0.1146 | n/a | n/a | 2.0471 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0180 | 0.0178 | 0.0170 | 0.0186 | 0.0006 | 0.0996 | n/a | n/a | 2.1612 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0018 | 0.0017 | 0.0014 | 0.0019 | 0.0002 | 0.0996 | n/a | n/a | 2.1612 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0389 | 0.0415 | 0.0373 | 0.0524 | 0.0056 | 0.0996 | n/a | n/a | 2.1612 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0194 | 0.0194 | 0.0156 | 0.0241 | 0.0027 | 0.1273 | n/a | n/a | 5.0605 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0025 | 0.0025 | 0.0023 | 0.0026 | 0.0001 | 0.1273 | n/a | n/a | 5.0605 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0979 | 0.0997 | 0.0856 | 0.1245 | 0.0132 | 0.1273 | n/a | n/a | 5.0605 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0242 | 0.0238 | 0.0221 | 0.0259 | 0.0014 | 0.3228 | n/a | n/a | 5.6918 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0078 | 0.0082 | 0.0073 | 0.0102 | 0.0010 | 0.3228 | n/a | n/a | 5.6918 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1377 | 0.1378 | 0.1342 | 0.1430 | 0.0031 | 0.3228 | n/a | n/a | 5.6918 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0309 | 0.0316 | 0.0271 | 0.0361 | 0.0036 | 0.1318 | n/a | n/a | 1.2437 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0041 | 0.0041 | 0.0037 | 0.0044 | 0.0003 | 0.1318 | n/a | n/a | 1.2437 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0385 | 0.0377 | 0.0308 | 0.0443 | 0.0054 | 0.1318 | n/a | n/a | 1.2437 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0299 | 0.0281 | 0.0199 | 0.0322 | 0.0044 | 0.1462 | n/a | n/a | 1.1193 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0044 | 0.0047 | 0.0043 | 0.0057 | 0.0005 | 0.1462 | n/a | n/a | 1.1193 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0335 | 0.0346 | 0.0302 | 0.0447 | 0.0052 | 0.1462 | n/a | n/a | 1.1193 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0214 | 0.0221 | 0.0208 | 0.0254 | 0.0017 | 0.1435 | n/a | n/a | 1.7406 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0031 | 0.0031 | 0.0029 | 0.0032 | 0.0001 | 0.1435 | n/a | n/a | 1.7406 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0372 | 0.0366 | 0.0312 | 0.0445 | 0.0046 | 0.1435 | n/a | n/a | 1.7406 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0168 | 0.0177 | 0.0165 | 0.0219 | 0.0021 | 0.2092 | n/a | n/a | 4.7843 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0035 | 0.0038 | 0.0035 | 0.0050 | 0.0006 | 0.2092 | n/a | n/a | 4.7843 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0802 | 0.0788 | 0.0680 | 0.0840 | 0.0057 | 0.2092 | n/a | n/a | 4.7843 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0346 | 0.0378 | 0.0338 | 0.0451 | 0.0046 | 0.3702 | n/a | n/a | 3.7388 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0128 | 0.0134 | 0.0116 | 0.0169 | 0.0019 | 0.3702 | n/a | n/a | 3.7388 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1294 | 0.1297 | 0.1196 | 0.1418 | 0.0074 | 0.3702 | n/a | n/a | 3.7388 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0673 | 0.0705 | 0.0659 | 0.0855 | 0.0075 | 0.1547 | n/a | n/a | 0.5060 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0104 | 0.0110 | 0.0102 | 0.0124 | 0.0009 | 0.1547 | n/a | n/a | 0.5060 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0341 | 0.0355 | 0.0331 | 0.0420 | 0.0033 | 0.1547 | n/a | n/a | 0.5060 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0330 | 0.0336 | 0.0320 | 0.0369 | 0.0018 | 0.2636 | n/a | n/a | 1.2288 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0087 | 0.0083 | 0.0065 | 0.0100 | 0.0013 | 0.2636 | n/a | n/a | 1.2288 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0405 | 0.0411 | 0.0388 | 0.0448 | 0.0023 | 0.2636 | n/a | n/a | 1.2288 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0501 | 0.0486 | 0.0387 | 0.0545 | 0.0053 | 0.1604 | n/a | n/a | 0.8808 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0080 | 0.0082 | 0.0076 | 0.0093 | 0.0006 | 0.1604 | n/a | n/a | 0.8808 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0441 | 0.0424 | 0.0353 | 0.0500 | 0.0058 | 0.1604 | n/a | n/a | 0.8808 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0281 | 0.0274 | 0.0224 | 0.0319 | 0.0042 | 0.5913 | n/a | n/a | 3.2784 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0166 | 0.0169 | 0.0163 | 0.0182 | 0.0007 | 0.5913 | n/a | n/a | 3.2784 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0921 | 0.0915 | 0.0867 | 0.0944 | 0.0027 | 0.5913 | n/a | n/a | 3.2784 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0918 | 0.0901 | 0.0838 | 0.0947 | 0.0041 | 0.3938 | n/a | n/a | 1.7950 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0362 | 0.0376 | 0.0337 | 0.0463 | 0.0045 | 0.3938 | n/a | n/a | 1.7950 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1649 | 0.1641 | 0.1582 | 0.1681 | 0.0034 | 0.3938 | n/a | n/a | 1.7950 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2323 | 0.2435 | 0.2162 | 0.2701 | 0.0220 | 0.1904 | n/a | n/a | 0.1686 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0442 | 0.0465 | 0.0388 | 0.0609 | 0.0076 | 0.1904 | n/a | n/a | 0.1686 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0392 | 0.0402 | 0.0378 | 0.0451 | 0.0026 | 0.1904 | n/a | n/a | 0.1686 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0827 | 0.0861 | 0.0807 | 0.1010 | 0.0076 | 0.2528 | n/a | n/a | 0.5264 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0209 | 0.0209 | 0.0200 | 0.0224 | 0.0009 | 0.2528 | n/a | n/a | 0.5264 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0435 | 0.0452 | 0.0400 | 0.0534 | 0.0047 | 0.2528 | n/a | n/a | 0.5264 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1073 | 0.1059 | 0.1005 | 0.1077 | 0.0027 | 0.2616 | n/a | n/a | 0.4187 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0281 | 0.0284 | 0.0271 | 0.0307 | 0.0013 | 0.2616 | n/a | n/a | 0.4187 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0449 | 0.0456 | 0.0424 | 0.0501 | 0.0029 | 0.2616 | n/a | n/a | 0.4187 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0158 | 0.0167 | 0.0157 | 0.0196 | 0.0015 | 0.1446 | n/a | n/a | 0.9051 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0023 | 0.0020 | 0.0027 | 0.0002 | 0.1446 | n/a | n/a | 0.9051 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0143 | 0.0143 | 0.0131 | 0.0153 | 0.0008 | 0.1446 | n/a | n/a | 0.9051 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0157 | 0.0166 | 0.0154 | 0.0188 | 0.0013 | 0.5655 | n/a | n/a | 0.8108 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0089 | 0.0090 | 0.0079 | 0.0100 | 0.0008 | 0.5655 | n/a | n/a | 0.8108 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0127 | 0.0135 | 0.0112 | 0.0167 | 0.0021 | 0.5655 | n/a | n/a | 0.8108 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0159 | 0.0161 | 0.0157 | 0.0167 | 0.0004 | 0.1375 | n/a | n/a | 0.9196 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0022 | 0.0023 | 0.0021 | 0.0026 | 0.0002 | 0.1375 | n/a | n/a | 0.9196 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0146 | 0.0144 | 0.0126 | 0.0158 | 0.0011 | 0.1375 | n/a | n/a | 0.9196 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0159 | 0.0163 | 0.0153 | 0.0175 | 0.0009 | 0.5384 | n/a | n/a | 0.7552 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0086 | 0.0095 | 0.0082 | 0.0117 | 0.0014 | 0.5384 | n/a | n/a | 0.7552 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0122 | 0.0117 | 0.0132 | 0.0005 | 0.5384 | n/a | n/a | 0.7552 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0174 | 0.0170 | 0.0157 | 0.0177 | 0.0007 | 0.1213 | n/a | n/a | 0.7523 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0021 | 0.0022 | 0.0021 | 0.0026 | 0.0002 | 0.1213 | n/a | n/a | 0.7523 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0131 | 0.0130 | 0.0126 | 0.0132 | 0.0002 | 0.1213 | n/a | n/a | 0.7523 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0158 | 0.0159 | 0.0156 | 0.0162 | 0.0002 | 0.5738 | n/a | n/a | 0.7857 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0091 | 0.0092 | 0.0082 | 0.0101 | 0.0007 | 0.5738 | n/a | n/a | 0.7857 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0124 | 0.0128 | 0.0117 | 0.0143 | 0.0009 | 0.5738 | n/a | n/a | 0.7857 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0156 | 0.0162 | 0.0155 | 0.0181 | 0.0010 | 0.1574 | n/a | n/a | 0.9657 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0025 | 0.0025 | 0.0021 | 0.0031 | 0.0004 | 0.1574 | n/a | n/a | 0.9657 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0150 | 0.0151 | 0.0140 | 0.0166 | 0.0008 | 0.1574 | n/a | n/a | 0.9657 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0156 | 0.0156 | 0.0153 | 0.0161 | 0.0003 | 0.5310 | n/a | n/a | 0.9732 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0083 | 0.0083 | 0.0081 | 0.0084 | 0.0001 | 0.5310 | n/a | n/a | 0.9732 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0152 | 0.0154 | 0.0135 | 0.0172 | 0.0012 | 0.5310 | n/a | n/a | 0.9732 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0174 | 0.0182 | 0.0168 | 0.0221 | 0.0020 | 0.1498 | n/a | n/a | 1.3066 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0026 | 0.0030 | 0.0025 | 0.0036 | 0.0005 | 0.1498 | n/a | n/a | 1.3066 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0228 | 0.0229 | 0.0214 | 0.0244 | 0.0013 | 0.1498 | n/a | n/a | 1.3066 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0170 | 0.0181 | 0.0160 | 0.0224 | 0.0023 | 0.5027 | n/a | n/a | 1.1200 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0086 | 0.0088 | 0.0081 | 0.0106 | 0.0009 | 0.5027 | n/a | n/a | 1.1200 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0191 | 0.0171 | 0.0130 | 0.0192 | 0.0026 | 0.5027 | n/a | n/a | 1.1200 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0203 | 0.0206 | 0.0192 | 0.0232 | 0.0014 | 0.1686 | n/a | n/a | 1.0405 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0034 | 0.0036 | 0.0033 | 0.0040 | 0.0003 | 0.1686 | n/a | n/a | 1.0405 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0212 | 0.0196 | 0.0158 | 0.0224 | 0.0027 | 0.1686 | n/a | n/a | 1.0405 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0220 | 0.0220 | 0.0200 | 0.0246 | 0.0017 | 0.4709 | n/a | n/a | 0.7898 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0103 | 0.0113 | 0.0091 | 0.0142 | 0.0021 | 0.4709 | n/a | n/a | 0.7898 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0174 | 0.0178 | 0.0156 | 0.0198 | 0.0015 | 0.4709 | n/a | n/a | 0.7898 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0326 | 0.0327 | 0.0319 | 0.0344 | 0.0009 | 0.1996 | n/a | n/a | 1.0009 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0065 | 0.0065 | 0.0065 | 0.0065 | 0.0000 | 0.1996 | n/a | n/a | 1.0009 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0326 | 0.0303 | 0.0205 | 0.0363 | 0.0058 | 0.1996 | n/a | n/a | 1.0009 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0324 | 0.0334 | 0.0322 | 0.0369 | 0.0018 | 0.4302 | n/a | n/a | 0.8766 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0139 | 0.0140 | 0.0131 | 0.0156 | 0.0009 | 0.4302 | n/a | n/a | 0.8766 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0284 | 0.0299 | 0.0277 | 0.0346 | 0.0025 | 0.4302 | n/a | n/a | 0.8766 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0162 | 0.0163 | 0.0160 | 0.0167 | 0.0003 | 0.1725 | n/a | n/a | 0.7724 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0028 | 0.0031 | 0.0026 | 0.0039 | 0.0005 | 0.1725 | n/a | n/a | 0.7724 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0125 | 0.0129 | 0.0121 | 0.0139 | 0.0007 | 0.1725 | n/a | n/a | 0.7724 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0104 | 0.0118 | 0.0102 | 0.0141 | 0.0018 | 1.0143 | n/a | n/a | 1.6617 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0105 | 0.0109 | 0.0098 | 0.0133 | 0.0012 | 1.0143 | n/a | n/a | 1.6617 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0172 | 0.0169 | 0.0135 | 0.0202 | 0.0028 | 1.0143 | n/a | n/a | 1.6617 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0194 | 0.0205 | 0.0187 | 0.0258 | 0.0027 | 0.2168 | n/a | n/a | 1.1192 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0049 | 0.0040 | 0.0077 | 0.0014 | 0.2168 | n/a | n/a | 1.1192 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0217 | 0.0216 | 0.0194 | 0.0234 | 0.0013 | 0.2168 | n/a | n/a | 1.1192 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0140 | 0.0158 | 0.0136 | 0.0201 | 0.0026 | 1.2706 | n/a | n/a | 1.4099 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0178 | 0.0170 | 0.0122 | 0.0209 | 0.0037 | 1.2706 | n/a | n/a | 1.4099 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0197 | 0.0192 | 0.0166 | 0.0215 | 0.0017 | 1.2706 | n/a | n/a | 1.4099 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0296 | 0.0295 | 0.0287 | 0.0301 | 0.0005 | 0.2604 | n/a | n/a | 0.8904 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0077 | 0.0084 | 0.0076 | 0.0110 | 0.0013 | 0.2604 | n/a | n/a | 0.8904 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0263 | 0.0273 | 0.0258 | 0.0309 | 0.0019 | 0.2604 | n/a | n/a | 0.8904 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0224 | 0.0231 | 0.0220 | 0.0266 | 0.0017 | 0.6673 | n/a | n/a | 0.8172 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0149 | 0.0152 | 0.0149 | 0.0157 | 0.0003 | 0.6673 | n/a | n/a | 0.8172 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0183 | 0.0181 | 0.0138 | 0.0246 | 0.0040 | 0.6673 | n/a | n/a | 0.8172 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0811 | 0.0814 | 0.0787 | 0.0849 | 0.0020 | 0.3053 | n/a | n/a | 0.9546 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0248 | 0.0247 | 0.0218 | 0.0292 | 0.0027 | 0.3053 | n/a | n/a | 0.9546 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0774 | 0.0769 | 0.0717 | 0.0799 | 0.0028 | 0.3053 | n/a | n/a | 0.9546 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0613 | 0.0609 | 0.0562 | 0.0644 | 0.0027 | 0.4053 | n/a | n/a | 0.4196 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0249 | 0.0254 | 0.0222 | 0.0308 | 0.0031 | 0.4053 | n/a | n/a | 0.4196 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0257 | 0.0272 | 0.0252 | 0.0307 | 0.0021 | 0.4053 | n/a | n/a | 0.4196 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0183 | 0.0185 | 0.0181 | 0.0192 | 0.0005 | 0.1067 | n/a | n/a | 0.8728 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0020 | 0.0000 | 0.1067 | n/a | n/a | 0.8728 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0159 | 0.0142 | 0.0104 | 0.0172 | 0.0031 | 0.1067 | n/a | n/a | 0.8728 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1188 | 0.1233 | 0.1086 | 0.1383 | 0.0115 | 0.1423 | n/a | n/a | 0.4406 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0169 | 0.0159 | 0.0105 | 0.0185 | 0.0028 | 0.1423 | n/a | n/a | 0.4406 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0523 | 0.0506 | 0.0467 | 0.0533 | 0.0027 | 0.1423 | n/a | n/a | 0.4406 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4052 | 0.4099 | 0.3907 | 0.4343 | 0.0145 | 0.8464 | n/a | n/a | 0.3031 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3430 | 0.3495 | 0.3278 | 0.3794 | 0.0184 | 0.8464 | n/a | n/a | 0.3031 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1228 | 0.1260 | 0.1092 | 0.1507 | 0.0149 | 0.8464 | n/a | n/a | 0.3031 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.5256 | 2.6118 | 2.0430 | 3.2932 | 0.4326 | 0.1775 | n/a | n/a | 0.1003 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4482 | 0.4622 | 0.3984 | 0.5301 | 0.0576 | 0.1775 | n/a | n/a | 0.1003 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2533 | 0.2622 | 0.2369 | 0.2899 | 0.0226 | 0.1775 | n/a | n/a | 0.1003 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1924 | 0.1917 | 0.1797 | 0.2036 | 0.0088 | 8.0171 | n/a | n/a | 0.6665 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.5425 | 1.5795 | 1.4912 | 1.7084 | 0.0874 | 8.0171 | n/a | n/a | 0.6665 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1282 | 0.1280 | 0.1232 | 0.1338 | 0.0039 | 8.0171 | n/a | n/a | 0.6665 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 6.2305 | 6.2748 | 6.0316 | 6.6686 | 0.2414 | 11.4369 | n/a | n/a | 0.0502 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 71.2577 | 74.4379 | 65.3868 | 89.6202 | 9.1893 | 11.4369 | n/a | n/a | 0.0502 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3129 | 0.3166 | 0.3039 | 0.3396 | 0.0126 | 11.4369 | n/a | n/a | 0.0502 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0284 | 0.0284 | 0.0272 | 0.0293 | 0.0007 | 5.9745 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1697 | 0.1671 | 0.1621 | 0.1708 | 0.0037 | 5.9745 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0270 | 0.0275 | 0.0265 | 0.0291 | 0.0011 | 22.0918 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5957 | 0.5981 | 0.5639 | 0.6477 | 0.0292 | 22.0918 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6852 | 0.6847 | 0.6751 | 0.6919 | 0.0058 | 11.9981 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.2217 | 8.3966 | 7.9501 | 9.2907 | 0.4700 | 11.9981 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7050 | 0.7027 | 0.6836 | 0.7157 | 0.0108 | 40.8351 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 28.7891 | 29.7118 | 28.3775 | 31.5286 | 1.3613 | 40.8351 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1748 | 0.1731 | 0.1570 | 0.1921 | 0.0141 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.4225 | 0.4141 | 0.3648 | 0.4329 | 0.0253 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2681 | 0.2880 | 0.2494 | 0.3322 | 0.0353 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.6185 | 0.6424 | 0.6032 | 0.7427 | 0.0512 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.2827 | 1.3147 | 1.2123 | 1.4789 | 0.0992 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.8153 | 2.8530 | 2.7472 | 3.0164 | 0.0908 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.4242 | 5.4138 | 5.3113 | 5.4957 | 0.0649 | 0.0287 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1557 | 0.1570 | 0.1520 | 0.1646 | 0.0044 | 0.0287 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
