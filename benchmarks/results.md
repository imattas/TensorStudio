# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.15.0`
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
- TensorStudio wins versus JAX CPU dispatch: `36`
- TensorStudio losses versus JAX CPU dispatch: `62`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0197 | 0.0206 | 0.0181 | 0.0245 | 0.0023 | 0.0937 | n/a | n/a | 0.7287 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0018 | 0.0018 | 0.0015 | 0.0021 | 0.0002 | 0.0937 | n/a | n/a | 0.7287 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0144 | 0.0145 | 0.0136 | 0.0161 | 0.0009 | 0.0937 | n/a | n/a | 0.7287 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0287 | 0.0293 | 0.0194 | 0.0359 | 0.0059 | 0.0760 | n/a | n/a | 0.8415 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0022 | 0.0021 | 0.0017 | 0.0025 | 0.0003 | 0.0760 | n/a | n/a | 0.8415 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0242 | 0.0249 | 0.0168 | 0.0392 | 0.0079 | 0.0760 | n/a | n/a | 0.8415 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0222 | 0.0230 | 0.0208 | 0.0254 | 0.0019 | 0.0479 | n/a | n/a | 0.7012 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0011 | 0.0011 | 0.0009 | 0.0016 | 0.0002 | 0.0479 | n/a | n/a | 0.7012 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0156 | 0.0162 | 0.0152 | 0.0180 | 0.0011 | 0.0479 | n/a | n/a | 0.7012 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0169 | 0.0170 | 0.0161 | 0.0190 | 0.0010 | 0.0439 | n/a | n/a | 0.5947 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0439 | n/a | n/a | 0.5947 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0100 | 0.0103 | 0.0095 | 0.0119 | 0.0008 | 0.0439 | n/a | n/a | 0.5947 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.1130 | 0.1099 | 0.0915 | 0.1310 | 0.0160 | 0.0528 | n/a | n/a | 0.8149 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0060 | 0.0062 | 0.0057 | 0.0073 | 0.0006 | 0.0528 | n/a | n/a | 0.8149 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0921 | 0.0930 | 0.0905 | 0.0970 | 0.0024 | 0.0528 | n/a | n/a | 0.8149 | no | n/a | n/a | no | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0162 | 0.0163 | 0.0157 | 0.0168 | 0.0004 | 0.0396 | n/a | n/a | 0.7273 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0006 | 0.0006 | 0.0006 | 0.0000 | 0.0396 | n/a | n/a | 0.7273 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0118 | 0.0136 | 0.0114 | 0.0178 | 0.0025 | 0.0396 | n/a | n/a | 0.7273 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0163 | 0.0165 | 0.0161 | 0.0173 | 0.0004 | 0.0430 | n/a | n/a | 0.7309 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0430 | n/a | n/a | 0.7309 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0119 | 0.0117 | 0.0120 | 0.0001 | 0.0430 | n/a | n/a | 0.7309 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0160 | 0.0165 | 0.0159 | 0.0181 | 0.0008 | 0.0451 | n/a | n/a | 0.7693 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0451 | n/a | n/a | 0.7693 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0123 | 0.0126 | 0.0115 | 0.0138 | 0.0009 | 0.0451 | n/a | n/a | 0.7693 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0158 | 0.0158 | 0.0157 | 0.0159 | 0.0001 | 0.0466 | n/a | n/a | 0.5851 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0466 | n/a | n/a | 0.5851 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0092 | 0.0095 | 0.0089 | 0.0105 | 0.0006 | 0.0466 | n/a | n/a | 0.5851 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.1015 | 0.1050 | 0.0917 | 0.1245 | 0.0116 | 0.0561 | n/a | n/a | 0.8932 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0057 | 0.0057 | 0.0054 | 0.0059 | 0.0002 | 0.0561 | n/a | n/a | 0.8932 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0907 | 0.0926 | 0.0880 | 0.1018 | 0.0048 | 0.0561 | n/a | n/a | 0.8932 | no | n/a | n/a | no | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0159 | 0.0164 | 0.0157 | 0.0188 | 0.0012 | 0.0406 | n/a | n/a | 0.7292 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0406 | n/a | n/a | 0.7292 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0116 | 0.0118 | 0.0115 | 0.0128 | 0.0005 | 0.0406 | n/a | n/a | 0.7292 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0167 | 0.0167 | 0.0160 | 0.0178 | 0.0006 | 0.0410 | n/a | n/a | 0.7167 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.0410 | n/a | n/a | 0.7167 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0120 | 0.0120 | 0.0117 | 0.0124 | 0.0002 | 0.0410 | n/a | n/a | 0.7167 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0202 | 0.0217 | 0.0180 | 0.0276 | 0.0035 | 0.0422 | n/a | n/a | 0.6212 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0009 | 0.0009 | 0.0008 | 0.0011 | 0.0001 | 0.0422 | n/a | n/a | 0.6212 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0126 | 0.0130 | 0.0124 | 0.0144 | 0.0008 | 0.0422 | n/a | n/a | 0.6212 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0182 | 0.0188 | 0.0168 | 0.0210 | 0.0014 | 0.0410 | n/a | n/a | 0.5536 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0410 | n/a | n/a | 0.5536 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0101 | 0.0100 | 0.0095 | 0.0104 | 0.0003 | 0.0410 | n/a | n/a | 0.5536 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0958 | 0.0963 | 0.0883 | 0.1082 | 0.0070 | 0.0578 | n/a | n/a | 0.9912 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0055 | 0.0056 | 0.0053 | 0.0058 | 0.0002 | 0.0578 | n/a | n/a | 0.9912 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0949 | 0.0979 | 0.0914 | 0.1076 | 0.0064 | 0.0578 | n/a | n/a | 0.9912 | no | n/a | n/a | no | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0202 | 0.0224 | 0.0164 | 0.0297 | 0.0054 | 0.0350 | n/a | n/a | 0.6248 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0350 | n/a | n/a | 0.6248 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0126 | 0.0126 | 0.0116 | 0.0133 | 0.0006 | 0.0350 | n/a | n/a | 0.6248 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0169 | 0.0169 | 0.0162 | 0.0175 | 0.0004 | 0.0425 | n/a | n/a | 0.6882 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0425 | n/a | n/a | 0.6882 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0117 | 0.0118 | 0.0115 | 0.0125 | 0.0004 | 0.0425 | n/a | n/a | 0.6882 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0159 | 0.0160 | 0.0158 | 0.0163 | 0.0002 | 0.0437 | n/a | n/a | 0.7326 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.0437 | n/a | n/a | 0.7326 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0117 | 0.0117 | 0.0115 | 0.0120 | 0.0002 | 0.0437 | n/a | n/a | 0.7326 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0162 | 0.0165 | 0.0158 | 0.0182 | 0.0008 | 0.0449 | n/a | n/a | 0.5612 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0449 | n/a | n/a | 0.5612 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0091 | 0.0092 | 0.0089 | 0.0095 | 0.0002 | 0.0449 | n/a | n/a | 0.5612 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0909 | 0.0899 | 0.0857 | 0.0927 | 0.0028 | 0.0634 | n/a | n/a | 1.0519 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0058 | 0.0058 | 0.0055 | 0.0061 | 0.0002 | 0.0634 | n/a | n/a | 1.0519 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0956 | 0.1023 | 0.0894 | 0.1378 | 0.0179 | 0.0634 | n/a | n/a | 1.0519 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0160 | 0.0163 | 0.0159 | 0.0168 | 0.0004 | 0.0689 | n/a | n/a | 0.8132 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0012 | 0.0001 | 0.0689 | n/a | n/a | 0.8132 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0130 | 0.0141 | 0.0126 | 0.0174 | 0.0018 | 0.0689 | n/a | n/a | 0.8132 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0166 | 0.0165 | 0.0161 | 0.0168 | 0.0003 | 0.0619 | n/a | n/a | 0.9175 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0013 | 0.0001 | 0.0619 | n/a | n/a | 0.9175 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0152 | 0.0148 | 0.0128 | 0.0161 | 0.0012 | 0.0619 | n/a | n/a | 0.9175 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0166 | 0.0165 | 0.0161 | 0.0167 | 0.0002 | 0.0647 | n/a | n/a | 0.9254 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0011 | 0.0000 | 0.0647 | n/a | n/a | 0.9254 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0153 | 0.0150 | 0.0130 | 0.0175 | 0.0016 | 0.0647 | n/a | n/a | 0.9254 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0162 | 0.0163 | 0.0160 | 0.0169 | 0.0004 | 0.0737 | n/a | n/a | 0.7112 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0012 | 0.0012 | 0.0011 | 0.0014 | 0.0001 | 0.0737 | n/a | n/a | 0.7112 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0115 | 0.0118 | 0.0096 | 0.0142 | 0.0019 | 0.0737 | n/a | n/a | 0.7112 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0861 | 0.0874 | 0.0855 | 0.0932 | 0.0029 | 0.0861 | n/a | n/a | 1.3773 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0074 | 0.0075 | 0.0073 | 0.0077 | 0.0001 | 0.0861 | n/a | n/a | 1.3773 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1185 | 0.1228 | 0.1137 | 0.1402 | 0.0098 | 0.0861 | n/a | n/a | 1.3773 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0172 | 0.0172 | 0.0170 | 0.0174 | 0.0001 | 0.0871 | n/a | n/a | 0.7841 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.0871 | n/a | n/a | 0.7841 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0135 | 0.0139 | 0.0125 | 0.0155 | 0.0010 | 0.0871 | n/a | n/a | 0.7841 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0170 | 0.0171 | 0.0167 | 0.0176 | 0.0003 | 0.0956 | n/a | n/a | 0.8211 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.0956 | n/a | n/a | 0.8211 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0139 | 0.0141 | 0.0134 | 0.0154 | 0.0007 | 0.0956 | n/a | n/a | 0.8211 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0175 | 0.0184 | 0.0169 | 0.0204 | 0.0015 | 0.0982 | n/a | n/a | 0.8356 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.0982 | n/a | n/a | 0.8356 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0146 | 0.0148 | 0.0143 | 0.0164 | 0.0008 | 0.0982 | n/a | n/a | 0.8356 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0176 | 0.0177 | 0.0173 | 0.0181 | 0.0003 | 0.0965 | n/a | n/a | 0.6867 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.0965 | n/a | n/a | 0.6867 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0121 | 0.0116 | 0.0100 | 0.0124 | 0.0009 | 0.0965 | n/a | n/a | 0.6867 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0939 | 0.0969 | 0.0885 | 0.1077 | 0.0077 | 0.1174 | n/a | n/a | 1.2439 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0110 | 0.0114 | 0.0101 | 0.0128 | 0.0009 | 0.1174 | n/a | n/a | 1.2439 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1167 | 0.1186 | 0.1094 | 0.1274 | 0.0069 | 0.1174 | n/a | n/a | 1.2439 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0205 | 0.0205 | 0.0202 | 0.0208 | 0.0002 | 0.1838 | n/a | n/a | 0.9278 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0039 | 0.0037 | 0.0042 | 0.0002 | 0.1838 | n/a | n/a | 0.9278 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0190 | 0.0189 | 0.0128 | 0.0284 | 0.0058 | 0.1838 | n/a | n/a | 0.9278 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0301 | 0.0303 | 0.0299 | 0.0311 | 0.0005 | 0.2052 | n/a | n/a | 0.8118 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0062 | 0.0061 | 0.0058 | 0.0062 | 0.0002 | 0.2052 | n/a | n/a | 0.8118 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0244 | 0.0232 | 0.0131 | 0.0350 | 0.0084 | 0.2052 | n/a | n/a | 0.8118 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0203 | 0.0204 | 0.0202 | 0.0210 | 0.0003 | 0.1907 | n/a | n/a | 1.2252 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0042 | 0.0001 | 0.1907 | n/a | n/a | 1.2252 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0249 | 0.0242 | 0.0193 | 0.0262 | 0.0025 | 0.1907 | n/a | n/a | 1.2252 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0310 | 0.0303 | 0.0281 | 0.0325 | 0.0016 | 0.1338 | n/a | n/a | 0.7044 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0041 | 0.0042 | 0.0041 | 0.0044 | 0.0001 | 0.1338 | n/a | n/a | 0.7044 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0218 | 0.0221 | 0.0209 | 0.0247 | 0.0014 | 0.1338 | n/a | n/a | 0.7044 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1261 | 0.1270 | 0.1172 | 0.1459 | 0.0104 | 0.2091 | n/a | n/a | 0.9355 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0264 | 0.0307 | 0.0255 | 0.0451 | 0.0074 | 0.2091 | n/a | n/a | 0.9355 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1180 | 0.1176 | 0.1081 | 0.1271 | 0.0061 | 0.2091 | n/a | n/a | 0.9355 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0152 | 0.0157 | 0.0148 | 0.0182 | 0.0013 | 0.1114 | n/a | n/a | 2.9332 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.1114 | n/a | n/a | 2.9332 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0445 | 0.0462 | 0.0440 | 0.0508 | 0.0026 | 0.1114 | n/a | n/a | 2.9332 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0148 | 0.0153 | 0.0148 | 0.0169 | 0.0008 | 0.2999 | n/a | n/a | 5.1742 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0044 | 0.0045 | 0.0043 | 0.0048 | 0.0002 | 0.2999 | n/a | n/a | 5.1742 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0767 | 0.0863 | 0.0719 | 0.1233 | 0.0190 | 0.2999 | n/a | n/a | 5.1742 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0211 | 0.0216 | 0.0191 | 0.0256 | 0.0023 | 0.0705 | n/a | n/a | 1.5304 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0015 | 0.0018 | 0.0014 | 0.0029 | 0.0006 | 0.0705 | n/a | n/a | 1.5304 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0323 | 0.0327 | 0.0304 | 0.0358 | 0.0021 | 0.0705 | n/a | n/a | 1.5304 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0184 | 0.0192 | 0.0149 | 0.0249 | 0.0039 | 0.0671 | n/a | n/a | 1.8213 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0671 | n/a | n/a | 1.8213 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0334 | 0.0355 | 0.0298 | 0.0491 | 0.0071 | 0.0671 | n/a | n/a | 1.8213 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0161 | 0.0164 | 0.0155 | 0.0180 | 0.0009 | 0.0881 | n/a | n/a | 2.0146 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0016 | 0.0001 | 0.0881 | n/a | n/a | 2.0146 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0325 | 0.0339 | 0.0309 | 0.0408 | 0.0036 | 0.0881 | n/a | n/a | 2.0146 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0151 | 0.0175 | 0.0149 | 0.0253 | 0.0040 | 0.1340 | n/a | n/a | 3.0493 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0020 | 0.0021 | 0.0017 | 0.0028 | 0.0004 | 0.1340 | n/a | n/a | 3.0493 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0460 | 0.0463 | 0.0451 | 0.0477 | 0.0011 | 0.1340 | n/a | n/a | 3.0493 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0169 | 0.0173 | 0.0150 | 0.0205 | 0.0020 | 0.7733 | n/a | n/a | 4.7128 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0131 | 0.0155 | 0.0096 | 0.0265 | 0.0061 | 0.7733 | n/a | n/a | 4.7128 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0798 | 0.0784 | 0.0744 | 0.0831 | 0.0034 | 0.7733 | n/a | n/a | 4.7128 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0166 | 0.0173 | 0.0149 | 0.0202 | 0.0021 | 0.0738 | n/a | n/a | 2.0237 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0738 | n/a | n/a | 2.0237 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0337 | 0.0337 | 0.0303 | 0.0373 | 0.0028 | 0.0738 | n/a | n/a | 2.0237 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0153 | 0.0157 | 0.0152 | 0.0171 | 0.0007 | 0.0930 | n/a | n/a | 2.0630 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0014 | 0.0019 | 0.0012 | 0.0029 | 0.0008 | 0.0930 | n/a | n/a | 2.0630 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0316 | 0.0318 | 0.0309 | 0.0333 | 0.0008 | 0.0930 | n/a | n/a | 2.0630 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0160 | 0.0167 | 0.0154 | 0.0195 | 0.0015 | 0.0780 | n/a | n/a | 2.1613 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0780 | n/a | n/a | 2.1613 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0346 | 0.0342 | 0.0315 | 0.0356 | 0.0014 | 0.0780 | n/a | n/a | 2.1613 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0176 | 0.0170 | 0.0154 | 0.0184 | 0.0013 | 0.0991 | n/a | n/a | 2.8528 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.0991 | n/a | n/a | 2.8528 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0503 | 0.0514 | 0.0497 | 0.0561 | 0.0024 | 0.0991 | n/a | n/a | 2.8528 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0198 | 0.0193 | 0.0158 | 0.0225 | 0.0023 | 0.3161 | n/a | n/a | 4.6348 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0062 | 0.0064 | 0.0048 | 0.0092 | 0.0016 | 0.3161 | n/a | n/a | 4.6348 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0916 | 0.0879 | 0.0737 | 0.1054 | 0.0117 | 0.3161 | n/a | n/a | 4.6348 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0162 | 0.0164 | 0.0153 | 0.0183 | 0.0011 | 0.0777 | n/a | n/a | 2.0103 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.0777 | n/a | n/a | 2.0103 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0326 | 0.0353 | 0.0308 | 0.0437 | 0.0049 | 0.0777 | n/a | n/a | 2.0103 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0179 | 0.0181 | 0.0156 | 0.0209 | 0.0023 | 0.0725 | n/a | n/a | 1.8457 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0013 | 0.0001 | 0.0725 | n/a | n/a | 1.8457 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0330 | 0.0329 | 0.0304 | 0.0346 | 0.0014 | 0.0725 | n/a | n/a | 1.8457 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0152 | 0.0153 | 0.0151 | 0.0158 | 0.0003 | 0.0855 | n/a | n/a | 2.2308 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0016 | 0.0013 | 0.0026 | 0.0005 | 0.0855 | n/a | n/a | 2.2308 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0340 | 0.0332 | 0.0306 | 0.0349 | 0.0017 | 0.0855 | n/a | n/a | 2.2308 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0150 | 0.0151 | 0.0150 | 0.0153 | 0.0001 | 0.1136 | n/a | n/a | 2.9926 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.1136 | n/a | n/a | 2.9926 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0449 | 0.0467 | 0.0445 | 0.0502 | 0.0024 | 0.1136 | n/a | n/a | 2.9926 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0161 | 0.0161 | 0.0156 | 0.0168 | 0.0005 | 0.2882 | n/a | n/a | 6.3000 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0046 | 0.0049 | 0.0043 | 0.0058 | 0.0006 | 0.2882 | n/a | n/a | 6.3000 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.1013 | 0.1060 | 0.0848 | 0.1363 | 0.0169 | 0.2882 | n/a | n/a | 6.3000 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0173 | 0.0178 | 0.0165 | 0.0202 | 0.0014 | 0.0889 | n/a | n/a | 1.9102 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.0889 | n/a | n/a | 1.9102 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0330 | 0.0347 | 0.0302 | 0.0402 | 0.0036 | 0.0889 | n/a | n/a | 1.9102 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0169 | 0.0183 | 0.0155 | 0.0261 | 0.0039 | 0.0795 | n/a | n/a | 1.9155 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.0795 | n/a | n/a | 1.9155 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0324 | 0.0325 | 0.0307 | 0.0358 | 0.0017 | 0.0795 | n/a | n/a | 1.9155 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0161 | 0.0160 | 0.0156 | 0.0165 | 0.0003 | 0.0920 | n/a | n/a | 2.0282 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0019 | 0.0014 | 0.0033 | 0.0007 | 0.0920 | n/a | n/a | 2.0282 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0327 | 0.0333 | 0.0307 | 0.0392 | 0.0030 | 0.0920 | n/a | n/a | 2.0282 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0157 | 0.0156 | 0.0154 | 0.0158 | 0.0001 | 0.1558 | n/a | n/a | 4.0401 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0024 | 0.0024 | 0.0023 | 0.0026 | 0.0001 | 0.1558 | n/a | n/a | 4.0401 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0634 | 0.0638 | 0.0592 | 0.0726 | 0.0048 | 0.1558 | n/a | n/a | 4.0401 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0199 | 0.0203 | 0.0192 | 0.0228 | 0.0013 | 0.4686 | n/a | n/a | 4.8359 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0093 | 0.0094 | 0.0071 | 0.0135 | 0.0023 | 0.4686 | n/a | n/a | 4.8359 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0963 | 0.1000 | 0.0929 | 0.1193 | 0.0097 | 0.4686 | n/a | n/a | 4.8359 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0263 | 0.0280 | 0.0255 | 0.0350 | 0.0035 | 0.1386 | n/a | n/a | 1.1588 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0042 | 0.0002 | 0.1386 | n/a | n/a | 1.1588 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0304 | 0.0307 | 0.0294 | 0.0334 | 0.0014 | 0.1386 | n/a | n/a | 1.1588 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0196 | 0.0198 | 0.0194 | 0.0210 | 0.0006 | 0.1385 | n/a | n/a | 1.5438 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.1385 | n/a | n/a | 1.5438 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0302 | 0.0308 | 0.0298 | 0.0337 | 0.0015 | 0.1385 | n/a | n/a | 1.5438 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0206 | 0.0224 | 0.0205 | 0.0266 | 0.0025 | 0.1421 | n/a | n/a | 1.4967 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0029 | 0.0030 | 0.0000 | 0.1421 | n/a | n/a | 1.4967 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0309 | 0.0319 | 0.0303 | 0.0355 | 0.0019 | 0.1421 | n/a | n/a | 1.4967 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0168 | 0.0170 | 0.0166 | 0.0179 | 0.0005 | 0.2256 | n/a | n/a | 4.3433 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0038 | 0.0039 | 0.0038 | 0.0042 | 0.0002 | 0.2256 | n/a | n/a | 4.3433 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0729 | 0.0683 | 0.0575 | 0.0745 | 0.0068 | 0.2256 | n/a | n/a | 4.3433 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0378 | 0.0390 | 0.0341 | 0.0461 | 0.0043 | 0.3186 | n/a | n/a | 2.6505 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0121 | 0.0121 | 0.0117 | 0.0126 | 0.0004 | 0.3186 | n/a | n/a | 2.6505 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1003 | 0.1032 | 0.0940 | 0.1208 | 0.0092 | 0.3186 | n/a | n/a | 2.6505 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0646 | 0.0668 | 0.0628 | 0.0784 | 0.0059 | 0.1731 | n/a | n/a | 0.5245 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0112 | 0.0128 | 0.0106 | 0.0159 | 0.0023 | 0.1731 | n/a | n/a | 0.5245 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0339 | 0.0353 | 0.0323 | 0.0399 | 0.0030 | 0.1731 | n/a | n/a | 0.5245 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0363 | 0.0378 | 0.0333 | 0.0442 | 0.0045 | 0.1951 | n/a | n/a | 0.9423 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0071 | 0.0086 | 0.0065 | 0.0119 | 0.0022 | 0.1951 | n/a | n/a | 0.9423 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0342 | 0.0355 | 0.0336 | 0.0389 | 0.0021 | 0.1951 | n/a | n/a | 0.9423 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0380 | 0.0389 | 0.0378 | 0.0422 | 0.0017 | 0.2417 | n/a | n/a | 0.8987 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0092 | 0.0095 | 0.0071 | 0.0122 | 0.0022 | 0.2417 | n/a | n/a | 0.8987 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0341 | 0.0360 | 0.0339 | 0.0432 | 0.0036 | 0.2417 | n/a | n/a | 0.8987 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0216 | 0.0217 | 0.0214 | 0.0222 | 0.0003 | 0.4035 | n/a | n/a | 4.4935 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0087 | 0.0088 | 0.0086 | 0.0092 | 0.0002 | 0.4035 | n/a | n/a | 4.4935 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0969 | 0.0862 | 0.0602 | 0.1008 | 0.0156 | 0.4035 | n/a | n/a | 4.4935 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0918 | 0.0905 | 0.0852 | 0.0954 | 0.0034 | 0.4214 | n/a | n/a | 1.2892 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0387 | 0.0443 | 0.0324 | 0.0566 | 0.0102 | 0.4214 | n/a | n/a | 1.2892 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1183 | 0.1168 | 0.0969 | 0.1318 | 0.0119 | 0.4214 | n/a | n/a | 1.2892 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2154 | 0.2254 | 0.2125 | 0.2662 | 0.0206 | 0.1774 | n/a | n/a | 0.1771 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0382 | 0.0423 | 0.0366 | 0.0608 | 0.0093 | 0.1774 | n/a | n/a | 0.1771 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0381 | 0.0396 | 0.0379 | 0.0425 | 0.0020 | 0.1774 | n/a | n/a | 0.1771 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0963 | 0.1099 | 0.0863 | 0.1437 | 0.0250 | 0.2272 | n/a | n/a | 0.4318 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0219 | 0.0246 | 0.0200 | 0.0316 | 0.0048 | 0.2272 | n/a | n/a | 0.4318 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0416 | 0.0425 | 0.0401 | 0.0468 | 0.0023 | 0.2272 | n/a | n/a | 0.4318 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1080 | 0.1174 | 0.1029 | 0.1522 | 0.0184 | 0.2308 | n/a | n/a | 0.4174 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0249 | 0.0276 | 0.0243 | 0.0382 | 0.0053 | 0.2308 | n/a | n/a | 0.4174 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0451 | 0.0499 | 0.0427 | 0.0709 | 0.0106 | 0.2308 | n/a | n/a | 0.4174 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0175 | 0.0178 | 0.0164 | 0.0194 | 0.0010 | 0.1715 | n/a | n/a | 0.7479 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0030 | 0.0033 | 0.0023 | 0.0045 | 0.0009 | 0.1715 | n/a | n/a | 0.7479 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0131 | 0.0134 | 0.0130 | 0.0145 | 0.0006 | 0.1715 | n/a | n/a | 0.7479 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0184 | 0.0185 | 0.0169 | 0.0199 | 0.0010 | 0.4597 | n/a | n/a | 0.8208 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0085 | 0.0089 | 0.0082 | 0.0098 | 0.0007 | 0.4597 | n/a | n/a | 0.8208 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0151 | 0.0183 | 0.0136 | 0.0248 | 0.0049 | 0.4597 | n/a | n/a | 0.8208 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0176 | 0.0183 | 0.0173 | 0.0208 | 0.0013 | 0.1290 | n/a | n/a | 0.7662 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0026 | 0.0002 | 0.1290 | n/a | n/a | 0.7662 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0135 | 0.0194 | 0.0124 | 0.0373 | 0.0094 | 0.1290 | n/a | n/a | 0.7662 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0159 | 0.0175 | 0.0155 | 0.0203 | 0.0022 | 0.4966 | n/a | n/a | 0.7679 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0079 | 0.0082 | 0.0078 | 0.0095 | 0.0007 | 0.4966 | n/a | n/a | 0.7679 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0122 | 0.0129 | 0.0115 | 0.0162 | 0.0017 | 0.4966 | n/a | n/a | 0.7679 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0155 | 0.0163 | 0.0154 | 0.0179 | 0.0010 | 0.1412 | n/a | n/a | 0.8873 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.1412 | n/a | n/a | 0.8873 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0138 | 0.0136 | 0.0126 | 0.0146 | 0.0007 | 0.1412 | n/a | n/a | 0.8873 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0166 | 0.0171 | 0.0163 | 0.0185 | 0.0008 | 0.6528 | n/a | n/a | 0.8959 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0108 | 0.0114 | 0.0088 | 0.0160 | 0.0025 | 0.6528 | n/a | n/a | 0.8959 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0148 | 0.0163 | 0.0125 | 0.0241 | 0.0040 | 0.6528 | n/a | n/a | 0.8959 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0169 | 0.0167 | 0.0163 | 0.0170 | 0.0003 | 0.1398 | n/a | n/a | 0.9203 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0024 | 0.0024 | 0.0022 | 0.0028 | 0.0002 | 0.1398 | n/a | n/a | 0.9203 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0156 | 0.0157 | 0.0139 | 0.0182 | 0.0015 | 0.1398 | n/a | n/a | 0.9203 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0198 | 0.0199 | 0.0181 | 0.0220 | 0.0013 | 0.7927 | n/a | n/a | 0.8329 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0157 | 0.0153 | 0.0143 | 0.0163 | 0.0008 | 0.7927 | n/a | n/a | 0.8329 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0165 | 0.0175 | 0.0148 | 0.0223 | 0.0026 | 0.7927 | n/a | n/a | 0.8329 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0233 | 0.0225 | 0.0192 | 0.0244 | 0.0018 | 0.1310 | n/a | n/a | 0.9313 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0031 | 0.0035 | 0.0028 | 0.0044 | 0.0007 | 0.1310 | n/a | n/a | 0.9313 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0217 | 0.0212 | 0.0171 | 0.0260 | 0.0032 | 0.1310 | n/a | n/a | 0.9313 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0184 | 0.0183 | 0.0169 | 0.0199 | 0.0011 | 0.5465 | n/a | n/a | 0.9414 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0100 | 0.0113 | 0.0093 | 0.0170 | 0.0029 | 0.5465 | n/a | n/a | 0.9414 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0173 | 0.0194 | 0.0152 | 0.0247 | 0.0039 | 0.5465 | n/a | n/a | 0.9414 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0211 | 0.0215 | 0.0209 | 0.0230 | 0.0008 | 0.1748 | n/a | n/a | 0.8479 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0037 | 0.0037 | 0.0035 | 0.0041 | 0.0002 | 0.1748 | n/a | n/a | 0.8479 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0179 | 0.0179 | 0.0151 | 0.0207 | 0.0018 | 0.1748 | n/a | n/a | 0.8479 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0278 | 0.0265 | 0.0204 | 0.0320 | 0.0042 | 0.4769 | n/a | n/a | 0.6712 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0132 | 0.0135 | 0.0099 | 0.0178 | 0.0033 | 0.4769 | n/a | n/a | 0.6712 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0186 | 0.0187 | 0.0164 | 0.0215 | 0.0017 | 0.4769 | n/a | n/a | 0.6712 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0383 | 0.0384 | 0.0371 | 0.0404 | 0.0011 | 0.4111 | n/a | n/a | 1.0820 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0158 | 0.0147 | 0.0096 | 0.0163 | 0.0026 | 0.4111 | n/a | n/a | 1.0820 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0415 | 0.0440 | 0.0402 | 0.0490 | 0.0040 | 0.4111 | n/a | n/a | 1.0820 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0398 | 0.0404 | 0.0379 | 0.0448 | 0.0025 | 0.3877 | n/a | n/a | 0.8676 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0154 | 0.0166 | 0.0153 | 0.0216 | 0.0025 | 0.3877 | n/a | n/a | 0.8676 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0345 | 0.0346 | 0.0268 | 0.0436 | 0.0059 | 0.3877 | n/a | n/a | 0.8676 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0196 | 0.0216 | 0.0188 | 0.0287 | 0.0037 | 0.2078 | n/a | n/a | 0.7818 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0041 | 0.0038 | 0.0033 | 0.0042 | 0.0004 | 0.2078 | n/a | n/a | 0.7818 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0153 | 0.0196 | 0.0148 | 0.0290 | 0.0059 | 0.2078 | n/a | n/a | 0.7818 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0190 | 0.0190 | 0.0185 | 0.0197 | 0.0004 | 0.8360 | n/a | n/a | 1.0510 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0159 | 0.0177 | 0.0121 | 0.0272 | 0.0058 | 0.8360 | n/a | n/a | 1.0510 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0200 | 0.0226 | 0.0170 | 0.0314 | 0.0057 | 0.8360 | n/a | n/a | 1.0510 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0264 | 0.0272 | 0.0221 | 0.0358 | 0.0047 | 0.1759 | n/a | n/a | 0.7926 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0047 | 0.0050 | 0.0045 | 0.0059 | 0.0005 | 0.1759 | n/a | n/a | 0.7926 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0210 | 0.0213 | 0.0189 | 0.0248 | 0.0021 | 0.1759 | n/a | n/a | 0.7926 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0161 | 0.0171 | 0.0156 | 0.0213 | 0.0021 | 0.8816 | n/a | n/a | 1.3914 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0142 | 0.0166 | 0.0138 | 0.0267 | 0.0050 | 0.8816 | n/a | n/a | 1.3914 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0225 | 0.0244 | 0.0198 | 0.0294 | 0.0037 | 0.8816 | n/a | n/a | 1.3914 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0369 | 0.0381 | 0.0350 | 0.0434 | 0.0029 | 0.2443 | n/a | n/a | 0.5882 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0090 | 0.0090 | 0.0090 | 0.0090 | 0.0000 | 0.2443 | n/a | n/a | 0.5882 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0217 | 0.0231 | 0.0204 | 0.0288 | 0.0030 | 0.2443 | n/a | n/a | 0.5882 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0276 | 0.0272 | 0.0258 | 0.0291 | 0.0012 | 0.6737 | n/a | n/a | 0.5686 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0186 | 0.0190 | 0.0169 | 0.0236 | 0.0024 | 0.6737 | n/a | n/a | 0.5686 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0157 | 0.0159 | 0.0151 | 0.0166 | 0.0006 | 0.6737 | n/a | n/a | 0.5686 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0888 | 0.0893 | 0.0880 | 0.0923 | 0.0016 | 0.2708 | n/a | n/a | 0.8321 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0240 | 0.0243 | 0.0235 | 0.0252 | 0.0007 | 0.2708 | n/a | n/a | 0.8321 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0739 | 0.0751 | 0.0703 | 0.0816 | 0.0041 | 0.2708 | n/a | n/a | 0.8321 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0717 | 0.0713 | 0.0655 | 0.0811 | 0.0057 | 0.3602 | n/a | n/a | 0.3351 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0258 | 0.0266 | 0.0240 | 0.0311 | 0.0024 | 0.3602 | n/a | n/a | 0.3351 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0240 | 0.0238 | 0.0188 | 0.0305 | 0.0039 | 0.3602 | n/a | n/a | 0.3351 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0214 | 0.0224 | 0.0213 | 0.0258 | 0.0018 | 0.1255 | n/a | n/a | 0.6955 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0027 | 0.0027 | 0.0025 | 0.0029 | 0.0002 | 0.1255 | n/a | n/a | 0.6955 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0149 | 0.0158 | 0.0144 | 0.0176 | 0.0014 | 0.1255 | n/a | n/a | 0.6955 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.1300 | 0.1280 | 0.1203 | 0.1357 | 0.0056 | 0.1065 | n/a | n/a | 0.1656 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0138 | 0.0143 | 0.0131 | 0.0163 | 0.0012 | 0.1065 | n/a | n/a | 0.1656 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0215 | 0.0242 | 0.0191 | 0.0377 | 0.0068 | 0.1065 | n/a | n/a | 0.1656 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.4360 | 0.4431 | 0.4136 | 0.4936 | 0.0294 | 0.7630 | n/a | n/a | 0.2808 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3327 | 0.3282 | 0.2863 | 0.3497 | 0.0221 | 0.7630 | n/a | n/a | 0.2808 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1224 | 0.1298 | 0.1201 | 0.1570 | 0.0138 | 0.7630 | n/a | n/a | 0.2808 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.5387 | 2.5117 | 2.1251 | 2.7574 | 0.2208 | 0.2045 | n/a | n/a | 0.0954 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.5193 | 0.5363 | 0.4095 | 0.6914 | 0.0934 | 0.2045 | n/a | n/a | 0.0954 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2421 | 0.2590 | 0.2236 | 0.3202 | 0.0349 | 0.2045 | n/a | n/a | 0.0954 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.2248 | 0.2248 | 0.2216 | 0.2285 | 0.0023 | 8.7672 | n/a | n/a | 0.5916 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.9712 | 1.9274 | 1.5985 | 2.1497 | 0.1799 | 8.7672 | n/a | n/a | 0.5916 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1330 | 0.1340 | 0.1175 | 0.1636 | 0.0161 | 8.7672 | n/a | n/a | 0.5916 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.6281 | 7.6571 | 7.5304 | 7.8370 | 0.1003 | 9.9442 | n/a | n/a | 0.0468 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 75.8550 | 79.5915 | 69.9205 | 89.0276 | 7.6098 | 9.9442 | n/a | n/a | 0.0468 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3570 | 0.3720 | 0.3430 | 0.4335 | 0.0324 | 9.9442 | n/a | n/a | 0.0468 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0342 | 0.0344 | 0.0311 | 0.0404 | 0.0033 | 4.9541 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1692 | 0.1780 | 0.1624 | 0.2027 | 0.0157 | 4.9541 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0312 | 0.0308 | 0.0280 | 0.0329 | 0.0018 | 18.9961 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5922 | 0.5858 | 0.5512 | 0.6204 | 0.0251 | 18.9961 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7252 | 0.7318 | 0.7061 | 0.7913 | 0.0309 | 11.8199 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.5712 | 8.8967 | 7.5694 | 10.7502 | 1.0523 | 11.8199 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.8032 | 0.7972 | 0.7321 | 0.8363 | 0.0354 | 41.5301 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 33.3556 | 32.9669 | 28.8550 | 35.0545 | 2.2113 | 41.5301 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1320 | 0.1409 | 0.1189 | 0.1811 | 0.0237 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.3174 | 0.3473 | 0.3004 | 0.4587 | 0.0574 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2473 | 0.2488 | 0.2439 | 0.2572 | 0.0045 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.5995 | 0.6056 | 0.5960 | 0.6165 | 0.0090 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1573 | 1.1682 | 1.1433 | 1.2194 | 0.0281 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7669 | 2.8717 | 2.7125 | 3.3758 | 0.2537 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.1364 | 5.2366 | 4.8499 | 5.7853 | 0.3263 | 0.0300 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1541 | 0.1540 | 0.1532 | 0.1554 | 0.0007 | 0.0300 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |
| backends | backend_info | `(1,)` | TensorStudio | 0.0025 | 0.0027 | 0.0025 | 0.0033 | 0.0003 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | available_devices | `(1,)` | TensorStudio | 0.0018 | 0.0019 | 0.0017 | 0.0024 | 0.0002 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cpu_transfer | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| backends | cuda_availability_check | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0021 | 0.0027 | 0.0003 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
