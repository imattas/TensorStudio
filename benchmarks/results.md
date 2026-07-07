# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.12.0`
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
- TensorStudio wins versus JAX CPU dispatch: `39`
- TensorStudio losses versus JAX CPU dispatch: `59`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0188 | 0.0190 | 0.0164 | 0.0219 | 0.0022 | 0.0525 | n/a | n/a | 0.7634 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0010 | 0.0013 | 0.0007 | 0.0029 | 0.0008 | 0.0525 | n/a | n/a | 0.7634 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0144 | 0.0308 | 0.0133 | 0.0587 | 0.0207 | 0.0525 | n/a | n/a | 0.7634 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0188 | 0.0195 | 0.0175 | 0.0225 | 0.0018 | 0.0403 | n/a | n/a | 0.9809 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0001 | 0.0403 | n/a | n/a | 0.9809 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0184 | 0.0177 | 0.0134 | 0.0198 | 0.0022 | 0.0403 | n/a | n/a | 0.9809 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0208 | 0.0199 | 0.0165 | 0.0230 | 0.0027 | 0.0408 | n/a | n/a | 0.6083 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0408 | n/a | n/a | 0.6083 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0127 | 0.0134 | 0.0125 | 0.0151 | 0.0011 | 0.0408 | n/a | n/a | 0.6083 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0197 | 0.0210 | 0.0177 | 0.0255 | 0.0028 | 0.0391 | n/a | n/a | 0.5000 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0000 | 0.0391 | n/a | n/a | 0.5000 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0099 | 0.0100 | 0.0095 | 0.0111 | 0.0006 | 0.0391 | n/a | n/a | 0.5000 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0900 | 0.0900 | 0.0853 | 0.0930 | 0.0027 | 0.0612 | n/a | n/a | 1.0103 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0055 | 0.0063 | 0.0054 | 0.0095 | 0.0016 | 0.0612 | n/a | n/a | 1.0103 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0910 | 0.0921 | 0.0898 | 0.0956 | 0.0022 | 0.0612 | n/a | n/a | 1.0103 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0175 | 0.0182 | 0.0163 | 0.0211 | 0.0018 | 0.0372 | n/a | n/a | 0.7168 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0372 | n/a | n/a | 0.7168 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0126 | 0.0126 | 0.0120 | 0.0134 | 0.0005 | 0.0372 | n/a | n/a | 0.7168 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0208 | 0.0254 | 0.0174 | 0.0401 | 0.0089 | 0.0358 | n/a | n/a | 0.6190 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0358 | n/a | n/a | 0.6190 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0129 | 0.0132 | 0.0127 | 0.0143 | 0.0006 | 0.0358 | n/a | n/a | 0.6190 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0160 | 0.0163 | 0.0159 | 0.0170 | 0.0005 | 0.0442 | n/a | n/a | 0.8075 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0442 | n/a | n/a | 0.8075 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0129 | 0.0134 | 0.0123 | 0.0150 | 0.0010 | 0.0442 | n/a | n/a | 0.8075 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0161 | 0.0164 | 0.0157 | 0.0174 | 0.0006 | 0.0445 | n/a | n/a | 0.5743 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0445 | n/a | n/a | 0.5743 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0093 | 0.0094 | 0.0092 | 0.0098 | 0.0002 | 0.0445 | n/a | n/a | 0.5743 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0845 | 0.0851 | 0.0839 | 0.0868 | 0.0011 | 0.0655 | n/a | n/a | 1.0761 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0055 | 0.0056 | 0.0055 | 0.0057 | 0.0001 | 0.0655 | n/a | n/a | 1.0761 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0909 | 0.0900 | 0.0882 | 0.0911 | 0.0012 | 0.0655 | n/a | n/a | 1.0761 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0161 | 0.0002 | 0.0424 | n/a | n/a | 0.7713 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0009 | 0.0001 | 0.0424 | n/a | n/a | 0.7713 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0122 | 0.0121 | 0.0118 | 0.0125 | 0.0002 | 0.0424 | n/a | n/a | 0.7713 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0159 | 0.0159 | 0.0158 | 0.0160 | 0.0001 | 0.0440 | n/a | n/a | 0.7700 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0010 | 0.0001 | 0.0440 | n/a | n/a | 0.7700 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0120 | 0.0127 | 0.0003 | 0.0440 | n/a | n/a | 0.7700 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0159 | 0.0160 | 0.0158 | 0.0163 | 0.0002 | 0.0467 | n/a | n/a | 0.7754 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0010 | 0.0007 | 0.0015 | 0.0004 | 0.0467 | n/a | n/a | 0.7754 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0123 | 0.0125 | 0.0122 | 0.0129 | 0.0003 | 0.0467 | n/a | n/a | 0.7754 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0164 | 0.0166 | 0.0159 | 0.0179 | 0.0007 | 0.0436 | n/a | n/a | 0.5856 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0436 | n/a | n/a | 0.5856 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0096 | 0.0096 | 0.0094 | 0.0099 | 0.0002 | 0.0436 | n/a | n/a | 0.5856 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0847 | 0.0848 | 0.0840 | 0.0857 | 0.0007 | 0.0656 | n/a | n/a | 1.0643 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0056 | 0.0055 | 0.0055 | 0.0056 | 0.0001 | 0.0656 | n/a | n/a | 1.0643 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0902 | 0.0903 | 0.0890 | 0.0917 | 0.0010 | 0.0656 | n/a | n/a | 1.0643 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0158 | 0.0158 | 0.0157 | 0.0158 | 0.0000 | 0.0422 | n/a | n/a | 0.7636 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0008 | 0.0001 | 0.0422 | n/a | n/a | 0.7636 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0120 | 0.0129 | 0.0119 | 0.0161 | 0.0016 | 0.0422 | n/a | n/a | 0.7636 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0162 | 0.0162 | 0.0161 | 0.0165 | 0.0001 | 0.0440 | n/a | n/a | 0.7601 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.0440 | n/a | n/a | 0.7601 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0123 | 0.0123 | 0.0121 | 0.0125 | 0.0001 | 0.0440 | n/a | n/a | 0.7601 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0158 | 0.0158 | 0.0157 | 0.0159 | 0.0001 | 0.0444 | n/a | n/a | 0.8004 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0444 | n/a | n/a | 0.8004 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0127 | 0.0128 | 0.0122 | 0.0139 | 0.0006 | 0.0444 | n/a | n/a | 0.8004 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0160 | 0.0160 | 0.0158 | 0.0161 | 0.0001 | 0.0458 | n/a | n/a | 0.6041 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0458 | n/a | n/a | 0.6041 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0097 | 0.0098 | 0.0096 | 0.0102 | 0.0002 | 0.0458 | n/a | n/a | 0.6041 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0849 | 0.0852 | 0.0842 | 0.0870 | 0.0011 | 0.0651 | n/a | n/a | 1.0335 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0055 | 0.0056 | 0.0055 | 0.0057 | 0.0001 | 0.0651 | n/a | n/a | 1.0335 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0877 | 0.0886 | 0.0872 | 0.0905 | 0.0014 | 0.0651 | n/a | n/a | 1.0335 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0160 | 0.0160 | 0.0159 | 0.0162 | 0.0001 | 0.0602 | n/a | n/a | 0.9327 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.0602 | n/a | n/a | 0.9327 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0150 | 0.0151 | 0.0122 | 0.0193 | 0.0025 | 0.0602 | n/a | n/a | 0.9327 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0164 | 0.0166 | 0.0161 | 0.0180 | 0.0007 | 0.0617 | n/a | n/a | 0.8510 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0012 | 0.0001 | 0.0617 | n/a | n/a | 0.8510 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0139 | 0.0141 | 0.0135 | 0.0152 | 0.0006 | 0.0617 | n/a | n/a | 0.8510 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0163 | 0.0162 | 0.0157 | 0.0165 | 0.0003 | 0.0642 | n/a | n/a | 0.9497 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0011 | 0.0000 | 0.0642 | n/a | n/a | 0.9497 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0155 | 0.0150 | 0.0131 | 0.0176 | 0.0016 | 0.0642 | n/a | n/a | 0.9497 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0162 | 0.0162 | 0.0159 | 0.0164 | 0.0002 | 0.0672 | n/a | n/a | 0.7362 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0012 | 0.0011 | 0.0016 | 0.0002 | 0.0672 | n/a | n/a | 0.7362 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0119 | 0.0125 | 0.0100 | 0.0170 | 0.0025 | 0.0672 | n/a | n/a | 0.7362 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0869 | 0.0871 | 0.0855 | 0.0904 | 0.0018 | 0.0870 | n/a | n/a | 1.4126 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0076 | 0.0076 | 0.0072 | 0.0085 | 0.0005 | 0.0870 | n/a | n/a | 1.4126 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1227 | 0.1222 | 0.1155 | 0.1282 | 0.0048 | 0.0870 | n/a | n/a | 1.4126 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0171 | 0.0172 | 0.0168 | 0.0177 | 0.0004 | 0.0996 | n/a | n/a | 0.8312 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0019 | 0.0001 | 0.0996 | n/a | n/a | 0.8312 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0142 | 0.0150 | 0.0130 | 0.0197 | 0.0025 | 0.0996 | n/a | n/a | 0.8312 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0174 | 0.0173 | 0.0172 | 0.0175 | 0.0001 | 0.0865 | n/a | n/a | 0.9120 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.0865 | n/a | n/a | 0.9120 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0159 | 0.0165 | 0.0154 | 0.0192 | 0.0014 | 0.0865 | n/a | n/a | 0.9120 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0173 | 0.0177 | 0.0167 | 0.0198 | 0.0011 | 0.0899 | n/a | n/a | 0.8888 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.0899 | n/a | n/a | 0.8888 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0154 | 0.0151 | 0.0132 | 0.0167 | 0.0012 | 0.0899 | n/a | n/a | 0.8888 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0178 | 0.0183 | 0.0172 | 0.0210 | 0.0014 | 0.1221 | n/a | n/a | 0.6250 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0022 | 0.0023 | 0.0016 | 0.0030 | 0.0006 | 0.1221 | n/a | n/a | 0.6250 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0111 | 0.0119 | 0.0104 | 0.0159 | 0.0020 | 0.1221 | n/a | n/a | 0.6250 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0899 | 0.0902 | 0.0889 | 0.0917 | 0.0010 | 0.1133 | n/a | n/a | 1.2470 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0102 | 0.0101 | 0.0097 | 0.0103 | 0.0002 | 0.1133 | n/a | n/a | 1.2470 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1121 | 0.1148 | 0.1099 | 0.1217 | 0.0051 | 0.1133 | n/a | n/a | 1.2470 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0213 | 0.0213 | 0.0208 | 0.0221 | 0.0005 | 0.1776 | n/a | n/a | 0.6309 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0038 | 0.0038 | 0.0000 | 0.1776 | n/a | n/a | 0.6309 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0134 | 0.0138 | 0.0128 | 0.0158 | 0.0011 | 0.1776 | n/a | n/a | 0.6309 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0213 | 0.0216 | 0.0208 | 0.0228 | 0.0007 | 0.1880 | n/a | n/a | 0.5975 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0040 | 0.0039 | 0.0038 | 0.0041 | 0.0001 | 0.1880 | n/a | n/a | 0.5975 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0128 | 0.0142 | 0.0126 | 0.0179 | 0.0021 | 0.1880 | n/a | n/a | 0.5975 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0208 | 0.0210 | 0.0206 | 0.0222 | 0.0006 | 0.1764 | n/a | n/a | 0.6536 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0037 | 0.0000 | 0.1764 | n/a | n/a | 0.6536 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0136 | 0.0157 | 0.0132 | 0.0206 | 0.0030 | 0.1764 | n/a | n/a | 0.6536 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0263 | 0.0255 | 0.0205 | 0.0328 | 0.0046 | 0.1629 | n/a | n/a | 0.4288 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0043 | 0.0046 | 0.0043 | 0.0058 | 0.0006 | 0.1629 | n/a | n/a | 0.4288 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0113 | 0.0123 | 0.0110 | 0.0168 | 0.0023 | 0.1629 | n/a | n/a | 0.4288 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1058 | 0.1057 | 0.1030 | 0.1091 | 0.0021 | 0.2133 | n/a | n/a | 1.3214 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0226 | 0.0230 | 0.0222 | 0.0241 | 0.0008 | 0.2133 | n/a | n/a | 1.3214 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1398 | 0.1395 | 0.1241 | 0.1511 | 0.0088 | 0.2133 | n/a | n/a | 1.3214 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0148 | 0.0000 | 0.1126 | n/a | n/a | 3.0433 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.1126 | n/a | n/a | 3.0433 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0450 | 0.0454 | 0.0447 | 0.0469 | 0.0008 | 0.1126 | n/a | n/a | 3.0433 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0150 | 0.0149 | 0.0147 | 0.0150 | 0.0001 | 0.2897 | n/a | n/a | 5.0151 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0043 | 0.0043 | 0.0043 | 0.0044 | 0.0001 | 0.2897 | n/a | n/a | 5.0151 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0750 | 0.0750 | 0.0742 | 0.0760 | 0.0007 | 0.2897 | n/a | n/a | 5.0151 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0154 | 0.0002 | 0.0816 | n/a | n/a | 2.0028 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0816 | n/a | n/a | 2.0028 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0297 | 0.0301 | 0.0291 | 0.0311 | 0.0008 | 0.0816 | n/a | n/a | 2.0028 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0149 | 0.0001 | 0.0822 | n/a | n/a | 1.9947 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0822 | n/a | n/a | 1.9947 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0296 | 0.0298 | 0.0295 | 0.0308 | 0.0005 | 0.0822 | n/a | n/a | 1.9947 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0149 | 0.0148 | 0.0147 | 0.0149 | 0.0001 | 0.0888 | n/a | n/a | 2.0044 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.0888 | n/a | n/a | 2.0044 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0298 | 0.0307 | 0.0293 | 0.0324 | 0.0014 | 0.0888 | n/a | n/a | 2.0044 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0149 | 0.0001 | 0.1141 | n/a | n/a | 2.9932 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.1141 | n/a | n/a | 2.9932 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0442 | 0.0440 | 0.0435 | 0.0442 | 0.0003 | 0.1141 | n/a | n/a | 2.9932 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0149 | 0.0001 | 0.2982 | n/a | n/a | 5.0409 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0001 | 0.2982 | n/a | n/a | 5.0409 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0745 | 0.0744 | 0.0731 | 0.0750 | 0.0007 | 0.2982 | n/a | n/a | 5.0409 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0150 | 0.0001 | 0.0819 | n/a | n/a | 2.0199 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0819 | n/a | n/a | 2.0199 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0300 | 0.0300 | 0.0295 | 0.0304 | 0.0003 | 0.0819 | n/a | n/a | 2.0199 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0150 | 0.0150 | 0.0148 | 0.0153 | 0.0002 | 0.0822 | n/a | n/a | 2.0139 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0822 | n/a | n/a | 2.0139 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0302 | 0.0304 | 0.0294 | 0.0324 | 0.0011 | 0.0822 | n/a | n/a | 2.0139 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0147 | 0.0147 | 0.0147 | 0.0148 | 0.0000 | 0.0872 | n/a | n/a | 2.0494 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.0872 | n/a | n/a | 2.0494 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0302 | 0.0304 | 0.0298 | 0.0314 | 0.0005 | 0.0872 | n/a | n/a | 2.0494 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0148 | 0.0149 | 0.0146 | 0.0156 | 0.0003 | 0.1163 | n/a | n/a | 2.9888 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0016 | 0.0023 | 0.0002 | 0.1163 | n/a | n/a | 2.9888 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0442 | 0.0442 | 0.0439 | 0.0446 | 0.0002 | 0.1163 | n/a | n/a | 2.9888 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0149 | 0.0000 | 0.2933 | n/a | n/a | 5.0713 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0001 | 0.2933 | n/a | n/a | 5.0713 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0753 | 0.0762 | 0.0730 | 0.0804 | 0.0028 | 0.2933 | n/a | n/a | 5.0713 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0152 | 0.0001 | 0.0847 | n/a | n/a | 2.0413 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0001 | 0.0847 | n/a | n/a | 2.0413 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0306 | 0.0310 | 0.0302 | 0.0331 | 0.0011 | 0.0847 | n/a | n/a | 2.0413 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0151 | 0.0153 | 0.0150 | 0.0166 | 0.0006 | 0.0816 | n/a | n/a | 1.9974 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0816 | n/a | n/a | 1.9974 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0301 | 0.0301 | 0.0297 | 0.0306 | 0.0003 | 0.0816 | n/a | n/a | 1.9974 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0150 | 0.0001 | 0.0875 | n/a | n/a | 1.9844 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.0875 | n/a | n/a | 1.9844 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0298 | 0.0298 | 0.0294 | 0.0304 | 0.0004 | 0.0875 | n/a | n/a | 1.9844 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0148 | 0.0150 | 0.0147 | 0.0154 | 0.0003 | 0.1142 | n/a | n/a | 3.0213 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.1142 | n/a | n/a | 3.0213 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0446 | 0.0446 | 0.0439 | 0.0453 | 0.0005 | 0.1142 | n/a | n/a | 3.0213 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0158 | 0.0159 | 0.0153 | 0.0166 | 0.0005 | 0.3294 | n/a | n/a | 4.6628 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0052 | 0.0058 | 0.0045 | 0.0072 | 0.0011 | 0.3294 | n/a | n/a | 4.6628 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0736 | 0.0751 | 0.0716 | 0.0801 | 0.0030 | 0.3294 | n/a | n/a | 4.6628 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0160 | 0.0161 | 0.0159 | 0.0165 | 0.0002 | 0.1063 | n/a | n/a | 1.8821 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0017 | 0.0020 | 0.0015 | 0.0032 | 0.0006 | 0.1063 | n/a | n/a | 1.8821 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0302 | 0.0302 | 0.0301 | 0.0304 | 0.0001 | 0.1063 | n/a | n/a | 1.8821 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0153 | 0.0153 | 0.0152 | 0.0154 | 0.0001 | 0.0919 | n/a | n/a | 1.9900 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.0919 | n/a | n/a | 1.9900 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0304 | 0.0303 | 0.0298 | 0.0306 | 0.0003 | 0.0919 | n/a | n/a | 1.9900 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0156 | 0.0156 | 0.0154 | 0.0160 | 0.0002 | 0.0956 | n/a | n/a | 1.9898 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0014 | 0.0021 | 0.0002 | 0.0956 | n/a | n/a | 1.9898 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0310 | 0.0308 | 0.0302 | 0.0312 | 0.0004 | 0.0956 | n/a | n/a | 1.9898 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0156 | 0.0155 | 0.0151 | 0.0157 | 0.0002 | 0.1493 | n/a | n/a | 4.0525 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.1493 | n/a | n/a | 4.0525 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0631 | 0.0626 | 0.0545 | 0.0692 | 0.0055 | 0.1493 | n/a | n/a | 4.0525 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0194 | 0.0200 | 0.0192 | 0.0223 | 0.0012 | 0.3572 | n/a | n/a | 4.6970 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0069 | 0.0068 | 0.0066 | 0.0070 | 0.0002 | 0.3572 | n/a | n/a | 4.6970 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0913 | 0.0906 | 0.0860 | 0.0948 | 0.0031 | 0.3572 | n/a | n/a | 4.6970 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0265 | 0.0271 | 0.0255 | 0.0302 | 0.0016 | 0.2154 | n/a | n/a | 1.1372 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0057 | 0.0057 | 0.0041 | 0.0068 | 0.0010 | 0.2154 | n/a | n/a | 1.1372 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0301 | 0.0317 | 0.0299 | 0.0380 | 0.0031 | 0.2154 | n/a | n/a | 1.1372 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0192 | 0.0194 | 0.0187 | 0.0205 | 0.0006 | 0.1390 | n/a | n/a | 1.5649 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.1390 | n/a | n/a | 1.5649 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0300 | 0.0304 | 0.0298 | 0.0311 | 0.0006 | 0.1390 | n/a | n/a | 1.5649 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0202 | 0.0204 | 0.0201 | 0.0210 | 0.0003 | 0.1576 | n/a | n/a | 1.5904 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0032 | 0.0031 | 0.0029 | 0.0033 | 0.0001 | 0.1576 | n/a | n/a | 1.5904 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0322 | 0.0323 | 0.0313 | 0.0334 | 0.0007 | 0.1576 | n/a | n/a | 1.5904 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0167 | 0.0166 | 0.0160 | 0.0170 | 0.0003 | 0.2206 | n/a | n/a | 3.6789 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0037 | 0.0036 | 0.0035 | 0.0038 | 0.0001 | 0.2206 | n/a | n/a | 3.6789 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0613 | 0.0617 | 0.0565 | 0.0655 | 0.0031 | 0.2206 | n/a | n/a | 3.6789 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0334 | 0.0333 | 0.0331 | 0.0334 | 0.0001 | 0.3587 | n/a | n/a | 3.0086 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0120 | 0.0120 | 0.0115 | 0.0125 | 0.0004 | 0.3587 | n/a | n/a | 3.0086 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1003 | 0.1006 | 0.0971 | 0.1035 | 0.0024 | 0.3587 | n/a | n/a | 3.0086 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0660 | 0.0675 | 0.0627 | 0.0743 | 0.0045 | 0.1599 | n/a | n/a | 0.5094 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0106 | 0.0105 | 0.0103 | 0.0106 | 0.0001 | 0.1599 | n/a | n/a | 0.5094 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0336 | 0.0340 | 0.0331 | 0.0360 | 0.0010 | 0.1599 | n/a | n/a | 0.5094 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0306 | 0.0306 | 0.0304 | 0.0310 | 0.0002 | 0.2028 | n/a | n/a | 1.0946 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0062 | 0.0062 | 0.0060 | 0.0062 | 0.0001 | 0.2028 | n/a | n/a | 1.0946 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0335 | 0.0334 | 0.0329 | 0.0336 | 0.0003 | 0.2028 | n/a | n/a | 1.0946 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0366 | 0.0366 | 0.0365 | 0.0367 | 0.0001 | 0.1969 | n/a | n/a | 0.9323 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0072 | 0.0073 | 0.0072 | 0.0077 | 0.0002 | 0.1969 | n/a | n/a | 0.9323 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0341 | 0.0342 | 0.0336 | 0.0351 | 0.0005 | 0.1969 | n/a | n/a | 0.9323 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0212 | 0.0214 | 0.0212 | 0.0218 | 0.0003 | 0.4050 | n/a | n/a | 3.3581 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0086 | 0.0090 | 0.0086 | 0.0105 | 0.0007 | 0.4050 | n/a | n/a | 3.3581 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0713 | 0.0756 | 0.0608 | 0.0973 | 0.0133 | 0.4050 | n/a | n/a | 3.3581 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0886 | 0.0970 | 0.0877 | 0.1290 | 0.0161 | 0.3727 | n/a | n/a | 1.1430 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0330 | 0.0338 | 0.0321 | 0.0368 | 0.0018 | 0.3727 | n/a | n/a | 1.1430 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1013 | 0.1047 | 0.0927 | 0.1245 | 0.0112 | 0.3727 | n/a | n/a | 1.1430 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2121 | 0.2163 | 0.2106 | 0.2318 | 0.0079 | 0.1894 | n/a | n/a | 0.1799 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0402 | 0.0428 | 0.0391 | 0.0548 | 0.0060 | 0.1894 | n/a | n/a | 0.1799 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0382 | 0.0400 | 0.0376 | 0.0453 | 0.0029 | 0.1894 | n/a | n/a | 0.1799 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0767 | 0.0767 | 0.0761 | 0.0773 | 0.0004 | 0.2519 | n/a | n/a | 0.5582 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0193 | 0.0198 | 0.0191 | 0.0217 | 0.0010 | 0.2519 | n/a | n/a | 0.5582 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0428 | 0.0433 | 0.0423 | 0.0453 | 0.0011 | 0.2519 | n/a | n/a | 0.5582 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1003 | 0.1030 | 0.0997 | 0.1140 | 0.0055 | 0.2526 | n/a | n/a | 0.4478 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0253 | 0.0252 | 0.0246 | 0.0258 | 0.0005 | 0.2526 | n/a | n/a | 0.4478 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0449 | 0.0449 | 0.0427 | 0.0470 | 0.0014 | 0.2526 | n/a | n/a | 0.4478 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0153 | 0.0153 | 0.0151 | 0.0157 | 0.0002 | 0.1492 | n/a | n/a | 0.8288 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0023 | 0.0021 | 0.0024 | 0.0001 | 0.1492 | n/a | n/a | 0.8288 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0127 | 0.0131 | 0.0122 | 0.0142 | 0.0008 | 0.1492 | n/a | n/a | 0.8288 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0153 | 0.0153 | 0.0152 | 0.0155 | 0.0001 | 0.5171 | n/a | n/a | 0.7845 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0079 | 0.0079 | 0.0078 | 0.0082 | 0.0001 | 0.5171 | n/a | n/a | 0.7845 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0120 | 0.0122 | 0.0116 | 0.0130 | 0.0005 | 0.5171 | n/a | n/a | 0.7845 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0154 | 0.0156 | 0.0152 | 0.0165 | 0.0005 | 0.1448 | n/a | n/a | 0.8656 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0000 | 0.1448 | n/a | n/a | 0.8656 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0133 | 0.0136 | 0.0128 | 0.0149 | 0.0008 | 0.1448 | n/a | n/a | 0.8656 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0154 | 0.0157 | 0.0152 | 0.0174 | 0.0008 | 0.5036 | n/a | n/a | 0.7952 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0078 | 0.0078 | 0.0078 | 0.0078 | 0.0000 | 0.5036 | n/a | n/a | 0.7952 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0123 | 0.0122 | 0.0118 | 0.0124 | 0.0002 | 0.5036 | n/a | n/a | 0.7952 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0152 | 0.0152 | 0.0151 | 0.0152 | 0.0001 | 0.1431 | n/a | n/a | 0.8760 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 0.1431 | n/a | n/a | 0.8760 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0133 | 0.0135 | 0.0128 | 0.0145 | 0.0006 | 0.1431 | n/a | n/a | 0.8760 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0154 | 0.0155 | 0.0154 | 0.0159 | 0.0002 | 0.5173 | n/a | n/a | 0.7927 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0080 | 0.0079 | 0.0078 | 0.0080 | 0.0001 | 0.5173 | n/a | n/a | 0.7927 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0122 | 0.0123 | 0.0121 | 0.0126 | 0.0002 | 0.5173 | n/a | n/a | 0.7927 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0153 | 0.0153 | 0.0152 | 0.0154 | 0.0000 | 0.1461 | n/a | n/a | 0.9628 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.1461 | n/a | n/a | 0.9628 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0147 | 0.0151 | 0.0139 | 0.0182 | 0.0016 | 0.1461 | n/a | n/a | 0.9628 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0154 | 0.0156 | 0.0152 | 0.0167 | 0.0005 | 0.5076 | n/a | n/a | 0.8933 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0078 | 0.0078 | 0.0077 | 0.0080 | 0.0001 | 0.5076 | n/a | n/a | 0.8933 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0138 | 0.0137 | 0.0131 | 0.0140 | 0.0003 | 0.5076 | n/a | n/a | 0.8933 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0164 | 0.0163 | 0.0161 | 0.0164 | 0.0001 | 0.1646 | n/a | n/a | 0.9468 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.1646 | n/a | n/a | 0.9468 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0155 | 0.0158 | 0.0147 | 0.0170 | 0.0009 | 0.1646 | n/a | n/a | 0.9468 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0166 | 0.0166 | 0.0162 | 0.0168 | 0.0003 | 0.4940 | n/a | n/a | 0.9101 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0082 | 0.0083 | 0.0080 | 0.0087 | 0.0003 | 0.4940 | n/a | n/a | 0.9101 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0151 | 0.0148 | 0.0132 | 0.0162 | 0.0013 | 0.4940 | n/a | n/a | 0.9101 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0197 | 0.0196 | 0.0193 | 0.0199 | 0.0002 | 0.1748 | n/a | n/a | 0.9276 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0034 | 0.0037 | 0.0034 | 0.0049 | 0.0006 | 0.1748 | n/a | n/a | 0.9276 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0183 | 0.0176 | 0.0156 | 0.0192 | 0.0015 | 0.1748 | n/a | n/a | 0.9276 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0199 | 0.0199 | 0.0193 | 0.0203 | 0.0003 | 0.4829 | n/a | n/a | 0.8357 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0096 | 0.0095 | 0.0092 | 0.0099 | 0.0002 | 0.4829 | n/a | n/a | 0.8357 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0166 | 0.0165 | 0.0153 | 0.0183 | 0.0011 | 0.4829 | n/a | n/a | 0.8357 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0367 | 0.0362 | 0.0319 | 0.0401 | 0.0034 | 0.1827 | n/a | n/a | 0.7779 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0067 | 0.0068 | 0.0067 | 0.0074 | 0.0003 | 0.1827 | n/a | n/a | 0.7779 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0286 | 0.0283 | 0.0208 | 0.0387 | 0.0062 | 0.1827 | n/a | n/a | 0.7779 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0326 | 0.0329 | 0.0320 | 0.0340 | 0.0008 | 0.4130 | n/a | n/a | 0.8380 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0135 | 0.0134 | 0.0131 | 0.0137 | 0.0002 | 0.4130 | n/a | n/a | 0.8380 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0273 | 0.0264 | 0.0196 | 0.0336 | 0.0050 | 0.4130 | n/a | n/a | 0.8380 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0160 | 0.0164 | 0.0157 | 0.0176 | 0.0007 | 0.1891 | n/a | n/a | 0.8180 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0030 | 0.0031 | 0.0030 | 0.0036 | 0.0002 | 0.1891 | n/a | n/a | 0.8180 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0131 | 0.0132 | 0.0128 | 0.0136 | 0.0003 | 0.1891 | n/a | n/a | 0.8180 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0105 | 0.0104 | 0.0100 | 0.0108 | 0.0003 | 0.9957 | n/a | n/a | 1.2216 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0104 | 0.0115 | 0.0098 | 0.0158 | 0.0022 | 0.9957 | n/a | n/a | 1.2216 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0128 | 0.0128 | 0.0126 | 0.0132 | 0.0002 | 0.9957 | n/a | n/a | 1.2216 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0190 | 0.0199 | 0.0184 | 0.0240 | 0.0021 | 0.2210 | n/a | n/a | 0.7935 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0049 | 0.0041 | 0.0070 | 0.0011 | 0.2210 | n/a | n/a | 0.7935 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0151 | 0.0161 | 0.0148 | 0.0189 | 0.0016 | 0.2210 | n/a | n/a | 0.7935 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0131 | 0.0132 | 0.0129 | 0.0138 | 0.0003 | 0.8902 | n/a | n/a | 1.1349 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0117 | 0.0118 | 0.0112 | 0.0127 | 0.0005 | 0.8902 | n/a | n/a | 1.1349 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0149 | 0.0147 | 0.0142 | 0.0152 | 0.0004 | 0.8902 | n/a | n/a | 1.1349 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0295 | 0.0296 | 0.0291 | 0.0305 | 0.0005 | 0.2609 | n/a | n/a | 0.6199 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0077 | 0.0077 | 0.0077 | 0.0077 | 0.0000 | 0.2609 | n/a | n/a | 0.6199 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0183 | 0.0215 | 0.0177 | 0.0344 | 0.0065 | 0.2609 | n/a | n/a | 0.6199 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0223 | 0.0225 | 0.0222 | 0.0230 | 0.0004 | 0.6562 | n/a | n/a | 0.7181 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0146 | 0.0147 | 0.0145 | 0.0151 | 0.0002 | 0.6562 | n/a | n/a | 0.7181 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0160 | 0.0179 | 0.0133 | 0.0271 | 0.0050 | 0.6562 | n/a | n/a | 0.7181 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0795 | 0.0808 | 0.0784 | 0.0879 | 0.0036 | 0.2761 | n/a | n/a | 0.8446 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0220 | 0.0229 | 0.0217 | 0.0264 | 0.0018 | 0.2761 | n/a | n/a | 0.8446 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0672 | 0.0679 | 0.0622 | 0.0757 | 0.0044 | 0.2761 | n/a | n/a | 0.8446 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0570 | 0.0570 | 0.0565 | 0.0576 | 0.0004 | 0.3802 | n/a | n/a | 0.2909 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0217 | 0.0224 | 0.0214 | 0.0252 | 0.0014 | 0.3802 | n/a | n/a | 0.2909 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0166 | 0.0210 | 0.0158 | 0.0303 | 0.0060 | 0.3802 | n/a | n/a | 0.2909 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0180 | 0.0190 | 0.0180 | 0.0225 | 0.0018 | 0.1088 | n/a | n/a | 0.6319 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0020 | 0.0020 | 0.0020 | 0.0020 | 0.0000 | 0.1088 | n/a | n/a | 0.6319 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0114 | 0.0147 | 0.0100 | 0.0247 | 0.0054 | 0.1088 | n/a | n/a | 0.6319 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0945 | 0.1006 | 0.0908 | 0.1263 | 0.0133 | 0.1112 | n/a | n/a | 0.1719 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0105 | 0.0107 | 0.0105 | 0.0116 | 0.0004 | 0.1112 | n/a | n/a | 0.1719 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0162 | 0.0163 | 0.0162 | 0.0165 | 0.0001 | 0.1112 | n/a | n/a | 0.1719 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3886 | 0.3975 | 0.3865 | 0.4194 | 0.0130 | 0.9052 | n/a | n/a | 0.2867 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3518 | 0.3469 | 0.3144 | 0.3695 | 0.0185 | 0.9052 | n/a | n/a | 0.2867 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1114 | 0.1092 | 0.0847 | 0.1293 | 0.0150 | 0.9052 | n/a | n/a | 0.2867 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.3427 | 2.2776 | 1.9926 | 2.4297 | 0.1510 | 0.1849 | n/a | n/a | 0.1060 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4332 | 0.4322 | 0.4031 | 0.4790 | 0.0273 | 0.1849 | n/a | n/a | 0.1060 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2482 | 0.2518 | 0.2364 | 0.2842 | 0.0175 | 0.1849 | n/a | n/a | 0.1060 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1983 | 0.1993 | 0.1968 | 0.2046 | 0.0027 | 6.3378 | n/a | n/a | 0.5029 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2565 | 1.2863 | 1.2351 | 1.3584 | 0.0475 | 6.3378 | n/a | n/a | 0.5029 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.0997 | 0.1063 | 0.0968 | 0.1334 | 0.0137 | 6.3378 | n/a | n/a | 0.5029 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 4.0998 | 4.1000 | 4.0819 | 4.1238 | 0.0136 | 14.8559 | n/a | n/a | 0.0831 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 60.9062 | 61.0579 | 59.9875 | 63.0410 | 1.1194 | 14.8559 | n/a | n/a | 0.0831 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3405 | 0.3384 | 0.3114 | 0.3535 | 0.0146 | 14.8559 | n/a | n/a | 0.0831 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0275 | 0.0275 | 0.0271 | 0.0279 | 0.0003 | 6.0918 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1678 | 0.1670 | 0.1600 | 0.1739 | 0.0046 | 6.0918 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0275 | 0.0293 | 0.0265 | 0.0377 | 0.0042 | 20.0031 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5499 | 0.5650 | 0.5335 | 0.6287 | 0.0343 | 20.0031 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6054 | 0.6081 | 0.5958 | 0.6259 | 0.0102 | 13.4368 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.1350 | 8.5601 | 7.8337 | 10.6434 | 1.0482 | 13.4368 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.7027 | 0.7031 | 0.6983 | 0.7127 | 0.0052 | 39.6008 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 27.8271 | 28.7687 | 26.9229 | 31.6505 | 1.8495 | 39.6008 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1157 | 0.1157 | 0.1150 | 0.1164 | 0.0005 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2934 | 0.2971 | 0.2892 | 0.3140 | 0.0090 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2444 | 0.2450 | 0.2419 | 0.2502 | 0.0029 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.6045 | 0.6020 | 0.5944 | 0.6076 | 0.0052 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1520 | 1.1610 | 1.1492 | 1.1938 | 0.0168 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7782 | 2.8295 | 2.7559 | 2.9551 | 0.0794 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 4.7430 | 4.7645 | 4.7127 | 4.8343 | 0.0467 | 0.0326 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1547 | 0.1590 | 0.1547 | 0.1758 | 0.0084 | 0.0326 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
