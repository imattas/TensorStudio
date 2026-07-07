# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.8.0`
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
- TensorStudio wins versus JAX CPU dispatch: `47`
- TensorStudio losses versus JAX CPU dispatch: `51`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0164 | 0.0169 | 0.0159 | 0.0191 | 0.0011 | 0.0391 | n/a | n/a | 0.7797 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0006 | 0.0009 | 0.0006 | 0.0013 | 0.0003 | 0.0391 | n/a | n/a | 0.7797 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0128 | 0.0144 | 0.0120 | 0.0215 | 0.0036 | 0.0391 | n/a | n/a | 0.7797 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0166 | 0.0167 | 0.0163 | 0.0171 | 0.0003 | 0.0406 | n/a | n/a | 0.7525 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0406 | n/a | n/a | 0.7525 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0125 | 0.0126 | 0.0120 | 0.0134 | 0.0004 | 0.0406 | n/a | n/a | 0.7525 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0159 | 0.0163 | 0.0159 | 0.0173 | 0.0005 | 0.0431 | n/a | n/a | 0.7596 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0431 | n/a | n/a | 0.7596 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0121 | 0.0123 | 0.0120 | 0.0128 | 0.0003 | 0.0431 | n/a | n/a | 0.7596 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0159 | 0.0162 | 0.0158 | 0.0172 | 0.0005 | 0.0446 | n/a | n/a | 0.5782 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0446 | n/a | n/a | 0.5782 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0092 | 0.0093 | 0.0091 | 0.0097 | 0.0002 | 0.0446 | n/a | n/a | 0.5782 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0897 | 0.0895 | 0.0854 | 0.0931 | 0.0025 | 0.0628 | n/a | n/a | 1.2298 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0056 | 0.0057 | 0.0055 | 0.0063 | 0.0003 | 0.0628 | n/a | n/a | 1.2298 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.1103 | 0.1125 | 0.0943 | 0.1328 | 0.0164 | 0.0628 | n/a | n/a | 1.2298 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0160 | 0.0159 | 0.0156 | 0.0161 | 0.0002 | 0.0402 | n/a | n/a | 0.7635 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.0402 | n/a | n/a | 0.7635 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0121 | 0.0124 | 0.0001 | 0.0402 | n/a | n/a | 0.7635 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0160 | 0.0160 | 0.0159 | 0.0161 | 0.0001 | 0.0428 | n/a | n/a | 0.7775 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.0428 | n/a | n/a | 0.7775 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0124 | 0.0123 | 0.0119 | 0.0127 | 0.0003 | 0.0428 | n/a | n/a | 0.7775 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0161 | 0.0163 | 0.0159 | 0.0169 | 0.0004 | 0.0447 | n/a | n/a | 0.7937 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0447 | n/a | n/a | 0.7937 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0128 | 0.0130 | 0.0126 | 0.0139 | 0.0005 | 0.0447 | n/a | n/a | 0.7937 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0161 | 0.0162 | 0.0160 | 0.0168 | 0.0003 | 0.0461 | n/a | n/a | 0.5862 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0461 | n/a | n/a | 0.5862 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0095 | 0.0095 | 0.0094 | 0.0097 | 0.0001 | 0.0461 | n/a | n/a | 0.5862 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0842 | 0.0843 | 0.0838 | 0.0852 | 0.0005 | 0.0664 | n/a | n/a | 1.0910 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0056 | 0.0057 | 0.0054 | 0.0062 | 0.0003 | 0.0664 | n/a | n/a | 1.0910 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0918 | 0.0917 | 0.0897 | 0.0937 | 0.0014 | 0.0664 | n/a | n/a | 1.0910 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0159 | 0.0160 | 0.0158 | 0.0164 | 0.0003 | 0.0408 | n/a | n/a | 0.7844 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0008 | 0.0001 | 0.0408 | n/a | n/a | 0.7844 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0124 | 0.0125 | 0.0122 | 0.0126 | 0.0002 | 0.0408 | n/a | n/a | 0.7844 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0159 | 0.0159 | 0.0158 | 0.0160 | 0.0001 | 0.0460 | n/a | n/a | 0.7733 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0460 | n/a | n/a | 0.7733 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0123 | 0.0123 | 0.0119 | 0.0126 | 0.0002 | 0.0460 | n/a | n/a | 0.7733 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0166 | 0.0167 | 0.0161 | 0.0172 | 0.0005 | 0.0419 | n/a | n/a | 0.7989 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0419 | n/a | n/a | 0.7989 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0133 | 0.0133 | 0.0122 | 0.0141 | 0.0007 | 0.0419 | n/a | n/a | 0.7989 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0173 | 0.0173 | 0.0161 | 0.0188 | 0.0009 | 0.0419 | n/a | n/a | 0.5869 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.0419 | n/a | n/a | 0.5869 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0101 | 0.0108 | 0.0094 | 0.0128 | 0.0015 | 0.0419 | n/a | n/a | 0.5869 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0885 | 0.0890 | 0.0862 | 0.0942 | 0.0027 | 0.0643 | n/a | n/a | 1.0609 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0057 | 0.0059 | 0.0055 | 0.0072 | 0.0006 | 0.0643 | n/a | n/a | 1.0609 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0939 | 0.0947 | 0.0913 | 0.1014 | 0.0035 | 0.0643 | n/a | n/a | 1.0609 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0161 | 0.0163 | 0.0160 | 0.0168 | 0.0003 | 0.0448 | n/a | n/a | 0.7905 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0448 | n/a | n/a | 0.7905 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0128 | 0.0193 | 0.0126 | 0.0396 | 0.0104 | 0.0448 | n/a | n/a | 0.7905 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0161 | 0.0167 | 0.0161 | 0.0191 | 0.0012 | 0.0424 | n/a | n/a | 0.7664 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0424 | n/a | n/a | 0.7664 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0124 | 0.0123 | 0.0121 | 0.0124 | 0.0001 | 0.0424 | n/a | n/a | 0.7664 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0163 | 0.0165 | 0.0160 | 0.0172 | 0.0005 | 0.0435 | n/a | n/a | 0.7971 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.0435 | n/a | n/a | 0.7971 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0130 | 0.0144 | 0.0128 | 0.0200 | 0.0028 | 0.0435 | n/a | n/a | 0.7971 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0159 | 0.0161 | 0.0159 | 0.0168 | 0.0004 | 0.0465 | n/a | n/a | 0.6165 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.0465 | n/a | n/a | 0.6165 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0098 | 0.0098 | 0.0093 | 0.0106 | 0.0005 | 0.0465 | n/a | n/a | 0.6165 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0861 | 0.0862 | 0.0847 | 0.0874 | 0.0009 | 0.0646 | n/a | n/a | 1.0971 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0056 | 0.0056 | 0.0055 | 0.0057 | 0.0001 | 0.0646 | n/a | n/a | 1.0971 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0944 | 0.0968 | 0.0914 | 0.1065 | 0.0058 | 0.0646 | n/a | n/a | 1.0971 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0162 | 0.0168 | 0.0159 | 0.0191 | 0.0012 | 0.0734 | n/a | n/a | 1.0326 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0017 | 0.0002 | 0.0734 | n/a | n/a | 1.0326 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0168 | 0.0175 | 0.0164 | 0.0206 | 0.0016 | 0.0734 | n/a | n/a | 1.0326 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0164 | 0.0164 | 0.0162 | 0.0167 | 0.0002 | 0.0751 | n/a | n/a | 0.9213 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.0751 | n/a | n/a | 0.9213 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0151 | 0.0147 | 0.0127 | 0.0169 | 0.0015 | 0.0751 | n/a | n/a | 0.9213 | no | n/a | n/a | no | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0164 | 0.0170 | 0.0163 | 0.0190 | 0.0010 | 0.0836 | n/a | n/a | 0.9017 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0014 | 0.0013 | 0.0013 | 0.0014 | 0.0001 | 0.0836 | n/a | n/a | 0.9017 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0148 | 0.0157 | 0.0134 | 0.0183 | 0.0018 | 0.0836 | n/a | n/a | 0.9017 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0161 | 0.0162 | 0.0160 | 0.0165 | 0.0002 | 0.0662 | n/a | n/a | 0.7117 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.0662 | n/a | n/a | 0.7117 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0115 | 0.0111 | 0.0101 | 0.0122 | 0.0008 | 0.0662 | n/a | n/a | 0.7117 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0875 | 0.0897 | 0.0860 | 0.0995 | 0.0051 | 0.0866 | n/a | n/a | 1.4800 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0076 | 0.0075 | 0.0072 | 0.0077 | 0.0002 | 0.0866 | n/a | n/a | 1.4800 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1295 | 0.1280 | 0.1204 | 0.1389 | 0.0069 | 0.0866 | n/a | n/a | 1.4800 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0173 | 0.0174 | 0.0170 | 0.0179 | 0.0003 | 0.0864 | n/a | n/a | 1.0773 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.0864 | n/a | n/a | 1.0773 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0186 | 0.0191 | 0.0164 | 0.0223 | 0.0026 | 0.0864 | n/a | n/a | 1.0773 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0179 | 0.0191 | 0.0176 | 0.0213 | 0.0017 | 0.0861 | n/a | n/a | 1.1306 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.0861 | n/a | n/a | 1.1306 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0203 | 0.0195 | 0.0176 | 0.0203 | 0.0011 | 0.0861 | n/a | n/a | 1.1306 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0177 | 0.0178 | 0.0169 | 0.0192 | 0.0008 | 0.0936 | n/a | n/a | 1.0693 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.0936 | n/a | n/a | 1.0693 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0189 | 0.0197 | 0.0150 | 0.0250 | 0.0041 | 0.0936 | n/a | n/a | 1.0693 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0172 | 0.0173 | 0.0171 | 0.0174 | 0.0001 | 0.1049 | n/a | n/a | 0.9096 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0018 | 0.0018 | 0.0016 | 0.0020 | 0.0002 | 0.1049 | n/a | n/a | 0.9096 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0157 | 0.0148 | 0.0114 | 0.0164 | 0.0019 | 0.1049 | n/a | n/a | 0.9096 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.1057 | 0.1047 | 0.0922 | 0.1179 | 0.0088 | 0.0953 | n/a | n/a | 1.3363 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0101 | 0.0101 | 0.0098 | 0.0103 | 0.0002 | 0.0953 | n/a | n/a | 1.3363 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1413 | 0.1400 | 0.1302 | 0.1531 | 0.0084 | 0.0953 | n/a | n/a | 1.3363 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0224 | 0.0225 | 0.0212 | 0.0248 | 0.0013 | 0.1815 | n/a | n/a | 0.8039 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0041 | 0.0042 | 0.0038 | 0.0049 | 0.0004 | 0.1815 | n/a | n/a | 0.8039 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0180 | 0.0195 | 0.0150 | 0.0263 | 0.0045 | 0.1815 | n/a | n/a | 0.8039 | no | n/a | n/a | no | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0212 | 0.0218 | 0.0210 | 0.0241 | 0.0012 | 0.1779 | n/a | n/a | 1.1192 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.1779 | n/a | n/a | 1.1192 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0238 | 0.0229 | 0.0161 | 0.0283 | 0.0040 | 0.1779 | n/a | n/a | 1.1192 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0215 | 0.0220 | 0.0210 | 0.0243 | 0.0012 | 0.1852 | n/a | n/a | 0.9464 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0040 | 0.0041 | 0.0038 | 0.0045 | 0.0002 | 0.1852 | n/a | n/a | 0.9464 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0203 | 0.0199 | 0.0133 | 0.0263 | 0.0054 | 0.1852 | n/a | n/a | 0.9464 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0217 | 0.0221 | 0.0209 | 0.0234 | 0.0009 | 0.2301 | n/a | n/a | 0.6575 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0050 | 0.0049 | 0.0046 | 0.0054 | 0.0003 | 0.2301 | n/a | n/a | 0.6575 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0143 | 0.0153 | 0.0138 | 0.0200 | 0.0024 | 0.2301 | n/a | n/a | 0.6575 | no | n/a | n/a | no | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.1188 | 0.1189 | 0.1048 | 0.1358 | 0.0122 | 0.3060 | n/a | n/a | 1.6244 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0364 | 0.0341 | 0.0229 | 0.0400 | 0.0064 | 0.3060 | n/a | n/a | 1.6244 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1930 | 0.2638 | 0.1432 | 0.5748 | 0.1592 | 0.3060 | n/a | n/a | 1.6244 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0151 | 0.0151 | 0.0146 | 0.0158 | 0.0005 | 0.1154 | n/a | n/a | 2.9763 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.1154 | n/a | n/a | 2.9763 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0449 | 0.0454 | 0.0447 | 0.0468 | 0.0008 | 0.1154 | n/a | n/a | 2.9763 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0150 | 0.0001 | 0.2956 | n/a | n/a | 5.0984 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0001 | 0.2956 | n/a | n/a | 5.0984 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0758 | 0.0756 | 0.0738 | 0.0768 | 0.0011 | 0.2956 | n/a | n/a | 5.0984 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0152 | 0.0001 | 0.0816 | n/a | n/a | 1.9919 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0816 | n/a | n/a | 1.9919 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0300 | 0.0307 | 0.0297 | 0.0327 | 0.0011 | 0.0816 | n/a | n/a | 1.9919 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0152 | 0.0001 | 0.0817 | n/a | n/a | 2.0346 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0817 | n/a | n/a | 2.0346 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0302 | 0.0306 | 0.0299 | 0.0315 | 0.0006 | 0.0817 | n/a | n/a | 2.0346 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0150 | 0.0151 | 0.0148 | 0.0153 | 0.0002 | 0.0863 | n/a | n/a | 1.9866 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0016 | 0.0012 | 0.0028 | 0.0006 | 0.0863 | n/a | n/a | 1.9866 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0298 | 0.0305 | 0.0295 | 0.0317 | 0.0009 | 0.0863 | n/a | n/a | 1.9866 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0151 | 0.0151 | 0.0150 | 0.0155 | 0.0002 | 0.1135 | n/a | n/a | 3.0652 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 0.1135 | n/a | n/a | 3.0652 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0462 | 0.0470 | 0.0448 | 0.0515 | 0.0024 | 0.1135 | n/a | n/a | 3.0652 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0149 | 0.0150 | 0.0149 | 0.0155 | 0.0002 | 0.3030 | n/a | n/a | 5.0278 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0045 | 0.0045 | 0.0044 | 0.0046 | 0.0000 | 0.3030 | n/a | n/a | 5.0278 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0752 | 0.0750 | 0.0740 | 0.0758 | 0.0007 | 0.3030 | n/a | n/a | 5.0278 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0151 | 0.0153 | 0.0151 | 0.0158 | 0.0003 | 0.0814 | n/a | n/a | 2.0133 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.0814 | n/a | n/a | 2.0133 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0305 | 0.0304 | 0.0299 | 0.0310 | 0.0004 | 0.0814 | n/a | n/a | 2.0133 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0151 | 0.0001 | 0.0853 | n/a | n/a | 2.0180 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.0853 | n/a | n/a | 2.0180 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0302 | 0.0306 | 0.0298 | 0.0326 | 0.0010 | 0.0853 | n/a | n/a | 2.0180 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0150 | 0.0150 | 0.0149 | 0.0151 | 0.0001 | 0.0862 | n/a | n/a | 2.0175 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.0862 | n/a | n/a | 2.0175 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0303 | 0.0303 | 0.0302 | 0.0306 | 0.0001 | 0.0862 | n/a | n/a | 2.0175 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0148 | 0.0148 | 0.0147 | 0.0150 | 0.0001 | 0.1169 | n/a | n/a | 3.0356 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 0.1169 | n/a | n/a | 3.0356 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0450 | 0.0453 | 0.0442 | 0.0463 | 0.0008 | 0.1169 | n/a | n/a | 3.0356 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0155 | 0.0156 | 0.0150 | 0.0165 | 0.0005 | 0.2860 | n/a | n/a | 4.9227 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0044 | 0.0044 | 0.0044 | 0.0045 | 0.0000 | 0.2860 | n/a | n/a | 4.9227 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0761 | 0.0761 | 0.0734 | 0.0788 | 0.0019 | 0.2860 | n/a | n/a | 4.9227 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0152 | 0.0153 | 0.0152 | 0.0153 | 0.0001 | 0.0840 | n/a | n/a | 2.0028 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.0840 | n/a | n/a | 2.0028 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0305 | 0.0310 | 0.0302 | 0.0323 | 0.0008 | 0.0840 | n/a | n/a | 2.0028 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0149 | 0.0150 | 0.0149 | 0.0152 | 0.0001 | 0.0812 | n/a | n/a | 2.0337 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.0812 | n/a | n/a | 2.0337 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0304 | 0.0305 | 0.0303 | 0.0310 | 0.0003 | 0.0812 | n/a | n/a | 2.0337 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0152 | 0.0151 | 0.0150 | 0.0153 | 0.0001 | 0.0865 | n/a | n/a | 1.9854 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.0865 | n/a | n/a | 1.9854 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0301 | 0.0300 | 0.0296 | 0.0302 | 0.0002 | 0.0865 | n/a | n/a | 1.9854 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0149 | 0.0149 | 0.0148 | 0.0150 | 0.0001 | 0.1194 | n/a | n/a | 3.0002 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0018 | 0.0019 | 0.0017 | 0.0022 | 0.0002 | 0.1194 | n/a | n/a | 3.0002 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0448 | 0.0450 | 0.0444 | 0.0460 | 0.0005 | 0.1194 | n/a | n/a | 3.0002 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0158 | 0.0160 | 0.0155 | 0.0168 | 0.0005 | 0.2975 | n/a | n/a | 4.8611 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0047 | 0.0047 | 0.0046 | 0.0048 | 0.0001 | 0.2975 | n/a | n/a | 4.8611 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0766 | 0.0760 | 0.0737 | 0.0776 | 0.0015 | 0.2975 | n/a | n/a | 4.8611 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0165 | 0.0167 | 0.0164 | 0.0173 | 0.0003 | 0.0914 | n/a | n/a | 1.8887 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0017 | 0.0001 | 0.0914 | n/a | n/a | 1.8887 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0312 | 0.0312 | 0.0300 | 0.0329 | 0.0010 | 0.0914 | n/a | n/a | 1.8887 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0158 | 0.0159 | 0.0154 | 0.0164 | 0.0003 | 0.0853 | n/a | n/a | 1.9763 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.0853 | n/a | n/a | 1.9763 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0313 | 0.0311 | 0.0302 | 0.0318 | 0.0005 | 0.0853 | n/a | n/a | 1.9763 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0161 | 0.0002 | 0.0945 | n/a | n/a | 1.9343 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0018 | 0.0001 | 0.0945 | n/a | n/a | 1.9343 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0306 | 0.0310 | 0.0305 | 0.0327 | 0.0009 | 0.0945 | n/a | n/a | 1.9343 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0157 | 0.0156 | 0.0152 | 0.0158 | 0.0003 | 0.1486 | n/a | n/a | 3.7670 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.1486 | n/a | n/a | 3.7670 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0593 | 0.0602 | 0.0588 | 0.0645 | 0.0021 | 0.1486 | n/a | n/a | 3.7670 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0201 | 0.0204 | 0.0196 | 0.0212 | 0.0006 | 0.3790 | n/a | n/a | 4.9521 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0076 | 0.0076 | 0.0072 | 0.0082 | 0.0003 | 0.3790 | n/a | n/a | 4.9521 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.0997 | 0.1031 | 0.0925 | 0.1131 | 0.0080 | 0.3790 | n/a | n/a | 4.9521 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0262 | 0.0262 | 0.0259 | 0.0267 | 0.0003 | 0.1435 | n/a | n/a | 1.2173 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0038 | 0.0038 | 0.0036 | 0.0040 | 0.0001 | 0.1435 | n/a | n/a | 1.2173 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0319 | 0.0321 | 0.0317 | 0.0330 | 0.0005 | 0.1435 | n/a | n/a | 1.2173 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0203 | 0.0204 | 0.0197 | 0.0214 | 0.0006 | 0.1269 | n/a | n/a | 1.5511 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0025 | 0.0027 | 0.0001 | 0.1269 | n/a | n/a | 1.5511 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0314 | 0.0319 | 0.0304 | 0.0347 | 0.0015 | 0.1269 | n/a | n/a | 1.5511 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0206 | 0.0208 | 0.0205 | 0.0215 | 0.0004 | 0.1452 | n/a | n/a | 1.4968 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0030 | 0.0031 | 0.0030 | 0.0033 | 0.0001 | 0.1452 | n/a | n/a | 1.4968 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0308 | 0.0313 | 0.0307 | 0.0320 | 0.0006 | 0.1452 | n/a | n/a | 1.4968 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0188 | 0.0185 | 0.0167 | 0.0203 | 0.0014 | 0.2069 | n/a | n/a | 3.9500 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0039 | 0.0042 | 0.0039 | 0.0053 | 0.0006 | 0.2069 | n/a | n/a | 3.9500 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0743 | 0.0753 | 0.0619 | 0.0888 | 0.0093 | 0.2069 | n/a | n/a | 3.9500 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0346 | 0.0361 | 0.0339 | 0.0405 | 0.0025 | 0.3469 | n/a | n/a | 3.2598 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0120 | 0.0120 | 0.0117 | 0.0122 | 0.0002 | 0.3469 | n/a | n/a | 3.2598 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1127 | 0.1101 | 0.1018 | 0.1167 | 0.0055 | 0.3469 | n/a | n/a | 3.2598 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0637 | 0.0652 | 0.0623 | 0.0683 | 0.0026 | 0.1835 | n/a | n/a | 0.5505 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0117 | 0.0121 | 0.0103 | 0.0154 | 0.0017 | 0.1835 | n/a | n/a | 0.5505 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0351 | 0.0351 | 0.0343 | 0.0363 | 0.0008 | 0.1835 | n/a | n/a | 0.5505 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0330 | 0.0326 | 0.0312 | 0.0342 | 0.0012 | 0.1949 | n/a | n/a | 1.0498 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0064 | 0.0068 | 0.0063 | 0.0079 | 0.0006 | 0.1949 | n/a | n/a | 1.0498 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0347 | 0.0351 | 0.0344 | 0.0361 | 0.0007 | 0.1949 | n/a | n/a | 1.0498 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0375 | 0.0374 | 0.0370 | 0.0377 | 0.0003 | 0.1958 | n/a | n/a | 0.9606 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0073 | 0.0073 | 0.0072 | 0.0074 | 0.0001 | 0.1958 | n/a | n/a | 0.9606 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0360 | 0.0360 | 0.0354 | 0.0366 | 0.0005 | 0.1958 | n/a | n/a | 0.9606 | no | n/a | n/a | no | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0234 | 0.0239 | 0.0227 | 0.0258 | 0.0011 | 0.3740 | n/a | n/a | 4.0446 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0087 | 0.0090 | 0.0087 | 0.0103 | 0.0006 | 0.3740 | n/a | n/a | 4.0446 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0945 | 0.0940 | 0.0797 | 0.1053 | 0.0084 | 0.3740 | n/a | n/a | 4.0446 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.0926 | 0.0930 | 0.0901 | 0.0969 | 0.0024 | 0.3705 | n/a | n/a | 1.3238 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0343 | 0.0349 | 0.0329 | 0.0397 | 0.0025 | 0.3705 | n/a | n/a | 1.3238 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1226 | 0.1251 | 0.1130 | 0.1449 | 0.0121 | 0.3705 | n/a | n/a | 1.3238 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2135 | 0.2145 | 0.2131 | 0.2186 | 0.0021 | 0.1912 | n/a | n/a | 0.1823 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0408 | 0.0405 | 0.0398 | 0.0410 | 0.0005 | 0.1912 | n/a | n/a | 0.1823 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0389 | 0.0389 | 0.0375 | 0.0401 | 0.0009 | 0.1912 | n/a | n/a | 0.1823 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.0806 | 0.0815 | 0.0775 | 0.0876 | 0.0033 | 0.2410 | n/a | n/a | 0.4979 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0194 | 0.0208 | 0.0192 | 0.0234 | 0.0018 | 0.2410 | n/a | n/a | 0.4979 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0401 | 0.0408 | 0.0394 | 0.0426 | 0.0013 | 0.2410 | n/a | n/a | 0.4979 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1035 | 0.1044 | 0.1019 | 0.1101 | 0.0030 | 0.2480 | n/a | n/a | 0.4550 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0257 | 0.0258 | 0.0253 | 0.0262 | 0.0004 | 0.2480 | n/a | n/a | 0.4550 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0471 | 0.0485 | 0.0430 | 0.0568 | 0.0048 | 0.2480 | n/a | n/a | 0.4550 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0157 | 0.0159 | 0.0155 | 0.0167 | 0.0005 | 0.1442 | n/a | n/a | 0.7830 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0027 | 0.0022 | 0.0042 | 0.0008 | 0.1442 | n/a | n/a | 0.7830 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0123 | 0.0124 | 0.0122 | 0.0127 | 0.0002 | 0.1442 | n/a | n/a | 0.7830 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0157 | 0.0159 | 0.0155 | 0.0164 | 0.0004 | 0.5751 | n/a | n/a | 0.7418 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0091 | 0.0098 | 0.0080 | 0.0116 | 0.0015 | 0.5751 | n/a | n/a | 0.7418 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0117 | 0.0120 | 0.0116 | 0.0132 | 0.0006 | 0.5751 | n/a | n/a | 0.7418 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0155 | 0.0156 | 0.0154 | 0.0161 | 0.0003 | 0.1530 | n/a | n/a | 0.8472 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0024 | 0.0024 | 0.0022 | 0.0025 | 0.0001 | 0.1530 | n/a | n/a | 0.8472 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0131 | 0.0131 | 0.0129 | 0.0134 | 0.0002 | 0.1530 | n/a | n/a | 0.8472 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0159 | 0.0163 | 0.0156 | 0.0174 | 0.0006 | 0.5086 | n/a | n/a | 0.7543 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0081 | 0.0081 | 0.0079 | 0.0084 | 0.0001 | 0.5086 | n/a | n/a | 0.7543 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0122 | 0.0118 | 0.0128 | 0.0004 | 0.5086 | n/a | n/a | 0.7543 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0160 | 0.0159 | 0.0157 | 0.0161 | 0.0002 | 0.1458 | n/a | n/a | 0.8329 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.1458 | n/a | n/a | 0.8329 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0133 | 0.0132 | 0.0128 | 0.0137 | 0.0004 | 0.1458 | n/a | n/a | 0.8329 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0158 | 0.0159 | 0.0157 | 0.0163 | 0.0002 | 0.5215 | n/a | n/a | 0.7968 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0082 | 0.0082 | 0.0080 | 0.0085 | 0.0002 | 0.5215 | n/a | n/a | 0.7968 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0126 | 0.0128 | 0.0125 | 0.0135 | 0.0004 | 0.5215 | n/a | n/a | 0.7968 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0158 | 0.0160 | 0.0156 | 0.0167 | 0.0004 | 0.1458 | n/a | n/a | 0.9056 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0023 | 0.0024 | 0.0022 | 0.0027 | 0.0002 | 0.1458 | n/a | n/a | 0.9056 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0143 | 0.0142 | 0.0137 | 0.0148 | 0.0004 | 0.1458 | n/a | n/a | 0.9056 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0156 | 0.0159 | 0.0155 | 0.0170 | 0.0005 | 0.5124 | n/a | n/a | 0.8789 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0080 | 0.0084 | 0.0079 | 0.0097 | 0.0007 | 0.5124 | n/a | n/a | 0.8789 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0137 | 0.0138 | 0.0134 | 0.0145 | 0.0004 | 0.5124 | n/a | n/a | 0.8789 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0167 | 0.0168 | 0.0164 | 0.0175 | 0.0004 | 0.1693 | n/a | n/a | 1.2890 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0028 | 0.0028 | 0.0026 | 0.0032 | 0.0002 | 0.1693 | n/a | n/a | 1.2890 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0215 | 0.0231 | 0.0194 | 0.0280 | 0.0035 | 0.1693 | n/a | n/a | 1.2890 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0166 | 0.0167 | 0.0164 | 0.0169 | 0.0002 | 0.5214 | n/a | n/a | 0.9997 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0086 | 0.0086 | 0.0083 | 0.0088 | 0.0002 | 0.5214 | n/a | n/a | 0.9997 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0166 | 0.0170 | 0.0131 | 0.0213 | 0.0027 | 0.5214 | n/a | n/a | 0.9997 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0198 | 0.0199 | 0.0194 | 0.0204 | 0.0003 | 0.1856 | n/a | n/a | 0.9947 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0037 | 0.0037 | 0.0035 | 0.0040 | 0.0002 | 0.1856 | n/a | n/a | 0.9947 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0197 | 0.0201 | 0.0179 | 0.0227 | 0.0019 | 0.1856 | n/a | n/a | 0.9947 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0205 | 0.0209 | 0.0195 | 0.0234 | 0.0013 | 0.4684 | n/a | n/a | 1.0248 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0096 | 0.0101 | 0.0095 | 0.0121 | 0.0010 | 0.4684 | n/a | n/a | 1.0248 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0210 | 0.0208 | 0.0176 | 0.0253 | 0.0029 | 0.4684 | n/a | n/a | 1.0248 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0335 | 0.0335 | 0.0324 | 0.0347 | 0.0009 | 0.2020 | n/a | n/a | 0.7547 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0068 | 0.0067 | 0.0066 | 0.0068 | 0.0001 | 0.2020 | n/a | n/a | 0.7547 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0253 | 0.0294 | 0.0239 | 0.0400 | 0.0064 | 0.2020 | n/a | n/a | 0.7547 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0344 | 0.0358 | 0.0328 | 0.0416 | 0.0031 | 0.3819 | n/a | n/a | 0.7749 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0132 | 0.0135 | 0.0129 | 0.0148 | 0.0007 | 0.3819 | n/a | n/a | 0.7749 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0267 | 0.0263 | 0.0218 | 0.0303 | 0.0035 | 0.3819 | n/a | n/a | 0.7749 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0161 | 0.0162 | 0.0160 | 0.0166 | 0.0002 | 0.1714 | n/a | n/a | 0.8417 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0028 | 0.0028 | 0.0027 | 0.0029 | 0.0001 | 0.1714 | n/a | n/a | 0.8417 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0136 | 0.0134 | 0.0125 | 0.0141 | 0.0006 | 0.1714 | n/a | n/a | 0.8417 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0105 | 0.0105 | 0.0101 | 0.0110 | 0.0003 | 1.0162 | n/a | n/a | 1.1529 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0106 | 0.0108 | 0.0101 | 0.0120 | 0.0007 | 1.0162 | n/a | n/a | 1.1529 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0121 | 0.0121 | 0.0117 | 0.0126 | 0.0003 | 1.0162 | n/a | n/a | 1.1529 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0190 | 0.0190 | 0.0189 | 0.0192 | 0.0001 | 0.2231 | n/a | n/a | 1.1642 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0044 | 0.0042 | 0.0052 | 0.0004 | 0.2231 | n/a | n/a | 1.1642 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0221 | 0.0216 | 0.0179 | 0.0236 | 0.0020 | 0.2231 | n/a | n/a | 1.1642 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0135 | 0.0141 | 0.0130 | 0.0169 | 0.0014 | 0.8762 | n/a | n/a | 1.3580 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0118 | 0.0120 | 0.0114 | 0.0130 | 0.0005 | 0.8762 | n/a | n/a | 1.3580 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0184 | 0.0193 | 0.0166 | 0.0235 | 0.0026 | 0.8762 | n/a | n/a | 1.3580 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0304 | 0.0303 | 0.0293 | 0.0309 | 0.0006 | 0.2638 | n/a | n/a | 0.8339 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0080 | 0.0084 | 0.0079 | 0.0099 | 0.0008 | 0.2638 | n/a | n/a | 0.8339 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0253 | 0.0286 | 0.0232 | 0.0376 | 0.0054 | 0.2638 | n/a | n/a | 0.8339 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0242 | 0.0238 | 0.0231 | 0.0243 | 0.0005 | 0.6597 | n/a | n/a | 0.8398 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0159 | 0.0165 | 0.0157 | 0.0190 | 0.0012 | 0.6597 | n/a | n/a | 0.8398 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0203 | 0.0215 | 0.0160 | 0.0330 | 0.0062 | 0.6597 | n/a | n/a | 0.8398 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0790 | 0.0794 | 0.0787 | 0.0810 | 0.0008 | 0.2844 | n/a | n/a | 0.8837 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0225 | 0.0225 | 0.0220 | 0.0230 | 0.0003 | 0.2844 | n/a | n/a | 0.8837 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0698 | 0.0727 | 0.0675 | 0.0820 | 0.0055 | 0.2844 | n/a | n/a | 0.8837 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0611 | 0.0612 | 0.0584 | 0.0639 | 0.0017 | 0.5159 | n/a | n/a | 0.4471 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0315 | 0.0323 | 0.0233 | 0.0427 | 0.0077 | 0.5159 | n/a | n/a | 0.4471 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0273 | 0.0283 | 0.0186 | 0.0438 | 0.0085 | 0.5159 | n/a | n/a | 0.4471 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0184 | 0.0193 | 0.0181 | 0.0226 | 0.0017 | 0.1053 | n/a | n/a | 0.7832 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0020 | 0.0019 | 0.0022 | 0.0001 | 0.1053 | n/a | n/a | 0.7832 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0144 | 0.0153 | 0.0099 | 0.0270 | 0.0061 | 0.1053 | n/a | n/a | 0.7832 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0970 | 0.1074 | 0.0948 | 0.1474 | 0.0202 | 0.1348 | n/a | n/a | 0.3200 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0131 | 0.0133 | 0.0114 | 0.0160 | 0.0015 | 0.1348 | n/a | n/a | 0.3200 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0310 | 0.0313 | 0.0220 | 0.0378 | 0.0057 | 0.1348 | n/a | n/a | 0.3200 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.3957 | 0.3965 | 0.3798 | 0.4179 | 0.0137 | 0.9076 | n/a | n/a | 0.2945 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3591 | 0.3651 | 0.3213 | 0.3975 | 0.0273 | 0.9076 | n/a | n/a | 0.2945 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1165 | 0.1378 | 0.1029 | 0.2331 | 0.0487 | 0.9076 | n/a | n/a | 0.2945 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.1900 | 2.2223 | 2.0101 | 2.4283 | 0.1504 | 0.2000 | n/a | n/a | 0.1192 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4380 | 0.4569 | 0.4316 | 0.5065 | 0.0297 | 0.2000 | n/a | n/a | 0.1192 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2610 | 0.2567 | 0.2377 | 0.2668 | 0.0108 | 0.2000 | n/a | n/a | 0.1192 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1995 | 0.2075 | 0.1859 | 0.2344 | 0.0188 | 6.2118 | n/a | n/a | 0.5443 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2390 | 1.2430 | 1.2333 | 1.2592 | 0.0090 | 6.2118 | n/a | n/a | 0.5443 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1086 | 0.1134 | 0.1025 | 0.1430 | 0.0151 | 6.2118 | n/a | n/a | 0.5443 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 5.6637 | 5.6840 | 5.5839 | 5.8084 | 0.0751 | 12.2538 | n/a | n/a | 0.0651 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 69.4021 | 68.9603 | 62.8922 | 76.2510 | 4.9158 | 12.2538 | n/a | n/a | 0.0651 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3686 | 0.3672 | 0.3302 | 0.3899 | 0.0206 | 12.2538 | n/a | n/a | 0.0651 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0286 | 0.0284 | 0.0275 | 0.0290 | 0.0005 | 5.8921 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1684 | 0.2084 | 0.1658 | 0.2778 | 0.0510 | 5.8921 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0286 | 0.0289 | 0.0279 | 0.0304 | 0.0008 | 19.4803 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5569 | 0.5642 | 0.5545 | 0.5938 | 0.0149 | 19.4803 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5507 | 0.5455 | 0.5272 | 0.5591 | 0.0123 | 14.9279 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.2212 | 8.3594 | 8.0018 | 9.1672 | 0.4250 | 14.9279 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.6059 | 0.6015 | 0.5767 | 0.6183 | 0.0142 | 46.3953 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 28.1123 | 28.2268 | 26.7658 | 29.6040 | 1.2100 | 46.3953 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.1200 | 0.1216 | 0.1155 | 0.1327 | 0.0058 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.2975 | 0.2990 | 0.2930 | 0.3050 | 0.0043 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2461 | 0.2499 | 0.2437 | 0.2666 | 0.0084 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.5992 | 0.6056 | 0.5956 | 0.6330 | 0.0140 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1602 | 1.1805 | 1.1468 | 1.2729 | 0.0468 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.7758 | 2.7763 | 2.7325 | 2.8289 | 0.0349 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 5.2182 | 5.2834 | 5.0684 | 5.6278 | 0.2017 | 0.0307 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1601 | 0.2081 | 0.1590 | 0.4017 | 0.0968 | 0.0307 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
