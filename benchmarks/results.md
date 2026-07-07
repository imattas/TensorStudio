# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.5.0`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `21`
- TensorStudio losses versus NumPy: `82`
- TensorStudio wins versus JAX CPU dispatch: `83`
- TensorStudio losses versus JAX CPU dispatch: `15`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0031 | 0.0030 | 0.0027 | 0.0034 | 0.0003 | 0.5045 | n/a | n/a | 3.9371 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.5045 | n/a | n/a | 3.9371 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0121 | 0.0120 | 0.0115 | 0.0123 | 0.0003 | 0.5045 | n/a | n/a | 3.9371 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0027 | 0.0027 | 0.0025 | 0.0029 | 0.0001 | 0.2684 | n/a | n/a | 4.4236 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0001 | 0.2684 | n/a | n/a | 4.4236 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0117 | 0.0122 | 0.0115 | 0.0144 | 0.0011 | 0.2684 | n/a | n/a | 4.4236 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0027 | 0.0028 | 0.0026 | 0.0033 | 0.0003 | 0.2544 | n/a | n/a | 4.4158 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.2544 | n/a | n/a | 4.4158 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0118 | 0.0118 | 0.0118 | 0.0119 | 0.0001 | 0.2544 | n/a | n/a | 4.4158 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0031 | 0.0031 | 0.0026 | 0.0035 | 0.0003 | 0.2271 | n/a | n/a | 3.1559 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.2271 | n/a | n/a | 3.1559 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0098 | 0.0098 | 0.0087 | 0.0114 | 0.0010 | 0.2271 | n/a | n/a | 3.1559 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0104 | 0.0104 | 0.0100 | 0.0108 | 0.0003 | 0.5418 | n/a | n/a | 8.6896 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0056 | 0.0056 | 0.0055 | 0.0059 | 0.0002 | 0.5418 | n/a | n/a | 8.6896 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0902 | 0.0920 | 0.0875 | 0.1022 | 0.0052 | 0.5418 | n/a | n/a | 8.6896 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0027 | 0.0027 | 0.0027 | 0.0029 | 0.0001 | 0.2908 | n/a | n/a | 4.4306 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2908 | n/a | n/a | 4.4306 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0120 | 0.0124 | 0.0114 | 0.0144 | 0.0011 | 0.2908 | n/a | n/a | 4.4306 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.2700 | n/a | n/a | 4.4292 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2700 | n/a | n/a | 4.4292 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0120 | 0.0117 | 0.0125 | 0.0003 | 0.2700 | n/a | n/a | 4.4292 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0025 | 0.0027 | 0.0025 | 0.0031 | 0.0002 | 0.2880 | n/a | n/a | 4.7936 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0001 | 0.2880 | n/a | n/a | 4.7936 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0119 | 0.0125 | 0.0002 | 0.2880 | n/a | n/a | 4.7936 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0027 | 0.0028 | 0.0026 | 0.0031 | 0.0002 | 0.2629 | n/a | n/a | 3.3506 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2629 | n/a | n/a | 3.3506 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0091 | 0.0092 | 0.0089 | 0.0097 | 0.0003 | 0.2629 | n/a | n/a | 3.3506 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0100 | 0.0100 | 0.0098 | 0.0102 | 0.0002 | 0.5536 | n/a | n/a | 9.2234 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0055 | 0.0056 | 0.0053 | 0.0061 | 0.0003 | 0.5536 | n/a | n/a | 9.2234 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0923 | 0.0926 | 0.0898 | 0.0956 | 0.0024 | 0.5536 | n/a | n/a | 9.2234 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0026 | 0.0027 | 0.0025 | 0.0029 | 0.0002 | 0.2563 | n/a | n/a | 4.5519 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0006 | 0.0008 | 0.0001 | 0.2563 | n/a | n/a | 4.5519 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0120 | 0.0125 | 0.0115 | 0.0147 | 0.0011 | 0.2563 | n/a | n/a | 4.5519 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0026 | 0.0027 | 0.0026 | 0.0031 | 0.0002 | 0.2707 | n/a | n/a | 4.7729 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2707 | n/a | n/a | 4.7729 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0126 | 0.0127 | 0.0118 | 0.0144 | 0.0009 | 0.2707 | n/a | n/a | 4.7729 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0034 | 0.0036 | 0.0028 | 0.0045 | 0.0007 | 0.2260 | n/a | n/a | 3.5742 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0013 | 0.0002 | 0.2260 | n/a | n/a | 3.5742 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0122 | 0.0124 | 0.0121 | 0.0129 | 0.0004 | 0.2260 | n/a | n/a | 3.5742 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0026 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.2954 | n/a | n/a | 3.6594 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2954 | n/a | n/a | 3.6594 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0096 | 0.0095 | 0.0089 | 0.0101 | 0.0004 | 0.2954 | n/a | n/a | 3.6594 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0102 | 0.0102 | 0.0101 | 0.0104 | 0.0001 | 0.5655 | n/a | n/a | 9.0853 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0058 | 0.0066 | 0.0054 | 0.0101 | 0.0018 | 0.5655 | n/a | n/a | 9.0853 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0926 | 0.0925 | 0.0908 | 0.0947 | 0.0014 | 0.5655 | n/a | n/a | 9.0853 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0030 | 0.0001 | 0.2460 | n/a | n/a | 4.9523 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0012 | 0.0002 | 0.2460 | n/a | n/a | 4.9523 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0131 | 0.0129 | 0.0117 | 0.0138 | 0.0007 | 0.2460 | n/a | n/a | 4.9523 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0029 | 0.0029 | 0.0027 | 0.0030 | 0.0002 | 0.2338 | n/a | n/a | 4.2683 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2338 | n/a | n/a | 4.2683 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0118 | 0.0125 | 0.0002 | 0.2338 | n/a | n/a | 4.2683 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.2585 | n/a | n/a | 4.4932 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2585 | n/a | n/a | 4.4932 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0122 | 0.0124 | 0.0120 | 0.0131 | 0.0004 | 0.2585 | n/a | n/a | 4.4932 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0029 | 0.0029 | 0.0027 | 0.0033 | 0.0002 | 0.2493 | n/a | n/a | 3.5060 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2493 | n/a | n/a | 3.5060 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0102 | 0.0101 | 0.0097 | 0.0104 | 0.0003 | 0.2493 | n/a | n/a | 3.5060 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0105 | 0.0106 | 0.0102 | 0.0109 | 0.0003 | 0.5285 | n/a | n/a | 8.5947 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0056 | 0.0055 | 0.0054 | 0.0057 | 0.0001 | 0.5285 | n/a | n/a | 8.5947 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0904 | 0.0979 | 0.0895 | 0.1220 | 0.0124 | 0.5285 | n/a | n/a | 8.5947 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0033 | 0.0033 | 0.0033 | 0.0034 | 0.0000 | 0.2885 | n/a | n/a | 4.6777 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0011 | 0.0001 | 0.2885 | n/a | n/a | 4.6777 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0156 | 0.0157 | 0.0133 | 0.0183 | 0.0019 | 0.2885 | n/a | n/a | 4.6777 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0037 | 0.0036 | 0.0033 | 0.0039 | 0.0003 | 0.2782 | n/a | n/a | 4.0743 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0011 | 0.0000 | 0.2782 | n/a | n/a | 4.0743 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0150 | 0.0152 | 0.0132 | 0.0174 | 0.0014 | 0.2782 | n/a | n/a | 4.0743 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0030 | 0.0032 | 0.0030 | 0.0034 | 0.0002 | 0.3393 | n/a | n/a | 4.6246 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3393 | n/a | n/a | 4.6246 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0141 | 0.0142 | 0.0131 | 0.0159 | 0.0010 | 0.3393 | n/a | n/a | 4.6246 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0037 | 0.0035 | 0.0033 | 0.0038 | 0.0002 | 0.2937 | n/a | n/a | 3.2657 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.2937 | n/a | n/a | 3.2657 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0121 | 0.0123 | 0.0112 | 0.0133 | 0.0008 | 0.2937 | n/a | n/a | 3.2657 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0122 | 0.0122 | 0.0118 | 0.0126 | 0.0003 | 0.6247 | n/a | n/a | 10.3590 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0076 | 0.0076 | 0.0071 | 0.0082 | 0.0004 | 0.6247 | n/a | n/a | 10.3590 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1262 | 0.1265 | 0.1172 | 0.1372 | 0.0069 | 0.6247 | n/a | n/a | 10.3590 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0050 | 0.0053 | 0.0048 | 0.0065 | 0.0007 | 0.2966 | n/a | n/a | 3.0274 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.2966 | n/a | n/a | 3.0274 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0150 | 0.0147 | 0.0127 | 0.0156 | 0.0010 | 0.2966 | n/a | n/a | 3.0274 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0052 | 0.0052 | 0.0049 | 0.0054 | 0.0002 | 0.2734 | n/a | n/a | 2.8184 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0017 | 0.0001 | 0.2734 | n/a | n/a | 2.8184 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0147 | 0.0146 | 0.0127 | 0.0168 | 0.0013 | 0.2734 | n/a | n/a | 2.8184 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0050 | 0.0054 | 0.0047 | 0.0069 | 0.0008 | 0.3211 | n/a | n/a | 2.7789 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.3211 | n/a | n/a | 2.7789 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0140 | 0.0141 | 0.0130 | 0.0152 | 0.0009 | 0.3211 | n/a | n/a | 2.7789 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0058 | 0.0059 | 0.0056 | 0.0067 | 0.0004 | 0.3021 | n/a | n/a | 1.9096 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.3021 | n/a | n/a | 1.9096 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0110 | 0.0111 | 0.0100 | 0.0124 | 0.0008 | 0.3021 | n/a | n/a | 1.9096 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0202 | 0.0209 | 0.0197 | 0.0225 | 0.0012 | 0.7050 | n/a | n/a | 6.5396 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0143 | 0.0148 | 0.0103 | 0.0183 | 0.0030 | 0.7050 | n/a | n/a | 6.5396 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1323 | 0.1330 | 0.1211 | 0.1522 | 0.0116 | 0.7050 | n/a | n/a | 6.5396 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0114 | 0.0121 | 0.0113 | 0.0141 | 0.0011 | 0.3349 | n/a | n/a | 1.8802 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0036 | 0.0041 | 0.0002 | 0.3349 | n/a | n/a | 1.8802 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0215 | 0.0204 | 0.0150 | 0.0233 | 0.0031 | 0.3349 | n/a | n/a | 1.8802 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0102 | 0.0102 | 0.0101 | 0.0103 | 0.0001 | 0.3818 | n/a | n/a | 1.2742 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0042 | 0.0001 | 0.3818 | n/a | n/a | 1.2742 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0130 | 0.0177 | 0.0127 | 0.0309 | 0.0070 | 0.3818 | n/a | n/a | 1.2742 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0099 | 0.0100 | 0.0093 | 0.0110 | 0.0006 | 0.5403 | n/a | n/a | 1.6402 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0054 | 0.0054 | 0.0043 | 0.0073 | 0.0011 | 0.5403 | n/a | n/a | 1.6402 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0163 | 0.0167 | 0.0125 | 0.0232 | 0.0036 | 0.5403 | n/a | n/a | 1.6402 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0134 | 0.0135 | 0.0132 | 0.0143 | 0.0004 | 0.3177 | n/a | n/a | 1.2731 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0043 | 0.0043 | 0.0042 | 0.0046 | 0.0002 | 0.3177 | n/a | n/a | 1.2731 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0171 | 0.0173 | 0.0170 | 0.0180 | 0.0004 | 0.3177 | n/a | n/a | 1.2731 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0456 | 0.0467 | 0.0445 | 0.0507 | 0.0023 | 0.4591 | n/a | n/a | 3.6702 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0209 | 0.0211 | 0.0208 | 0.0216 | 0.0003 | 0.4591 | n/a | n/a | 3.6702 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1674 | 0.1661 | 0.1535 | 0.1778 | 0.0088 | 0.4591 | n/a | n/a | 3.6702 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0021 | 0.0024 | 0.0020 | 0.0029 | 0.0004 | 0.9530 | n/a | n/a | 22.9541 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0020 | 0.0020 | 0.0018 | 0.0023 | 0.0002 | 0.9530 | n/a | n/a | 22.9541 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0475 | 0.0519 | 0.0439 | 0.0742 | 0.0113 | 0.9530 | n/a | n/a | 22.9541 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0022 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 2.7308 | n/a | n/a | 35.5319 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0059 | 0.0061 | 0.0046 | 0.0080 | 0.0011 | 2.7308 | n/a | n/a | 35.5319 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0768 | 0.0763 | 0.0719 | 0.0791 | 0.0026 | 2.7308 | n/a | n/a | 35.5319 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 0.6640 | n/a | n/a | 16.4363 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0002 | 0.6640 | n/a | n/a | 16.4363 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0302 | 0.0301 | 0.0293 | 0.0309 | 0.0006 | 0.6640 | n/a | n/a | 16.4363 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 0.6797 | n/a | n/a | 18.1556 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0017 | 0.0002 | 0.6797 | n/a | n/a | 18.1556 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0333 | 0.0360 | 0.0293 | 0.0517 | 0.0080 | 0.6797 | n/a | n/a | 18.1556 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0029 | 0.0031 | 0.0026 | 0.0037 | 0.0004 | 0.8290 | n/a | n/a | 13.4052 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0024 | 0.0022 | 0.0015 | 0.0026 | 0.0005 | 0.8290 | n/a | n/a | 13.4052 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0393 | 0.0414 | 0.0327 | 0.0515 | 0.0068 | 0.8290 | n/a | n/a | 13.4052 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0024 | 0.0026 | 0.0021 | 0.0037 | 0.0005 | 0.9011 | n/a | n/a | 18.7401 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0022 | 0.0024 | 0.0019 | 0.0031 | 0.0004 | 0.9011 | n/a | n/a | 18.7401 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0449 | 0.0485 | 0.0447 | 0.0592 | 0.0056 | 0.9011 | n/a | n/a | 18.7401 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0020 | 0.0020 | 0.0018 | 0.0021 | 0.0001 | 2.2891 | n/a | n/a | 37.0896 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0045 | 0.0046 | 0.0045 | 0.0047 | 0.0001 | 2.2891 | n/a | n/a | 37.0896 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0731 | 0.0731 | 0.0723 | 0.0741 | 0.0006 | 2.2891 | n/a | n/a | 37.0896 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0020 | 0.0020 | 0.0019 | 0.0023 | 0.0002 | 0.6112 | n/a | n/a | 14.6714 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0002 | 0.6112 | n/a | n/a | 14.6714 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0296 | 0.0298 | 0.0295 | 0.0305 | 0.0004 | 0.6112 | n/a | n/a | 14.6714 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0020 | 0.0020 | 0.0019 | 0.0021 | 0.0001 | 0.6296 | n/a | n/a | 15.0804 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0001 | 0.6296 | n/a | n/a | 15.0804 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0296 | 0.0296 | 0.0295 | 0.0298 | 0.0001 | 0.6296 | n/a | n/a | 15.0804 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0019 | 0.0020 | 0.0019 | 0.0021 | 0.0001 | 0.7185 | n/a | n/a | 16.1285 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 0.7185 | n/a | n/a | 16.1285 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0304 | 0.0305 | 0.0298 | 0.0313 | 0.0005 | 0.7185 | n/a | n/a | 16.1285 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0022 | 0.0002 | 0.9837 | n/a | n/a | 25.0611 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.9837 | n/a | n/a | 25.0611 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0444 | 0.0443 | 0.0437 | 0.0447 | 0.0004 | 0.9837 | n/a | n/a | 25.0611 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0022 | 0.0001 | 2.1943 | n/a | n/a | 35.7060 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0045 | 0.0046 | 0.0044 | 0.0052 | 0.0003 | 2.1943 | n/a | n/a | 35.7060 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0732 | 0.0740 | 0.0723 | 0.0769 | 0.0016 | 2.1943 | n/a | n/a | 35.7060 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0021 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.6255 | n/a | n/a | 14.3488 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 0.6255 | n/a | n/a | 14.3488 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0303 | 0.0308 | 0.0301 | 0.0322 | 0.0008 | 0.6255 | n/a | n/a | 14.3488 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0022 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 0.6117 | n/a | n/a | 17.3060 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0012 | 0.0017 | 0.0002 | 0.6117 | n/a | n/a | 17.3060 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0374 | 0.0402 | 0.0304 | 0.0571 | 0.0094 | 0.6117 | n/a | n/a | 17.3060 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0023 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.7273 | n/a | n/a | 20.6170 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0017 | 0.0020 | 0.0015 | 0.0034 | 0.0007 | 0.7273 | n/a | n/a | 20.6170 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0472 | 0.0559 | 0.0330 | 0.0850 | 0.0208 | 0.7273 | n/a | n/a | 20.6170 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0025 | 0.0026 | 0.0024 | 0.0032 | 0.0003 | 2.0332 | n/a | n/a | 22.2553 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0050 | 0.0046 | 0.0025 | 0.0054 | 0.0011 | 2.0332 | n/a | n/a | 22.2553 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0550 | 0.0621 | 0.0458 | 0.0990 | 0.0195 | 2.0332 | n/a | n/a | 22.2553 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0029 | 0.0030 | 0.0028 | 0.0033 | 0.0002 | 1.6310 | n/a | n/a | 26.3039 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0047 | 0.0049 | 0.0044 | 0.0055 | 0.0005 | 1.6310 | n/a | n/a | 26.3039 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0762 | 0.0794 | 0.0741 | 0.0907 | 0.0062 | 1.6310 | n/a | n/a | 26.3039 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0031 | 0.0032 | 0.0030 | 0.0034 | 0.0002 | 0.4949 | n/a | n/a | 9.7386 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0020 | 0.0002 | 0.4949 | n/a | n/a | 9.7386 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0303 | 0.0304 | 0.0293 | 0.0323 | 0.0011 | 0.4949 | n/a | n/a | 9.7386 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0027 | 0.0028 | 0.0026 | 0.0031 | 0.0002 | 0.4925 | n/a | n/a | 11.4323 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 0.4925 | n/a | n/a | 11.4323 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0304 | 0.0303 | 0.0301 | 0.0306 | 0.0002 | 0.4925 | n/a | n/a | 11.4323 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0029 | 0.0029 | 0.0028 | 0.0030 | 0.0001 | 0.4893 | n/a | n/a | 10.6297 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0001 | 0.4893 | n/a | n/a | 10.6297 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0306 | 0.0327 | 0.0301 | 0.0407 | 0.0041 | 0.4893 | n/a | n/a | 10.6297 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0024 | 0.0025 | 0.0024 | 0.0025 | 0.0000 | 0.9334 | n/a | n/a | 25.9395 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.9334 | n/a | n/a | 25.9395 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0634 | 0.0638 | 0.0558 | 0.0713 | 0.0063 | 0.9334 | n/a | n/a | 25.9395 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0094 | 0.0094 | 0.0090 | 0.0102 | 0.0004 | 0.7371 | n/a | n/a | 11.2737 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0069 | 0.0072 | 0.0067 | 0.0087 | 0.0008 | 0.7371 | n/a | n/a | 11.2737 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1054 | 0.1156 | 0.0996 | 0.1563 | 0.0208 | 0.7371 | n/a | n/a | 11.2737 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0141 | 0.0140 | 0.0132 | 0.0144 | 0.0005 | 0.2871 | n/a | n/a | 2.1155 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0041 | 0.0042 | 0.0036 | 0.0055 | 0.0007 | 0.2871 | n/a | n/a | 2.1155 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0298 | 0.0298 | 0.0296 | 0.0301 | 0.0002 | 0.2871 | n/a | n/a | 2.1155 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0085 | 0.0085 | 0.0081 | 0.0091 | 0.0004 | 0.2984 | n/a | n/a | 3.8005 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0025 | 0.0026 | 0.0025 | 0.0030 | 0.0002 | 0.2984 | n/a | n/a | 3.8005 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0324 | 0.0341 | 0.0304 | 0.0413 | 0.0038 | 0.2984 | n/a | n/a | 3.8005 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0105 | 0.0111 | 0.0098 | 0.0132 | 0.0014 | 0.2972 | n/a | n/a | 3.2837 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0031 | 0.0034 | 0.0031 | 0.0042 | 0.0004 | 0.2972 | n/a | n/a | 3.2837 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0344 | 0.0351 | 0.0330 | 0.0401 | 0.0026 | 0.2972 | n/a | n/a | 3.2837 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0051 | 0.0058 | 0.0049 | 0.0080 | 0.0012 | 0.6973 | n/a | n/a | 12.5006 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0036 | 0.0038 | 0.0035 | 0.0045 | 0.0004 | 0.6973 | n/a | n/a | 12.5006 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0639 | 0.0694 | 0.0601 | 0.0925 | 0.0121 | 0.6973 | n/a | n/a | 12.5006 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0336 | 0.0335 | 0.0322 | 0.0349 | 0.0009 | 0.4139 | n/a | n/a | 3.1440 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0139 | 0.0139 | 0.0121 | 0.0152 | 0.0011 | 0.4139 | n/a | n/a | 3.1440 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1055 | 0.1040 | 0.0950 | 0.1129 | 0.0062 | 0.4139 | n/a | n/a | 3.1440 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0625 | 0.0636 | 0.0592 | 0.0713 | 0.0044 | 0.1645 | n/a | n/a | 0.5368 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0103 | 0.0105 | 0.0101 | 0.0112 | 0.0004 | 0.1645 | n/a | n/a | 0.5368 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0335 | 0.0345 | 0.0327 | 0.0396 | 0.0025 | 0.1645 | n/a | n/a | 0.5368 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0288 | 0.0292 | 0.0278 | 0.0308 | 0.0010 | 0.2008 | n/a | n/a | 1.1411 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0058 | 0.0060 | 0.0058 | 0.0063 | 0.0002 | 0.2008 | n/a | n/a | 1.1411 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0329 | 0.0331 | 0.0326 | 0.0338 | 0.0004 | 0.2008 | n/a | n/a | 1.1411 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0316 | 0.0314 | 0.0305 | 0.0327 | 0.0008 | 0.2465 | n/a | n/a | 1.1868 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0078 | 0.0083 | 0.0073 | 0.0099 | 0.0011 | 0.2465 | n/a | n/a | 1.1868 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0375 | 0.0379 | 0.0352 | 0.0432 | 0.0028 | 0.2465 | n/a | n/a | 1.1868 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0139 | 0.0138 | 0.0133 | 0.0145 | 0.0005 | 0.7008 | n/a | n/a | 7.4512 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0097 | 0.0103 | 0.0089 | 0.0143 | 0.0020 | 0.7008 | n/a | n/a | 7.4512 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.1034 | 0.0936 | 0.0659 | 0.1124 | 0.0193 | 0.7008 | n/a | n/a | 7.4512 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1247 | 0.1243 | 0.1203 | 0.1300 | 0.0035 | 0.2735 | n/a | n/a | 0.8775 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0341 | 0.0340 | 0.0321 | 0.0360 | 0.0012 | 0.2735 | n/a | n/a | 0.8775 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1094 | 0.1157 | 0.1046 | 0.1367 | 0.0116 | 0.2735 | n/a | n/a | 0.8775 | no | n/a | n/a | no | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2360 | 0.2388 | 0.2339 | 0.2530 | 0.0071 | 0.1587 | n/a | n/a | 0.1668 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0374 | 0.0388 | 0.0373 | 0.0430 | 0.0022 | 0.1587 | n/a | n/a | 0.1668 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0394 | 0.0397 | 0.0370 | 0.0430 | 0.0022 | 0.1587 | n/a | n/a | 0.1668 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1050 | 0.1053 | 0.1022 | 0.1098 | 0.0026 | 0.1792 | n/a | n/a | 0.3945 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0188 | 0.0189 | 0.0187 | 0.0193 | 0.0002 | 0.1792 | n/a | n/a | 0.3945 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0414 | 0.0416 | 0.0395 | 0.0435 | 0.0017 | 0.1792 | n/a | n/a | 0.3945 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1163 | 0.1181 | 0.1136 | 0.1278 | 0.0052 | 0.2244 | n/a | n/a | 0.3724 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0261 | 0.0262 | 0.0247 | 0.0283 | 0.0013 | 0.2244 | n/a | n/a | 0.3724 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0433 | 0.0441 | 0.0428 | 0.0464 | 0.0013 | 0.2244 | n/a | n/a | 0.3724 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0022 | 0.0025 | 0.0022 | 0.0033 | 0.0004 | 1.0775 | n/a | n/a | 5.5767 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0024 | 0.0026 | 0.0022 | 0.0031 | 0.0003 | 1.0775 | n/a | n/a | 5.5767 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0125 | 0.0144 | 0.0119 | 0.0201 | 0.0031 | 1.0775 | n/a | n/a | 5.5767 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0023 | 0.0025 | 0.0021 | 0.0035 | 0.0005 | 3.6183 | n/a | n/a | 4.7284 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0085 | 0.0085 | 0.0084 | 0.0087 | 0.0002 | 3.6183 | n/a | n/a | 4.7284 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0111 | 0.0111 | 0.0108 | 0.0113 | 0.0002 | 3.6183 | n/a | n/a | 4.7284 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0024 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.9673 | n/a | n/a | 5.2066 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0023 | 0.0025 | 0.0023 | 0.0030 | 0.0003 | 0.9673 | n/a | n/a | 5.2066 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0123 | 0.0123 | 0.0121 | 0.0124 | 0.0002 | 0.9673 | n/a | n/a | 5.2066 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 3.4917 | n/a | n/a | 5.2246 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0080 | 0.0080 | 0.0078 | 0.0081 | 0.0001 | 3.4917 | n/a | n/a | 5.2246 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0120 | 0.0118 | 0.0125 | 0.0002 | 3.4917 | n/a | n/a | 5.2246 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0030 | 0.0002 | 0.8970 | n/a | n/a | 5.5543 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.8970 | n/a | n/a | 5.5543 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0138 | 0.0140 | 0.0135 | 0.0147 | 0.0004 | 0.8970 | n/a | n/a | 5.5543 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 3.5399 | n/a | n/a | 5.1625 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0085 | 0.0085 | 0.0081 | 0.0089 | 0.0003 | 3.5399 | n/a | n/a | 5.1625 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0124 | 0.0126 | 0.0119 | 0.0139 | 0.0007 | 3.5399 | n/a | n/a | 5.1625 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0031 | 0.0032 | 0.0030 | 0.0035 | 0.0002 | 0.7083 | n/a | n/a | 4.1105 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0022 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.7083 | n/a | n/a | 4.1105 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0128 | 0.0130 | 0.0128 | 0.0133 | 0.0002 | 0.7083 | n/a | n/a | 4.1105 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0030 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 2.6626 | n/a | n/a | 4.3466 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0080 | 0.0080 | 0.0079 | 0.0081 | 0.0001 | 2.6626 | n/a | n/a | 4.3466 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0131 | 0.0128 | 0.0124 | 0.0133 | 0.0004 | 2.6626 | n/a | n/a | 4.3466 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0087 | 0.0088 | 0.0084 | 0.0092 | 0.0004 | 0.3240 | n/a | n/a | 2.2722 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0028 | 0.0029 | 0.0027 | 0.0032 | 0.0002 | 0.3240 | n/a | n/a | 2.2722 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0197 | 0.0186 | 0.0138 | 0.0238 | 0.0035 | 0.3240 | n/a | n/a | 2.2722 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0088 | 0.0090 | 0.0084 | 0.0101 | 0.0006 | 0.9709 | n/a | n/a | 1.7352 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0085 | 0.0089 | 0.0083 | 0.0097 | 0.0006 | 0.9709 | n/a | n/a | 1.7352 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0153 | 0.0152 | 0.0140 | 0.0159 | 0.0006 | 0.9709 | n/a | n/a | 1.7352 | no | n/a | n/a | yes | NumPy | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0282 | 0.0281 | 0.0276 | 0.0285 | 0.0004 | 0.1244 | n/a | n/a | 0.5552 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0035 | 0.0038 | 0.0035 | 0.0049 | 0.0005 | 0.1244 | n/a | n/a | 0.5552 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0156 | 0.0157 | 0.0147 | 0.0177 | 0.0011 | 0.1244 | n/a | n/a | 0.5552 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0277 | 0.0277 | 0.0274 | 0.0280 | 0.0002 | 0.3573 | n/a | n/a | 0.5893 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0099 | 0.0098 | 0.0091 | 0.0106 | 0.0005 | 0.3573 | n/a | n/a | 0.5893 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0163 | 0.0164 | 0.0154 | 0.0184 | 0.0011 | 0.3573 | n/a | n/a | 0.5893 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1039 | 0.1038 | 0.1022 | 0.1045 | 0.0008 | 0.0657 | n/a | n/a | 0.2813 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0068 | 0.0068 | 0.0068 | 0.0068 | 0.0000 | 0.0657 | n/a | n/a | 0.2813 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0292 | 0.0277 | 0.0214 | 0.0336 | 0.0042 | 0.0657 | n/a | n/a | 0.2813 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1038 | 0.1047 | 0.1026 | 0.1080 | 0.0021 | 0.1233 | n/a | n/a | 0.1937 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0128 | 0.0128 | 0.0128 | 0.0128 | 0.0000 | 0.1233 | n/a | n/a | 0.1937 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0201 | 0.0241 | 0.0194 | 0.0405 | 0.0082 | 0.1233 | n/a | n/a | 0.1937 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0029 | 0.0029 | 0.0027 | 0.0033 | 0.0002 | 0.9395 | n/a | n/a | 4.5506 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0027 | 0.0027 | 0.0027 | 0.0027 | 0.0000 | 0.9395 | n/a | n/a | 4.5506 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0132 | 0.0131 | 0.0129 | 0.0133 | 0.0002 | 0.9395 | n/a | n/a | 4.5506 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0026 | 0.0026 | 0.0026 | 0.0026 | 0.0000 | 4.1560 | n/a | n/a | 4.3614 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0108 | 0.0106 | 0.0099 | 0.0112 | 0.0005 | 4.1560 | n/a | n/a | 4.3614 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0114 | 0.0116 | 0.0111 | 0.0123 | 0.0005 | 4.1560 | n/a | n/a | 4.3614 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0060 | 0.0061 | 0.0054 | 0.0068 | 0.0006 | 0.7029 | n/a | n/a | 2.5567 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0042 | 0.0042 | 0.0040 | 0.0045 | 0.0002 | 0.7029 | n/a | n/a | 2.5567 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0152 | 0.0152 | 0.0147 | 0.0156 | 0.0003 | 0.7029 | n/a | n/a | 2.5567 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0037 | 0.0039 | 0.0036 | 0.0045 | 0.0003 | 3.1997 | n/a | n/a | 4.1369 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0119 | 0.0121 | 0.0115 | 0.0130 | 0.0006 | 3.1997 | n/a | n/a | 4.1369 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0154 | 0.0155 | 0.0148 | 0.0165 | 0.0006 | 3.1997 | n/a | n/a | 4.1369 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0165 | 0.0170 | 0.0160 | 0.0183 | 0.0009 | 0.4603 | n/a | n/a | 1.5143 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0076 | 0.0076 | 0.0075 | 0.0076 | 0.0000 | 0.4603 | n/a | n/a | 1.5143 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0250 | 0.0248 | 0.0173 | 0.0325 | 0.0061 | 0.4603 | n/a | n/a | 1.5143 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0080 | 0.0088 | 0.0066 | 0.0124 | 0.0021 | 1.8698 | n/a | n/a | 1.8454 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0150 | 0.0150 | 0.0146 | 0.0158 | 0.0004 | 1.8698 | n/a | n/a | 1.8454 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0148 | 0.0159 | 0.0138 | 0.0210 | 0.0026 | 1.8698 | n/a | n/a | 1.8454 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0717 | 0.0717 | 0.0653 | 0.0794 | 0.0054 | 0.3053 | n/a | n/a | 1.2335 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0219 | 0.0221 | 0.0217 | 0.0230 | 0.0005 | 0.3053 | n/a | n/a | 1.2335 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0884 | 0.0807 | 0.0631 | 0.0962 | 0.0145 | 0.3053 | n/a | n/a | 1.2335 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0212 | 0.0243 | 0.0209 | 0.0325 | 0.0045 | 1.3010 | n/a | n/a | 0.7679 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0276 | 0.0308 | 0.0231 | 0.0392 | 0.0069 | 1.3010 | n/a | n/a | 0.7679 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0163 | 0.0209 | 0.0160 | 0.0354 | 0.0075 | 1.3010 | n/a | n/a | 0.7679 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0033 | 0.0033 | 0.0033 | 0.0033 | 0.0000 | 0.5759 | n/a | n/a | 2.9663 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 0.5759 | n/a | n/a | 2.9663 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0098 | 0.0138 | 0.0090 | 0.0258 | 0.0064 | 0.5759 | n/a | n/a | 2.9663 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0349 | 0.0349 | 0.0348 | 0.0350 | 0.0001 | 0.2961 | n/a | n/a | 0.7494 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0103 | 0.0104 | 0.0103 | 0.0105 | 0.0001 | 0.2961 | n/a | n/a | 0.7494 | no | n/a | n/a | no | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0262 | 0.0369 | 0.0239 | 0.0793 | 0.0213 | 0.2961 | n/a | n/a | 0.7494 | no | n/a | n/a | no | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.2944 | 0.3141 | 0.2664 | 0.3855 | 0.0487 | 1.1553 | n/a | n/a | 0.4765 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3401 | 0.3412 | 0.3056 | 0.3733 | 0.0216 | 1.1553 | n/a | n/a | 0.4765 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1403 | 0.1651 | 0.1070 | 0.3057 | 0.0721 | 1.1553 | n/a | n/a | 0.4765 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.2976 | 2.5542 | 2.2122 | 3.0882 | 0.3872 | 0.1929 | n/a | n/a | 0.1273 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4432 | 0.4457 | 0.4288 | 0.4707 | 0.0148 | 0.1929 | n/a | n/a | 0.1273 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2925 | 0.2981 | 0.2396 | 0.3809 | 0.0521 | 0.1929 | n/a | n/a | 0.1273 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1520 | 0.1549 | 0.1486 | 0.1640 | 0.0061 | 8.7030 | n/a | n/a | 0.7106 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.3230 | 1.3311 | 1.2782 | 1.4238 | 0.0498 | 8.7030 | n/a | n/a | 0.7106 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1080 | 0.1078 | 0.0967 | 0.1200 | 0.0090 | 8.7030 | n/a | n/a | 0.7106 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.4741 | 7.4737 | 7.3557 | 7.6517 | 0.1014 | 8.7217 | n/a | n/a | 0.0425 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 65.1864 | 63.8329 | 61.2489 | 65.7126 | 2.0227 | 8.7217 | n/a | n/a | 0.0425 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3179 | 0.3394 | 0.3108 | 0.4305 | 0.0459 | 8.7217 | n/a | n/a | 0.0425 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0141 | 0.0148 | 0.0141 | 0.0160 | 0.0009 | 11.7657 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1658 | 0.1659 | 0.1642 | 0.1674 | 0.0011 | 11.7657 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0141 | 0.0166 | 0.0137 | 0.0212 | 0.0033 | 40.3752 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.5710 | 0.5741 | 0.5564 | 0.6005 | 0.0152 | 40.3752 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5639 | 0.5609 | 0.5538 | 0.5663 | 0.0051 | 13.9211 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 7.8500 | 7.8556 | 7.7725 | 7.9230 | 0.0517 | 13.9211 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5462 | 0.5453 | 0.5414 | 0.5477 | 0.0024 | 49.7668 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 27.1808 | 27.3334 | 26.8306 | 28.3429 | 0.5442 | 49.7668 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0150 | 0.0152 | 0.0149 | 0.0155 | 0.0003 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0321 | 0.0324 | 0.0318 | 0.0332 | 0.0005 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1412 | 0.1417 | 0.1408 | 0.1441 | 0.0012 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3310 | 0.3357 | 0.3259 | 0.3471 | 0.0083 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.0338 | 1.0598 | 1.0206 | 1.1743 | 0.0576 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.4195 | 2.4112 | 2.3850 | 2.4388 | 0.0200 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.5992 | 1.6195 | 1.5795 | 1.7011 | 0.0439 | 0.1022 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1635 | 0.1653 | 0.1526 | 0.1831 | 0.0101 | 0.1022 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
