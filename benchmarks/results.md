# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.3.6`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `27`
- TensorStudio losses versus NumPy: `76`
- TensorStudio wins versus JAX CPU dispatch: `85`
- TensorStudio losses versus JAX CPU dispatch: `13`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0026 | 0.0026 | 0.0025 | 0.0027 | 0.0001 | 0.2604 | n/a | n/a | 4.3415 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2604 | n/a | n/a | 4.3415 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0113 | 0.0112 | 0.0109 | 0.0115 | 0.0002 | 0.2604 | n/a | n/a | 4.3415 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0030 | 0.0001 | 0.2655 | n/a | n/a | 4.1326 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2655 | n/a | n/a | 4.1326 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0110 | 0.0110 | 0.0107 | 0.0116 | 0.0003 | 0.2655 | n/a | n/a | 4.1326 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0025 | 0.0026 | 0.0025 | 0.0027 | 0.0001 | 0.3038 | n/a | n/a | 4.5050 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0008 | 0.0000 | 0.3038 | n/a | n/a | 4.5050 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0113 | 0.0123 | 0.0110 | 0.0144 | 0.0014 | 0.3038 | n/a | n/a | 4.5050 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0026 | 0.0026 | 0.0026 | 0.0026 | 0.0000 | 0.2896 | n/a | n/a | 3.2589 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2896 | n/a | n/a | 3.2589 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0085 | 0.0087 | 0.0085 | 0.0089 | 0.0002 | 0.2896 | n/a | n/a | 3.2589 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0093 | 0.0094 | 0.0093 | 0.0096 | 0.0001 | 0.6152 | n/a | n/a | 8.9306 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0058 | 0.0059 | 0.0056 | 0.0064 | 0.0003 | 0.6152 | n/a | n/a | 8.9306 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.0835 | 0.0843 | 0.0826 | 0.0883 | 0.0021 | 0.6152 | n/a | n/a | 8.9306 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0025 | 0.0026 | 0.0025 | 0.0026 | 0.0000 | 0.2712 | n/a | n/a | 4.3466 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2712 | n/a | n/a | 4.3466 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0111 | 0.0111 | 0.0108 | 0.0113 | 0.0001 | 0.2712 | n/a | n/a | 4.3466 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0001 | 0.2749 | n/a | n/a | 4.5515 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2749 | n/a | n/a | 4.5515 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0119 | 0.0119 | 0.0117 | 0.0122 | 0.0002 | 0.2749 | n/a | n/a | 4.5515 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0026 | 0.0027 | 0.0025 | 0.0028 | 0.0001 | 0.2912 | n/a | n/a | 4.4156 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0009 | 0.0001 | 0.2912 | n/a | n/a | 4.4156 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0116 | 0.0117 | 0.0113 | 0.0120 | 0.0003 | 0.2912 | n/a | n/a | 4.4156 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0028 | 0.0001 | 0.2860 | n/a | n/a | 3.2805 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0008 | 0.0009 | 0.0001 | 0.2860 | n/a | n/a | 3.2805 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0088 | 0.0088 | 0.0084 | 0.0091 | 0.0002 | 0.2860 | n/a | n/a | 3.2805 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0095 | 0.0096 | 0.0094 | 0.0100 | 0.0002 | 0.6266 | n/a | n/a | 9.3277 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0060 | 0.0060 | 0.0057 | 0.0062 | 0.0002 | 0.6266 | n/a | n/a | 9.3277 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.0887 | 0.0892 | 0.0873 | 0.0934 | 0.0022 | 0.6266 | n/a | n/a | 9.3277 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0027 | 0.0029 | 0.0025 | 0.0035 | 0.0004 | 0.2683 | n/a | n/a | 4.3975 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2683 | n/a | n/a | 4.3975 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0117 | 0.0120 | 0.0111 | 0.0134 | 0.0008 | 0.2683 | n/a | n/a | 4.3975 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0027 | 0.0030 | 0.0025 | 0.0042 | 0.0006 | 0.3366 | n/a | n/a | 4.2809 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0009 | 0.0011 | 0.0007 | 0.0016 | 0.0004 | 0.3366 | n/a | n/a | 4.2809 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0116 | 0.0117 | 0.0115 | 0.0120 | 0.0002 | 0.3366 | n/a | n/a | 4.2809 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0028 | 0.0030 | 0.0024 | 0.0043 | 0.0007 | 0.2792 | n/a | n/a | 4.4103 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.2792 | n/a | n/a | 4.4103 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0122 | 0.0123 | 0.0116 | 0.0131 | 0.0005 | 0.2792 | n/a | n/a | 4.4103 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0027 | 0.0027 | 0.0025 | 0.0029 | 0.0001 | 0.2858 | n/a | n/a | 3.4552 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0011 | 0.0001 | 0.2858 | n/a | n/a | 3.4552 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0092 | 0.0093 | 0.0089 | 0.0101 | 0.0004 | 0.2858 | n/a | n/a | 3.4552 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0097 | 0.0097 | 0.0094 | 0.0101 | 0.0002 | 0.6673 | n/a | n/a | 9.6285 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0064 | 0.0063 | 0.0057 | 0.0066 | 0.0003 | 0.6673 | n/a | n/a | 9.6285 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.0931 | 0.0938 | 0.0897 | 0.0999 | 0.0034 | 0.6673 | n/a | n/a | 9.6285 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.2631 | n/a | n/a | 4.4624 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.2631 | n/a | n/a | 4.4624 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0119 | 0.0120 | 0.0112 | 0.0132 | 0.0007 | 0.2631 | n/a | n/a | 4.4624 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0027 | 0.0029 | 0.0026 | 0.0034 | 0.0004 | 0.2600 | n/a | n/a | 4.1726 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.2600 | n/a | n/a | 4.1726 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0115 | 0.0116 | 0.0110 | 0.0121 | 0.0004 | 0.2600 | n/a | n/a | 4.1726 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0028 | 0.0030 | 0.0026 | 0.0034 | 0.0004 | 0.2797 | n/a | n/a | 4.2982 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0013 | 0.0002 | 0.2797 | n/a | n/a | 4.2982 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0122 | 0.0122 | 0.0116 | 0.0129 | 0.0005 | 0.2797 | n/a | n/a | 4.2982 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0029 | 0.0036 | 0.0026 | 0.0048 | 0.0010 | 0.2942 | n/a | n/a | 3.2582 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0009 | 0.0009 | 0.0007 | 0.0011 | 0.0001 | 0.2942 | n/a | n/a | 3.2582 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0096 | 0.0096 | 0.0093 | 0.0102 | 0.0003 | 0.2942 | n/a | n/a | 3.2582 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0100 | 0.0127 | 0.0099 | 0.0227 | 0.0050 | 0.5998 | n/a | n/a | 8.9405 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0060 | 0.0063 | 0.0059 | 0.0071 | 0.0005 | 0.5998 | n/a | n/a | 8.9405 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.0894 | 0.0927 | 0.0886 | 0.1009 | 0.0048 | 0.5998 | n/a | n/a | 8.9405 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0036 | 0.0036 | 0.0031 | 0.0041 | 0.0004 | 0.2806 | n/a | n/a | 4.9749 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.2806 | n/a | n/a | 4.9749 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0181 | 0.0182 | 0.0175 | 0.0191 | 0.0005 | 0.2806 | n/a | n/a | 4.9749 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0030 | 0.0031 | 0.0030 | 0.0036 | 0.0002 | 0.3588 | n/a | n/a | 5.6346 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0014 | 0.0001 | 0.3588 | n/a | n/a | 5.6346 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0171 | 0.0166 | 0.0142 | 0.0180 | 0.0013 | 0.3588 | n/a | n/a | 5.6346 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0030 | 0.0032 | 0.0029 | 0.0035 | 0.0002 | 0.3595 | n/a | n/a | 5.9298 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.3595 | n/a | n/a | 5.9298 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0179 | 0.0164 | 0.0126 | 0.0189 | 0.0025 | 0.3595 | n/a | n/a | 5.9298 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0036 | 0.0035 | 0.0032 | 0.0041 | 0.0003 | 0.3111 | n/a | n/a | 3.9700 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0011 | 0.0000 | 0.3111 | n/a | n/a | 3.9700 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0141 | 0.0139 | 0.0127 | 0.0146 | 0.0007 | 0.3111 | n/a | n/a | 3.9700 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0121 | 0.0129 | 0.0118 | 0.0150 | 0.0012 | 0.6188 | n/a | n/a | 13.2424 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0075 | 0.0081 | 0.0074 | 0.0098 | 0.0009 | 0.6188 | n/a | n/a | 13.2424 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1606 | 0.1591 | 0.1525 | 0.1614 | 0.0034 | 0.6188 | n/a | n/a | 13.2424 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0048 | 0.0050 | 0.0047 | 0.0057 | 0.0004 | 0.3565 | n/a | n/a | 4.0521 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.3565 | n/a | n/a | 4.0521 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0192 | 0.0192 | 0.0184 | 0.0198 | 0.0005 | 0.3565 | n/a | n/a | 4.0521 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0046 | 0.0047 | 0.0045 | 0.0049 | 0.0001 | 0.3165 | n/a | n/a | 3.6751 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.3165 | n/a | n/a | 3.6751 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0170 | 0.0158 | 0.0122 | 0.0191 | 0.0030 | 0.3165 | n/a | n/a | 3.6751 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0048 | 0.0051 | 0.0044 | 0.0059 | 0.0007 | 0.3480 | n/a | n/a | 4.0870 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0017 | 0.0018 | 0.0016 | 0.0022 | 0.0002 | 0.3480 | n/a | n/a | 4.0870 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0195 | 0.0190 | 0.0167 | 0.0203 | 0.0012 | 0.3480 | n/a | n/a | 4.0870 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0055 | 0.0055 | 0.0054 | 0.0058 | 0.0001 | 0.3327 | n/a | n/a | 2.6388 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0018 | 0.0021 | 0.0017 | 0.0029 | 0.0004 | 0.3327 | n/a | n/a | 2.6388 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0146 | 0.0134 | 0.0100 | 0.0155 | 0.0023 | 0.3327 | n/a | n/a | 2.6388 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0202 | 0.0198 | 0.0189 | 0.0204 | 0.0006 | 0.5374 | n/a | n/a | 8.1573 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0108 | 0.0108 | 0.0102 | 0.0116 | 0.0005 | 0.5374 | n/a | n/a | 8.1573 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1646 | 0.1623 | 0.1517 | 0.1660 | 0.0053 | 0.5374 | n/a | n/a | 8.1573 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0113 | 0.0117 | 0.0105 | 0.0130 | 0.0009 | 0.3327 | n/a | n/a | 1.9066 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0038 | 0.0000 | 0.3327 | n/a | n/a | 1.9066 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0216 | 0.0224 | 0.0212 | 0.0262 | 0.0019 | 0.3327 | n/a | n/a | 1.9066 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0113 | 0.0113 | 0.0099 | 0.0131 | 0.0011 | 0.3442 | n/a | n/a | 1.2470 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0040 | 0.0038 | 0.0043 | 0.0002 | 0.3442 | n/a | n/a | 1.2470 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0141 | 0.0143 | 0.0140 | 0.0150 | 0.0004 | 0.3442 | n/a | n/a | 1.2470 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0158 | 0.0149 | 0.0102 | 0.0210 | 0.0038 | 0.2569 | n/a | n/a | 0.8206 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0041 | 0.0041 | 0.0040 | 0.0045 | 0.0002 | 0.2569 | n/a | n/a | 0.8206 | no | n/a | n/a | no | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0130 | 0.0133 | 0.0129 | 0.0140 | 0.0005 | 0.2569 | n/a | n/a | 0.8206 | no | n/a | n/a | no | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0132 | 0.0139 | 0.0132 | 0.0162 | 0.0012 | 0.3312 | n/a | n/a | 1.3575 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0044 | 0.0045 | 0.0044 | 0.0047 | 0.0001 | 0.3312 | n/a | n/a | 1.3575 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0180 | 0.0206 | 0.0170 | 0.0281 | 0.0042 | 0.3312 | n/a | n/a | 1.3575 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0471 | 0.0469 | 0.0448 | 0.0486 | 0.0013 | 0.4607 | n/a | n/a | 3.6296 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0217 | 0.0226 | 0.0216 | 0.0264 | 0.0019 | 0.4607 | n/a | n/a | 3.6296 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.1708 | 0.1711 | 0.1685 | 0.1742 | 0.0023 | 0.4607 | n/a | n/a | 3.6296 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0017 | 0.0018 | 0.0017 | 0.0021 | 0.0002 | 1.2054 | n/a | n/a | 25.0949 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0021 | 0.0020 | 0.0018 | 0.0022 | 0.0002 | 1.2054 | n/a | n/a | 25.0949 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(1,)` | JAX CPU dispatch | 0.0435 | 0.0436 | 0.0431 | 0.0441 | 0.0004 | 1.2054 | n/a | n/a | 25.0949 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 2.6209 | n/a | n/a | 40.3871 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0047 | 0.0049 | 0.0046 | 0.0055 | 0.0003 | 2.6209 | n/a | n/a | 40.3871 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | JAX CPU dispatch | 0.0721 | 0.0723 | 0.0714 | 0.0739 | 0.0009 | 2.6209 | n/a | n/a | 40.3871 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 0.6913 | n/a | n/a | 16.5592 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0014 | 0.0012 | 0.0018 | 0.0002 | 0.6913 | n/a | n/a | 16.5592 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | JAX CPU dispatch | 0.0299 | 0.0301 | 0.0292 | 0.0319 | 0.0009 | 0.6913 | n/a | n/a | 16.5592 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0024 | 0.0023 | 0.0019 | 0.0026 | 0.0002 | 0.5332 | n/a | n/a | 12.4549 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0018 | 0.0002 | 0.5332 | n/a | n/a | 12.4549 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1,)` | JAX CPU dispatch | 0.0299 | 0.0302 | 0.0293 | 0.0320 | 0.0010 | 0.5332 | n/a | n/a | 12.4549 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0020 | 0.0020 | 0.0018 | 0.0022 | 0.0002 | 0.6746 | n/a | n/a | 14.9008 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.6746 | n/a | n/a | 14.9008 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1,)` | JAX CPU dispatch | 0.0299 | 0.0300 | 0.0294 | 0.0310 | 0.0006 | 0.6746 | n/a | n/a | 14.9008 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0017 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 1.0598 | n/a | n/a | 25.9586 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0018 | 0.0019 | 0.0018 | 0.0022 | 0.0002 | 1.0598 | n/a | n/a | 25.9586 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(8,)` | JAX CPU dispatch | 0.0451 | 0.0458 | 0.0442 | 0.0491 | 0.0017 | 1.0598 | n/a | n/a | 25.9586 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 2.5449 | n/a | n/a | 49.5471 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0047 | 0.0047 | 0.0045 | 0.0048 | 0.0001 | 2.5449 | n/a | n/a | 49.5471 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | JAX CPU dispatch | 0.0916 | 0.0945 | 0.0737 | 0.1350 | 0.0224 | 2.5449 | n/a | n/a | 49.5471 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0018 | 0.0019 | 0.0018 | 0.0024 | 0.0002 | 0.7568 | n/a | n/a | 15.8006 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0012 | 0.0017 | 0.0001 | 0.7568 | n/a | n/a | 15.8006 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | JAX CPU dispatch | 0.0289 | 0.0291 | 0.0288 | 0.0296 | 0.0003 | 0.7568 | n/a | n/a | 15.8006 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0018 | 0.0020 | 0.0018 | 0.0026 | 0.0003 | 0.7478 | n/a | n/a | 16.5986 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0012 | 0.0017 | 0.0002 | 0.7478 | n/a | n/a | 16.5986 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(8,)` | JAX CPU dispatch | 0.0300 | 0.0301 | 0.0298 | 0.0305 | 0.0003 | 0.7478 | n/a | n/a | 16.5986 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 0.7768 | n/a | n/a | 15.8175 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0015 | 0.0015 | 0.0013 | 0.0016 | 0.0001 | 0.7768 | n/a | n/a | 15.8175 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(8,)` | JAX CPU dispatch | 0.0295 | 0.0297 | 0.0286 | 0.0316 | 0.0010 | 0.7768 | n/a | n/a | 15.8175 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 1.0232 | n/a | n/a | 23.8213 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0018 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 1.0232 | n/a | n/a | 23.8213 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | relu | `(32,)` | JAX CPU dispatch | 0.0430 | 0.0433 | 0.0427 | 0.0446 | 0.0007 | 1.0232 | n/a | n/a | 23.8213 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0001 | 2.4028 | n/a | n/a | 37.3031 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0047 | 0.0049 | 0.0047 | 0.0052 | 0.0002 | 2.4028 | n/a | n/a | 37.3031 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | JAX CPU dispatch | 0.0737 | 0.0735 | 0.0725 | 0.0742 | 0.0006 | 2.4028 | n/a | n/a | 37.3031 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0022 | 0.0024 | 0.0021 | 0.0028 | 0.0003 | 0.5848 | n/a | n/a | 13.1374 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 0.5848 | n/a | n/a | 13.1374 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | JAX CPU dispatch | 0.0295 | 0.0301 | 0.0290 | 0.0328 | 0.0014 | 0.5848 | n/a | n/a | 13.1374 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0024 | 0.0001 | 0.6041 | n/a | n/a | 13.9319 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0015 | 0.0001 | 0.6041 | n/a | n/a | 13.9319 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(32,)` | JAX CPU dispatch | 0.0297 | 0.0300 | 0.0289 | 0.0324 | 0.0013 | 0.6041 | n/a | n/a | 13.9319 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0021 | 0.0022 | 0.0019 | 0.0025 | 0.0003 | 0.6153 | n/a | n/a | 14.4325 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0018 | 0.0002 | 0.6153 | n/a | n/a | 14.4325 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(32,)` | JAX CPU dispatch | 0.0300 | 0.0301 | 0.0293 | 0.0310 | 0.0005 | 0.6153 | n/a | n/a | 14.4325 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 0.9802 | n/a | n/a | 22.8329 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0021 | 0.0001 | 0.9802 | n/a | n/a | 22.8329 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(128,)` | JAX CPU dispatch | 0.0441 | 0.0440 | 0.0429 | 0.0453 | 0.0008 | 0.9802 | n/a | n/a | 22.8329 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0028 | 0.0028 | 0.0026 | 0.0031 | 0.0002 | 1.7769 | n/a | n/a | 25.6381 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0050 | 0.0051 | 0.0046 | 0.0060 | 0.0005 | 1.7769 | n/a | n/a | 25.6381 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | JAX CPU dispatch | 0.0721 | 0.0728 | 0.0713 | 0.0761 | 0.0017 | 1.7769 | n/a | n/a | 25.6381 | yes | n/a | n/a | yes | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0030 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 0.5311 | n/a | n/a | 10.0826 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0016 | 0.0017 | 0.0015 | 0.0020 | 0.0002 | 0.5311 | n/a | n/a | 10.0826 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | JAX CPU dispatch | 0.0301 | 0.0302 | 0.0295 | 0.0311 | 0.0005 | 0.5311 | n/a | n/a | 10.0826 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0027 | 0.0027 | 0.0025 | 0.0029 | 0.0001 | 0.5038 | n/a | n/a | 11.1650 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0018 | 0.0002 | 0.5038 | n/a | n/a | 11.1650 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(128,)` | JAX CPU dispatch | 0.0301 | 0.0299 | 0.0292 | 0.0303 | 0.0004 | 0.5038 | n/a | n/a | 11.1650 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0028 | 0.0029 | 0.0027 | 0.0032 | 0.0002 | 0.5353 | n/a | n/a | 10.9066 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.5353 | n/a | n/a | 10.9066 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(128,)` | JAX CPU dispatch | 0.0304 | 0.0302 | 0.0288 | 0.0312 | 0.0008 | 0.5353 | n/a | n/a | 10.9066 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0051 | 0.0043 | 0.0024 | 0.0061 | 0.0016 | 0.4881 | n/a | n/a | 16.0759 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0025 | 0.0025 | 0.0025 | 0.0025 | 0.0000 | 0.4881 | n/a | n/a | 16.0759 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | JAX CPU dispatch | 0.0819 | 0.0770 | 0.0564 | 0.0834 | 0.0103 | 0.4881 | n/a | n/a | 16.0759 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0092 | 0.0095 | 0.0092 | 0.0104 | 0.0005 | 0.7810 | n/a | n/a | 14.6484 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0072 | 0.0076 | 0.0072 | 0.0088 | 0.0006 | 0.7810 | n/a | n/a | 14.6484 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | JAX CPU dispatch | 0.1353 | 0.1360 | 0.1346 | 0.1377 | 0.0013 | 0.7810 | n/a | n/a | 14.6484 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0139 | 0.0140 | 0.0136 | 0.0146 | 0.0004 | 0.2809 | n/a | n/a | 2.1942 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0040 | 0.0001 | 0.2809 | n/a | n/a | 2.1942 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | JAX CPU dispatch | 0.0305 | 0.0319 | 0.0292 | 0.0386 | 0.0034 | 0.2809 | n/a | n/a | 2.1942 | no | n/a | n/a | yes | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0089 | 0.0090 | 0.0081 | 0.0099 | 0.0007 | 0.2969 | n/a | n/a | 3.5192 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0027 | 0.0000 | 0.2969 | n/a | n/a | 3.5192 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | JAX CPU dispatch | 0.0315 | 0.0313 | 0.0309 | 0.0316 | 0.0003 | 0.2969 | n/a | n/a | 3.5192 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0086 | 0.0087 | 0.0084 | 0.0092 | 0.0003 | 0.3322 | n/a | n/a | 3.6510 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0031 | 0.0028 | 0.0036 | 0.0003 | 0.3322 | n/a | n/a | 3.6510 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(1024,)` | JAX CPU dispatch | 0.0313 | 0.0316 | 0.0308 | 0.0333 | 0.0009 | 0.3322 | n/a | n/a | 3.6510 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0053 | 0.0053 | 0.0050 | 0.0058 | 0.0003 | 0.7256 | n/a | n/a | 15.5506 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0038 | 0.0039 | 0.0038 | 0.0044 | 0.0002 | 0.7256 | n/a | n/a | 15.5506 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | JAX CPU dispatch | 0.0820 | 0.0823 | 0.0808 | 0.0844 | 0.0014 | 0.7256 | n/a | n/a | 15.5506 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0324 | 0.0324 | 0.0322 | 0.0327 | 0.0002 | 0.3669 | n/a | n/a | 3.2790 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0119 | 0.0125 | 0.0116 | 0.0138 | 0.0009 | 0.3669 | n/a | n/a | 3.2790 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | JAX CPU dispatch | 0.1063 | 0.1031 | 0.0864 | 0.1159 | 0.0101 | 0.3669 | n/a | n/a | 3.2790 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0661 | 0.0690 | 0.0603 | 0.0876 | 0.0101 | 0.1742 | n/a | n/a | 0.4973 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0115 | 0.0136 | 0.0106 | 0.0196 | 0.0034 | 0.1742 | n/a | n/a | 0.4973 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | JAX CPU dispatch | 0.0329 | 0.0336 | 0.0322 | 0.0369 | 0.0017 | 0.1742 | n/a | n/a | 0.4973 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0302 | 0.0316 | 0.0281 | 0.0381 | 0.0037 | 0.2002 | n/a | n/a | 1.1157 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0060 | 0.0067 | 0.0059 | 0.0090 | 0.0012 | 0.2002 | n/a | n/a | 1.1157 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | JAX CPU dispatch | 0.0336 | 0.0336 | 0.0333 | 0.0340 | 0.0002 | 0.2002 | n/a | n/a | 1.1157 | no | n/a | n/a | yes | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0303 | 0.0303 | 0.0301 | 0.0305 | 0.0002 | 0.2441 | n/a | n/a | 1.1674 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0074 | 0.0076 | 0.0074 | 0.0082 | 0.0003 | 0.2441 | n/a | n/a | 1.1674 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | log | `(4096,)` | JAX CPU dispatch | 0.0353 | 0.0357 | 0.0346 | 0.0368 | 0.0009 | 0.2441 | n/a | n/a | 1.1674 | no | n/a | n/a | yes | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0136 | 0.0152 | 0.0132 | 0.0217 | 0.0032 | 0.6651 | n/a | n/a | 6.7101 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0090 | 0.0094 | 0.0090 | 0.0111 | 0.0008 | 0.6651 | n/a | n/a | 6.7101 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | JAX CPU dispatch | 0.0910 | 0.0915 | 0.0892 | 0.0945 | 0.0020 | 0.6651 | n/a | n/a | 6.7101 | no | n/a | n/a | yes | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1241 | 0.1238 | 0.1205 | 0.1257 | 0.0019 | 0.2796 | n/a | n/a | 1.2600 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0347 | 0.0353 | 0.0339 | 0.0371 | 0.0012 | 0.2796 | n/a | n/a | 1.2600 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | JAX CPU dispatch | 0.1563 | 0.1550 | 0.1498 | 0.1574 | 0.0028 | 0.2796 | n/a | n/a | 1.2600 | no | n/a | n/a | yes | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.2519 | 0.2483 | 0.2386 | 0.2566 | 0.0080 | 0.1508 | n/a | n/a | 0.1509 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0380 | 0.0424 | 0.0377 | 0.0537 | 0.0062 | 0.1508 | n/a | n/a | 0.1509 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | tanh | `(16384,)` | JAX CPU dispatch | 0.0380 | 0.0393 | 0.0377 | 0.0430 | 0.0020 | 0.1508 | n/a | n/a | 0.1509 | no | n/a | n/a | no | NumPy | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1050 | 0.1040 | 0.1007 | 0.1075 | 0.0025 | 0.1777 | n/a | n/a | 0.3935 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0187 | 0.0190 | 0.0184 | 0.0205 | 0.0008 | 0.1777 | n/a | n/a | 0.3935 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | JAX CPU dispatch | 0.0413 | 0.0412 | 0.0390 | 0.0438 | 0.0018 | 0.1777 | n/a | n/a | 0.3935 | no | n/a | n/a | no | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1131 | 0.1126 | 0.1082 | 0.1174 | 0.0030 | 0.2184 | n/a | n/a | 0.3879 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0247 | 0.0253 | 0.0246 | 0.0271 | 0.0009 | 0.2184 | n/a | n/a | 0.3879 | no | n/a | n/a | no | NumPy | NumPy baseline |
| activations | log | `(16384,)` | JAX CPU dispatch | 0.0439 | 0.0447 | 0.0432 | 0.0478 | 0.0017 | 0.2184 | n/a | n/a | 0.3879 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0021 | 0.0026 | 0.0002 | 1.0375 | n/a | n/a | 5.2655 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 1.0375 | n/a | n/a | 5.2655 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0115 | 0.0115 | 0.0112 | 0.0117 | 0.0002 | 1.0375 | n/a | n/a | 5.2655 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0021 | 0.0024 | 0.0021 | 0.0033 | 0.0004 | 4.1243 | n/a | n/a | 5.2052 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0088 | 0.0087 | 0.0084 | 0.0092 | 0.0003 | 4.1243 | n/a | n/a | 5.2052 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0111 | 0.0112 | 0.0108 | 0.0119 | 0.0004 | 4.1243 | n/a | n/a | 5.2052 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0001 | 1.0613 | n/a | n/a | 5.6130 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0023 | 0.0025 | 0.0023 | 0.0028 | 0.0002 | 1.0613 | n/a | n/a | 5.6130 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0121 | 0.0122 | 0.0119 | 0.0127 | 0.0003 | 1.0613 | n/a | n/a | 5.6130 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 4.0185 | n/a | n/a | 5.6748 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0087 | 0.0095 | 0.0086 | 0.0116 | 0.0012 | 4.0185 | n/a | n/a | 5.6748 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0123 | 0.0122 | 0.0115 | 0.0128 | 0.0004 | 4.0185 | n/a | n/a | 5.6748 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0025 | 0.0025 | 0.0023 | 0.0027 | 0.0001 | 0.9256 | n/a | n/a | 5.3569 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0026 | 0.0001 | 0.9256 | n/a | n/a | 5.3569 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0131 | 0.0131 | 0.0126 | 0.0139 | 0.0004 | 0.9256 | n/a | n/a | 5.3569 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0024 | 0.0025 | 0.0023 | 0.0027 | 0.0002 | 3.7243 | n/a | n/a | 4.8115 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0089 | 0.0089 | 0.0085 | 0.0093 | 0.0003 | 3.7243 | n/a | n/a | 4.8115 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0115 | 0.0115 | 0.0110 | 0.0122 | 0.0004 | 3.7243 | n/a | n/a | 4.8115 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0032 | 0.0031 | 0.0028 | 0.0037 | 0.0003 | 0.7673 | n/a | n/a | 4.1468 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0024 | 0.0027 | 0.0023 | 0.0038 | 0.0006 | 0.7673 | n/a | n/a | 4.1468 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0132 | 0.0133 | 0.0131 | 0.0138 | 0.0002 | 0.7673 | n/a | n/a | 4.1468 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0030 | 0.0030 | 0.0029 | 0.0031 | 0.0001 | 2.8256 | n/a | n/a | 4.2652 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0085 | 0.0085 | 0.0084 | 0.0088 | 0.0002 | 2.8256 | n/a | n/a | 4.2652 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0128 | 0.0130 | 0.0122 | 0.0145 | 0.0008 | 2.8256 | n/a | n/a | 4.2652 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0085 | 0.0088 | 0.0083 | 0.0095 | 0.0005 | 0.3132 | n/a | n/a | 2.2445 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0026 | 0.0027 | 0.0000 | 0.3132 | n/a | n/a | 2.2445 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0191 | 0.0185 | 0.0147 | 0.0205 | 0.0020 | 0.3132 | n/a | n/a | 2.2445 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0084 | 0.0086 | 0.0083 | 0.0097 | 0.0005 | 1.1476 | n/a | n/a | 2.1492 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0096 | 0.0096 | 0.0086 | 0.0110 | 0.0009 | 1.1476 | n/a | n/a | 2.1492 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0180 | 0.0167 | 0.0141 | 0.0185 | 0.0019 | 1.1476 | n/a | n/a | 2.1492 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0284 | 0.0284 | 0.0282 | 0.0289 | 0.0003 | 0.1262 | n/a | n/a | 0.7003 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0036 | 0.0038 | 0.0036 | 0.0045 | 0.0004 | 0.1262 | n/a | n/a | 0.7003 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0199 | 0.0186 | 0.0150 | 0.0213 | 0.0023 | 0.1262 | n/a | n/a | 0.7003 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0278 | 0.0282 | 0.0273 | 0.0302 | 0.0011 | 0.3866 | n/a | n/a | 0.6819 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0108 | 0.0112 | 0.0100 | 0.0127 | 0.0012 | 0.3866 | n/a | n/a | 0.6819 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0190 | 0.0182 | 0.0148 | 0.0199 | 0.0018 | 0.3866 | n/a | n/a | 0.6819 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1155 | 0.1149 | 0.1124 | 0.1169 | 0.0017 | 0.0623 | n/a | n/a | 0.2400 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0072 | 0.0078 | 0.0070 | 0.0107 | 0.0014 | 0.0623 | n/a | n/a | 0.2400 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0277 | 0.0283 | 0.0268 | 0.0299 | 0.0012 | 0.0623 | n/a | n/a | 0.2400 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1029 | 0.1178 | 0.1022 | 0.1694 | 0.0261 | 0.1284 | n/a | n/a | 0.2503 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0132 | 0.0144 | 0.0132 | 0.0179 | 0.0018 | 0.1284 | n/a | n/a | 0.2503 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0258 | 0.0260 | 0.0253 | 0.0271 | 0.0006 | 0.1284 | n/a | n/a | 0.2503 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0027 | 0.0029 | 0.0027 | 0.0035 | 0.0003 | 1.0159 | n/a | n/a | 4.8244 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0028 | 0.0029 | 0.0027 | 0.0032 | 0.0002 | 1.0159 | n/a | n/a | 4.8244 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0131 | 0.0131 | 0.0123 | 0.0141 | 0.0006 | 1.0159 | n/a | n/a | 4.8244 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0027 | 0.0029 | 0.0027 | 0.0038 | 0.0005 | 6.9616 | n/a | n/a | 4.3981 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0188 | 0.0173 | 0.0107 | 0.0224 | 0.0044 | 6.9616 | n/a | n/a | 4.3981 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0119 | 0.0121 | 0.0110 | 0.0137 | 0.0009 | 6.9616 | n/a | n/a | 4.3981 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0056 | 0.0057 | 0.0055 | 0.0060 | 0.0002 | 0.7324 | n/a | n/a | 3.6234 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0041 | 0.0042 | 0.0041 | 0.0047 | 0.0002 | 0.7324 | n/a | n/a | 3.6234 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0205 | 0.0190 | 0.0158 | 0.0208 | 0.0020 | 0.7324 | n/a | n/a | 3.6234 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0037 | 0.0038 | 0.0037 | 0.0041 | 0.0002 | 3.4748 | n/a | n/a | 5.0602 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0129 | 0.0129 | 0.0120 | 0.0135 | 0.0005 | 3.4748 | n/a | n/a | 5.0602 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0188 | 0.0174 | 0.0130 | 0.0203 | 0.0026 | 3.4748 | n/a | n/a | 5.0602 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0162 | 0.0162 | 0.0160 | 0.0164 | 0.0001 | 0.4971 | n/a | n/a | 1.5494 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0081 | 0.0081 | 0.0080 | 0.0081 | 0.0000 | 0.4971 | n/a | n/a | 1.5494 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0251 | 0.0227 | 0.0175 | 0.0255 | 0.0033 | 0.4971 | n/a | n/a | 1.5494 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0066 | 0.0066 | 0.0066 | 0.0066 | 0.0000 | 2.6388 | n/a | n/a | 2.1855 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0174 | 0.0167 | 0.0150 | 0.0182 | 0.0014 | 2.6388 | n/a | n/a | 2.1855 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0144 | 0.0162 | 0.0128 | 0.0207 | 0.0035 | 2.6388 | n/a | n/a | 2.1855 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0648 | 0.0656 | 0.0642 | 0.0680 | 0.0015 | 0.3345 | n/a | n/a | 1.1645 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0217 | 0.0227 | 0.0215 | 0.0259 | 0.0017 | 0.3345 | n/a | n/a | 1.1645 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0754 | 0.0763 | 0.0728 | 0.0810 | 0.0027 | 0.3345 | n/a | n/a | 1.1645 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0184 | 0.0186 | 0.0183 | 0.0194 | 0.0004 | 1.2350 | n/a | n/a | 1.3719 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0227 | 0.0250 | 0.0223 | 0.0297 | 0.0032 | 1.2350 | n/a | n/a | 1.3719 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0252 | 0.0255 | 0.0246 | 0.0269 | 0.0009 | 1.2350 | n/a | n/a | 1.3719 | yes | n/a | n/a | yes | TensorStudio | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0034 | 0.0034 | 0.0034 | 0.0034 | 0.0000 | 1.0573 | n/a | n/a | 2.6977 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0036 | 0.0036 | 0.0028 | 0.0044 | 0.0007 | 1.0573 | n/a | n/a | 2.6977 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| matmul | matmul | `(16, 16)` | JAX CPU dispatch | 0.0091 | 0.0113 | 0.0090 | 0.0152 | 0.0027 | 1.0573 | n/a | n/a | 2.6977 | yes | n/a | n/a | yes | TensorStudio | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0345 | 0.0368 | 0.0343 | 0.0463 | 0.0047 | 0.3155 | n/a | n/a | 1.4296 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0109 | 0.0109 | 0.0108 | 0.0110 | 0.0001 | 0.3155 | n/a | n/a | 1.4296 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | JAX CPU dispatch | 0.0494 | 0.0428 | 0.0232 | 0.0519 | 0.0107 | 0.3155 | n/a | n/a | 1.4296 | no | n/a | n/a | yes | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.2620 | 0.2663 | 0.2610 | 0.2799 | 0.0072 | 1.3408 | n/a | n/a | 0.4619 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.3513 | 0.3511 | 0.3144 | 0.3951 | 0.0259 | 1.3408 | n/a | n/a | 0.4619 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(128, 128)` | JAX CPU dispatch | 0.1210 | 0.1196 | 0.1126 | 0.1284 | 0.0062 | 1.3408 | n/a | n/a | 0.4619 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.2228 | 2.2646 | 2.2090 | 2.3544 | 0.0617 | 0.2021 | n/a | n/a | 0.1261 | no | n/a | n/a | no | JAX CPU dispatch | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4492 | 0.4439 | 0.4029 | 0.4722 | 0.0254 | 0.2021 | n/a | n/a | 0.1261 | no | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| matmul | matmul | `(256, 256)` | JAX CPU dispatch | 0.2803 | 0.2786 | 0.2731 | 0.2821 | 0.0032 | 0.2021 | n/a | n/a | 0.1261 | no | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | TensorStudio | 0.1517 | 0.1518 | 0.1497 | 0.1545 | 0.0016 | 8.5189 | n/a | n/a | 0.7899 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | NumPy | 1.2925 | 1.3066 | 1.2823 | 1.3362 | 0.0242 | 8.5189 | n/a | n/a | 0.7899 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(1, 1, 8, 8)` | JAX CPU dispatch | 0.1198 | 0.1211 | 0.1164 | 0.1270 | 0.0043 | 8.5189 | n/a | n/a | 0.7899 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | TensorStudio | 7.5418 | 7.5529 | 7.5164 | 7.6132 | 0.0360 | 8.4226 | n/a | n/a | 0.0411 | yes | n/a | n/a | no | JAX CPU dispatch | win vs NumPy |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | NumPy | 63.5214 | 63.7343 | 62.8244 | 64.6576 | 0.6405 | 8.4226 | n/a | n/a | 0.0411 | yes | n/a | n/a | no | JAX CPU dispatch | NumPy baseline |
| conv2d | conv2d_3x3_padding1 | `(4, 3, 16, 16)` | JAX CPU dispatch | 0.3099 | 0.3194 | 0.3036 | 0.3439 | 0.0156 | 8.4226 | n/a | n/a | 0.0411 | yes | n/a | n/a | no | JAX CPU dispatch | reference |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0144 | 0.0149 | 0.0143 | 0.0168 | 0.0010 | 12.2930 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.1765 | 0.1762 | 0.1734 | 0.1806 | 0.0026 | 12.2930 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | TensorStudio | 0.0156 | 0.0152 | 0.0140 | 0.0166 | 0.0011 | 39.5541 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(1, 1, 16, 16)` | NumPy | 0.6162 | 0.6166 | 0.5966 | 0.6521 | 0.0196 | 39.5541 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5568 | 0.5904 | 0.5423 | 0.6611 | 0.0485 | 15.2546 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | max_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 8.4943 | 8.5757 | 8.3747 | 8.8608 | 0.1833 | 15.2546 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | TensorStudio | 0.5530 | 0.5529 | 0.5463 | 0.5626 | 0.0055 | 53.3335 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | win vs NumPy |
| pooling | avg_pool2d_2x2 | `(4, 3, 32, 32)` | NumPy | 29.4954 | 29.7615 | 29.3748 | 30.6629 | 0.4739 | 53.3335 | n/a | n/a | n/a | yes | n/a | n/a | n/a | TensorStudio | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0148 | 0.0159 | 0.0143 | 0.0197 | 0.0020 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0307 | 0.0319 | 0.0301 | 0.0368 | 0.0025 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1458 | 0.1640 | 0.1390 | 0.2102 | 0.0285 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3263 | 0.3526 | 0.3238 | 0.4602 | 0.0538 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.0264 | 1.0239 | 1.0159 | 1.0312 | 0.0057 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.4146 | 2.4096 | 2.3658 | 2.4579 | 0.0302 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | TensorStudio | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.5981 | 1.5986 | 1.5944 | 1.6036 | 0.0034 | 0.1010 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1614 | 0.1753 | 0.1602 | 0.2106 | 0.0198 | 0.1010 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
