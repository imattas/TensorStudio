# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.0rc2`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `10`
- TensorStudio losses versus NumPy: `79`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0026 | 0.0026 | 0.0022 | 0.0030 | 0.0003 | 0.2591 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0012 | 0.0002 | 0.2591 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(1,)` | TensorStudio | 0.0027 | 0.0034 | 0.0024 | 0.0051 | 0.0011 | 0.6387 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0018 | 0.0016 | 0.0008 | 0.0020 | 0.0004 | 0.6387 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(1,)` | TensorStudio | 0.0022 | 0.0027 | 0.0021 | 0.0043 | 0.0008 | 0.4409 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0010 | 0.0009 | 0.0007 | 0.0011 | 0.0002 | 0.4409 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(1,)` | TensorStudio | 0.0022 | 0.0022 | 0.0021 | 0.0023 | 0.0001 | 0.3531 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3531 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0084 | 0.0091 | 0.0083 | 0.0109 | 0.0010 | 0.4651 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0039 | 0.0041 | 0.0038 | 0.0044 | 0.0003 | 0.4651 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(8,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0026 | 0.0001 | 0.2741 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0010 | 0.0001 | 0.2741 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(8,)` | TensorStudio | 0.0029 | 0.0029 | 0.0026 | 0.0033 | 0.0002 | 0.2570 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0009 | 0.0007 | 0.0014 | 0.0002 | 0.2570 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(8,)` | TensorStudio | 0.0027 | 0.0027 | 0.0025 | 0.0028 | 0.0001 | 0.3175 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0009 | 0.0000 | 0.3175 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(8,)` | TensorStudio | 0.0029 | 0.0029 | 0.0024 | 0.0036 | 0.0004 | 0.2537 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2537 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0096 | 0.0095 | 0.0092 | 0.0098 | 0.0002 | 0.4443 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0043 | 0.0042 | 0.0040 | 0.0044 | 0.0001 | 0.4443 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(32,)` | TensorStudio | 0.0031 | 0.0031 | 0.0030 | 0.0034 | 0.0001 | 0.2626 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0008 | 0.0010 | 0.0007 | 0.0016 | 0.0003 | 0.2626 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(32,)` | TensorStudio | 0.0031 | 0.0031 | 0.0030 | 0.0033 | 0.0001 | 0.2265 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.2265 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(32,)` | TensorStudio | 0.0032 | 0.0033 | 0.0030 | 0.0040 | 0.0004 | 0.2391 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2391 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(32,)` | TensorStudio | 0.0034 | 0.0037 | 0.0031 | 0.0045 | 0.0006 | 0.2180 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2180 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0125 | 0.0124 | 0.0110 | 0.0138 | 0.0009 | 0.3747 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0047 | 0.0048 | 0.0042 | 0.0060 | 0.0006 | 0.3747 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(128,)` | TensorStudio | 0.0060 | 0.0062 | 0.0057 | 0.0071 | 0.0005 | 0.1140 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.1140 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(128,)` | TensorStudio | 0.0058 | 0.0058 | 0.0056 | 0.0059 | 0.0001 | 0.1237 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.1237 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(128,)` | TensorStudio | 0.0057 | 0.0057 | 0.0055 | 0.0058 | 0.0001 | 0.1330 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0011 | 0.0001 | 0.1330 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(128,)` | TensorStudio | 0.0060 | 0.0060 | 0.0059 | 0.0061 | 0.0001 | 0.1269 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.1269 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0193 | 0.0194 | 0.0189 | 0.0202 | 0.0005 | 0.2166 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0042 | 0.0043 | 0.0040 | 0.0049 | 0.0003 | 0.2166 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(1024,)` | TensorStudio | 0.0291 | 0.0292 | 0.0288 | 0.0296 | 0.0003 | 0.0369 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0011 | 0.0000 | 0.0369 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0284 | 0.0284 | 0.0281 | 0.0285 | 0.0001 | 0.0348 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0348 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0290 | 0.0288 | 0.0282 | 0.0292 | 0.0004 | 0.0337 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0337 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(1024,)` | TensorStudio | 0.0310 | 0.0313 | 0.0306 | 0.0329 | 0.0008 | 0.0339 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0010 | 0.0011 | 0.0000 | 0.0339 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0897 | 0.0939 | 0.0895 | 0.1024 | 0.0054 | 0.0604 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0054 | 0.0054 | 0.0053 | 0.0055 | 0.0001 | 0.0604 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(4096,)` | TensorStudio | 0.1053 | 0.1055 | 0.1048 | 0.1068 | 0.0007 | 0.0170 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.0170 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(4096,)` | TensorStudio | 0.1047 | 0.1053 | 0.1043 | 0.1082 | 0.0015 | 0.0131 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.0131 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(4096,)` | TensorStudio | 0.1037 | 0.1036 | 0.1032 | 0.1038 | 0.0002 | 0.0140 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.0140 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(4096,)` | TensorStudio | 0.1148 | 0.1155 | 0.1143 | 0.1185 | 0.0015 | 0.0141 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 0.0141 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.3297 | 0.3303 | 0.3274 | 0.3342 | 0.0023 | 0.0231 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0076 | 0.0077 | 0.0076 | 0.0078 | 0.0001 | 0.0231 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(16384,)` | TensorStudio | 0.4187 | 0.4259 | 0.4152 | 0.4466 | 0.0122 | 0.0092 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0038 | 0.0038 | 0.0038 | 0.0039 | 0.0000 | 0.0092 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(16384,)` | TensorStudio | 0.4063 | 0.4077 | 0.4053 | 0.4136 | 0.0031 | 0.0096 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0039 | 0.0000 | 0.0096 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(16384,)` | TensorStudio | 0.4108 | 0.4100 | 0.4075 | 0.4128 | 0.0020 | 0.0088 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0000 | 0.0088 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(16384,)` | TensorStudio | 0.4530 | 0.4528 | 0.4508 | 0.4539 | 0.0011 | 0.0093 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0042 | 0.0041 | 0.0044 | 0.0001 | 0.0093 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 1.3060 | 1.3129 | 1.2721 | 1.3699 | 0.0348 | 0.0158 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0206 | 0.0201 | 0.0189 | 0.0214 | 0.0010 | 0.0158 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0001 | 0.8858 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 0.8858 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0016 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 2.2407 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0035 | 0.0035 | 0.0034 | 0.0037 | 0.0001 | 2.2407 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0001 | 0.7886 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0014 | 0.0001 | 0.7886 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.8043 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.8043 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.8068 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.8068 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.9057 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0016 | 0.0001 | 0.9057 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 2.1159 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0035 | 0.0035 | 0.0034 | 0.0038 | 0.0001 | 2.1159 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(8,)` | TensorStudio | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.7196 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.7196 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(8,)` | TensorStudio | 0.0017 | 0.0017 | 0.0016 | 0.0018 | 0.0000 | 0.7206 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0001 | 0.7206 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(8,)` | TensorStudio | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 0.7452 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.7452 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(32,)` | TensorStudio | 0.0021 | 0.0025 | 0.0020 | 0.0041 | 0.0008 | 0.6512 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0014 | 0.0016 | 0.0014 | 0.0020 | 0.0003 | 0.6512 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0024 | 0.0025 | 0.0024 | 0.0030 | 0.0002 | 1.4918 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0036 | 0.0001 | 1.4918 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(32,)` | TensorStudio | 0.0027 | 0.0027 | 0.0025 | 0.0028 | 0.0001 | 0.5250 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0014 | 0.0015 | 0.0012 | 0.0017 | 0.0002 | 0.5250 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0025 | 0.0001 | 0.5193 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.5193 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(32,)` | TensorStudio | 0.0026 | 0.0029 | 0.0023 | 0.0044 | 0.0008 | 0.5073 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.5073 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(128,)` | TensorStudio | 0.0037 | 0.0038 | 0.0037 | 0.0039 | 0.0001 | 0.3995 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | 0.3995 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0055 | 0.0055 | 0.0054 | 0.0056 | 0.0001 | 0.6726 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0037 | 0.0039 | 0.0035 | 0.0046 | 0.0004 | 0.6726 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(128,)` | TensorStudio | 0.0059 | 0.0059 | 0.0058 | 0.0059 | 0.0000 | 0.2522 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.2522 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(128,)` | TensorStudio | 0.0050 | 0.0052 | 0.0048 | 0.0061 | 0.0005 | 0.2646 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0014 | 0.0001 | 0.2646 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(128,)` | TensorStudio | 0.0048 | 0.0048 | 0.0047 | 0.0049 | 0.0001 | 0.2937 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.2937 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(1024,)` | TensorStudio | 0.0194 | 0.0198 | 0.0194 | 0.0213 | 0.0008 | 0.1002 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0020 | 0.0000 | 0.1002 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0330 | 0.0331 | 0.0329 | 0.0335 | 0.0002 | 0.1776 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0059 | 0.0060 | 0.0057 | 0.0067 | 0.0004 | 0.1776 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(1024,)` | TensorStudio | 0.0383 | 0.0387 | 0.0382 | 0.0404 | 0.0008 | 0.0948 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0039 | 0.0001 | 0.0948 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(1024,)` | TensorStudio | 0.0280 | 0.0281 | 0.0279 | 0.0285 | 0.0002 | 0.0894 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0025 | 0.0025 | 0.0025 | 0.0025 | 0.0000 | 0.0894 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(1024,)` | TensorStudio | 0.0275 | 0.0273 | 0.0268 | 0.0276 | 0.0003 | 0.1044 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0029 | 0.0028 | 0.0033 | 0.0002 | 0.1044 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(4096,)` | TensorStudio | 0.0743 | 0.0751 | 0.0733 | 0.0780 | 0.0018 | 0.0411 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0031 | 0.0031 | 0.0030 | 0.0034 | 0.0001 | 0.0411 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.1290 | 0.1290 | 0.1278 | 0.1303 | 0.0009 | 0.0831 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0107 | 0.0106 | 0.0103 | 0.0108 | 0.0002 | 0.0831 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(4096,)` | TensorStudio | 0.1496 | 0.1500 | 0.1488 | 0.1516 | 0.0011 | 0.0705 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0105 | 0.0106 | 0.0105 | 0.0108 | 0.0001 | 0.0705 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(4096,)` | TensorStudio | 0.1078 | 0.1080 | 0.1074 | 0.1092 | 0.0007 | 0.0581 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0063 | 0.0063 | 0.0062 | 0.0064 | 0.0001 | 0.0581 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(4096,)` | TensorStudio | 0.1054 | 0.1059 | 0.1052 | 0.1074 | 0.0008 | 0.0689 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0073 | 0.0073 | 0.0072 | 0.0074 | 0.0001 | 0.0689 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(16384,)` | TensorStudio | 0.2876 | 0.2878 | 0.2869 | 0.2890 | 0.0007 | 0.0270 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0078 | 0.0078 | 0.0077 | 0.0082 | 0.0002 | 0.0270 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.5057 | 0.5086 | 0.5047 | 0.5199 | 0.0057 | 0.0590 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0298 | 0.0299 | 0.0294 | 0.0303 | 0.0003 | 0.0590 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(16384,)` | TensorStudio | 0.5909 | 0.6104 | 0.5874 | 0.6851 | 0.0375 | 0.0678 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0400 | 0.0400 | 0.0393 | 0.0406 | 0.0004 | 0.0678 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(16384,)` | TensorStudio | 0.4358 | 0.4407 | 0.4347 | 0.4516 | 0.0070 | 0.0436 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0190 | 0.0193 | 0.0189 | 0.0201 | 0.0005 | 0.0436 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(16384,)` | TensorStudio | 0.4203 | 0.4304 | 0.4176 | 0.4631 | 0.0170 | 0.0610 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0256 | 0.0258 | 0.0255 | 0.0261 | 0.0002 | 0.0610 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1,)` | TensorStudio | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0001 | 1.3405 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0000 | 1.3405 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1,)` | TensorStudio | 0.0014 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 5.3752 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0073 | 0.0075 | 0.0073 | 0.0081 | 0.0003 | 5.3752 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 1.2617 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0018 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 1.2617 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(8,)` | TensorStudio | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 5.4095 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0076 | 0.0076 | 0.0074 | 0.0079 | 0.0002 | 5.4095 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(32,)` | TensorStudio | 0.0020 | 0.0021 | 0.0018 | 0.0025 | 0.0003 | 1.3282 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0026 | 0.0028 | 0.0019 | 0.0040 | 0.0007 | 1.3282 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(32,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0022 | 0.0001 | 4.5904 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0086 | 0.0084 | 0.0079 | 0.0090 | 0.0004 | 4.5904 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(128,)` | TensorStudio | 0.0032 | 0.0033 | 0.0029 | 0.0040 | 0.0004 | 0.9028 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0029 | 0.0031 | 0.0023 | 0.0045 | 0.0008 | 0.9028 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(128,)` | TensorStudio | 0.0032 | 0.0034 | 0.0031 | 0.0043 | 0.0005 | 2.4459 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0078 | 0.0079 | 0.0076 | 0.0086 | 0.0004 | 2.4459 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1024,)` | TensorStudio | 0.0142 | 0.0150 | 0.0133 | 0.0184 | 0.0019 | 0.1581 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.1581 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1024,)` | TensorStudio | 0.0158 | 0.0166 | 0.0136 | 0.0200 | 0.0026 | 0.6764 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0107 | 0.0111 | 0.0092 | 0.0137 | 0.0015 | 0.6764 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(4096,)` | TensorStudio | 0.0570 | 0.0606 | 0.0562 | 0.0723 | 0.0060 | 0.0529 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0000 | 0.0529 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(4096,)` | TensorStudio | 0.0499 | 0.0508 | 0.0490 | 0.0559 | 0.0026 | 0.2093 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0104 | 0.0107 | 0.0090 | 0.0132 | 0.0015 | 0.2093 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(16384,)` | TensorStudio | 0.1962 | 0.1977 | 0.1911 | 0.2053 | 0.0048 | 0.0316 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0062 | 0.0062 | 0.0062 | 0.0063 | 0.0000 | 0.0316 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(16384,)` | TensorStudio | 0.2016 | 0.2005 | 0.1934 | 0.2115 | 0.0067 | 0.0664 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0134 | 0.0138 | 0.0120 | 0.0167 | 0.0016 | 0.0664 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0909 | 0.0911 | 0.0886 | 0.0944 | 0.0019 | 0.0488 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0044 | 0.0038 | 0.0020 | 0.0049 | 0.0011 | 0.0488 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | TensorStudio | 4.0749 | 4.0557 | 3.9468 | 4.2038 | 0.0914 | 0.0026 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0105 | 0.0105 | 0.0105 | 0.0105 | 0.0000 | 0.0026 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | TensorStudio | 31.4727 | 31.5941 | 30.8843 | 32.3546 | 0.5534 | 0.0066 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2077 | 0.2238 | 0.1890 | 0.2850 | 0.0353 | 0.0066 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | TensorStudio | 255.6362 | 257.4515 | 250.8605 | 264.0951 | 4.7707 | 0.0015 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.3914 | 0.4243 | 0.3204 | 0.5206 | 0.0772 | 0.0015 | n/a | n/a | n/a | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0173 | 0.0174 | 0.0156 | 0.0192 | 0.0012 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0321 | 0.0315 | 0.0290 | 0.0339 | 0.0019 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1629 | 0.1761 | 0.1601 | 0.2295 | 0.0268 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3856 | 0.3803 | 0.3542 | 0.4039 | 0.0191 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.2275 | 1.2383 | 1.1766 | 1.3003 | 0.0459 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.6160 | 2.6149 | 2.5369 | 2.6858 | 0.0482 | n/a | n/a | n/a | n/a | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.6895 | 1.6985 | 1.5573 | 1.8600 | 0.0992 | 0.0850 | n/a | n/a | n/a | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1436 | 0.1459 | 0.1420 | 0.1524 | 0.0041 | 0.0850 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
