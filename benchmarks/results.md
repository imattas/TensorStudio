# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.0rc1`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `14`
- TensorStudio losses versus NumPy: `75`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0027 | 0.0028 | 0.0021 | 0.0042 | 0.0007 | 0.2572 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.2572 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(1,)` | TensorStudio | 0.0025 | 0.0028 | 0.0022 | 0.0040 | 0.0007 | 0.3711 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0009 | 0.0009 | 0.0008 | 0.0012 | 0.0001 | 0.3711 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(1,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.2876 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0010 | 0.0007 | 0.0020 | 0.0005 | 0.2876 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(1,)` | TensorStudio | 0.0027 | 0.0028 | 0.0024 | 0.0033 | 0.0003 | 0.3534 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0010 | 0.0011 | 0.0007 | 0.0019 | 0.0004 | 0.3534 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0104 | 0.0105 | 0.0091 | 0.0118 | 0.0011 | 0.4464 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0046 | 0.0052 | 0.0039 | 0.0085 | 0.0017 | 0.4464 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(8,)` | TensorStudio | 0.0028 | 0.0029 | 0.0024 | 0.0038 | 0.0005 | 0.6718 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0019 | 0.0016 | 0.0008 | 0.0021 | 0.0005 | 0.6718 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(8,)` | TensorStudio | 0.0032 | 0.0030 | 0.0026 | 0.0034 | 0.0003 | 0.2276 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.2276 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(8,)` | TensorStudio | 0.0028 | 0.0031 | 0.0025 | 0.0040 | 0.0005 | 0.5754 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0016 | 0.0018 | 0.0009 | 0.0029 | 0.0007 | 0.5754 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(8,)` | TensorStudio | 0.0032 | 0.0031 | 0.0025 | 0.0034 | 0.0003 | 0.2631 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0008 | 0.0011 | 0.0008 | 0.0021 | 0.0005 | 0.2631 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0099 | 0.0105 | 0.0095 | 0.0120 | 0.0010 | 0.5629 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0056 | 0.0058 | 0.0048 | 0.0078 | 0.0011 | 0.5629 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(32,)` | TensorStudio | 0.0037 | 0.0037 | 0.0029 | 0.0048 | 0.0007 | 0.2602 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0010 | 0.0000 | 0.2602 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(32,)` | TensorStudio | 0.0043 | 0.0041 | 0.0031 | 0.0056 | 0.0009 | 0.2892 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0013 | 0.0012 | 0.0008 | 0.0019 | 0.0004 | 0.2892 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(32,)` | TensorStudio | 0.0033 | 0.0035 | 0.0029 | 0.0047 | 0.0006 | 0.2344 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.2344 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(32,)` | TensorStudio | 0.0038 | 0.0043 | 0.0032 | 0.0067 | 0.0012 | 0.2044 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0009 | 0.0001 | 0.2044 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0123 | 0.0123 | 0.0109 | 0.0141 | 0.0011 | 0.3919 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0048 | 0.0049 | 0.0046 | 0.0057 | 0.0004 | 0.3919 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(128,)` | TensorStudio | 0.0053 | 0.0055 | 0.0046 | 0.0067 | 0.0008 | 0.1591 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0010 | 0.0001 | 0.1591 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(128,)` | TensorStudio | 0.0051 | 0.0052 | 0.0047 | 0.0057 | 0.0003 | 0.1471 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.1471 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(128,)` | TensorStudio | 0.0053 | 0.0053 | 0.0047 | 0.0058 | 0.0004 | 0.2418 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0013 | 0.0014 | 0.0008 | 0.0021 | 0.0005 | 0.2418 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(128,)` | TensorStudio | 0.0052 | 0.0054 | 0.0049 | 0.0059 | 0.0005 | 0.1777 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0009 | 0.0011 | 0.0008 | 0.0018 | 0.0003 | 0.1777 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0172 | 0.0175 | 0.0165 | 0.0186 | 0.0008 | 0.2932 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0050 | 0.0055 | 0.0044 | 0.0066 | 0.0008 | 0.2932 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(1024,)` | TensorStudio | 0.0211 | 0.0216 | 0.0206 | 0.0237 | 0.0011 | 0.0471 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0014 | 0.0002 | 0.0471 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0223 | 0.0225 | 0.0202 | 0.0249 | 0.0017 | 0.0463 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0012 | 0.0001 | 0.0463 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0223 | 0.0229 | 0.0212 | 0.0254 | 0.0014 | 0.0444 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0011 | 0.0010 | 0.0013 | 0.0001 | 0.0444 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(1024,)` | TensorStudio | 0.0234 | 0.0226 | 0.0210 | 0.0238 | 0.0012 | 0.0440 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.0440 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0768 | 0.0765 | 0.0728 | 0.0799 | 0.0024 | 0.0764 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0059 | 0.0071 | 0.0055 | 0.0114 | 0.0022 | 0.0764 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(4096,)` | TensorStudio | 0.0805 | 0.0826 | 0.0766 | 0.0979 | 0.0079 | 0.0209 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0017 | 0.0017 | 0.0016 | 0.0017 | 0.0000 | 0.0209 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0810 | 0.0900 | 0.0780 | 0.1074 | 0.0126 | 0.0316 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0026 | 0.0026 | 0.0021 | 0.0033 | 0.0004 | 0.0316 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(4096,)` | TensorStudio | 0.1048 | 0.1061 | 0.0969 | 0.1155 | 0.0068 | 0.0190 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0020 | 0.0019 | 0.0014 | 0.0026 | 0.0004 | 0.0190 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(4096,)` | TensorStudio | 0.1061 | 0.1065 | 0.0926 | 0.1222 | 0.0095 | 0.0342 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0037 | 0.0000 | 0.0342 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.3731 | 0.3685 | 0.3441 | 0.3998 | 0.0214 | 0.0305 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0114 | 0.0120 | 0.0097 | 0.0160 | 0.0022 | 0.0305 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(16384,)` | TensorStudio | 0.4004 | 0.3914 | 0.3452 | 0.4528 | 0.0394 | 0.0199 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0080 | 0.0069 | 0.0037 | 0.0090 | 0.0019 | 0.0199 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(16384,)` | TensorStudio | 0.3938 | 0.3952 | 0.3656 | 0.4381 | 0.0261 | 0.0125 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0049 | 0.0051 | 0.0046 | 0.0063 | 0.0006 | 0.0125 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(16384,)` | TensorStudio | 0.4298 | 0.4316 | 0.4210 | 0.4494 | 0.0106 | 0.0138 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0059 | 0.0055 | 0.0045 | 0.0065 | 0.0008 | 0.0138 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(16384,)` | TensorStudio | 0.4200 | 0.4071 | 0.3526 | 0.4381 | 0.0296 | 0.0325 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0136 | 0.0139 | 0.0105 | 0.0171 | 0.0022 | 0.0325 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 1.3524 | 1.3299 | 1.2393 | 1.4101 | 0.0590 | 0.0197 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0267 | 0.0292 | 0.0198 | 0.0434 | 0.0087 | 0.0197 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(1,)` | TensorStudio | 0.0024 | 0.0025 | 0.0021 | 0.0031 | 0.0003 | 1.1917 | n/a | n/a | n/a | win vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0029 | 0.0029 | 0.0023 | 0.0037 | 0.0006 | 1.1917 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0021 | 0.0022 | 0.0019 | 0.0026 | 0.0003 | 2.8681 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0059 | 0.0055 | 0.0037 | 0.0067 | 0.0012 | 2.8681 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(1,)` | TensorStudio | 0.0027 | 0.0026 | 0.0017 | 0.0031 | 0.0005 | 1.1506 | n/a | n/a | n/a | win vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0031 | 0.0029 | 0.0019 | 0.0037 | 0.0007 | 1.1506 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(1,)` | TensorStudio | 0.0020 | 0.0019 | 0.0014 | 0.0028 | 0.0005 | 1.3072 | n/a | n/a | n/a | win vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0026 | 0.0026 | 0.0018 | 0.0036 | 0.0006 | 1.3072 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(1,)` | TensorStudio | 0.0034 | 0.0031 | 0.0020 | 0.0038 | 0.0007 | 0.8913 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0030 | 0.0027 | 0.0013 | 0.0032 | 0.0007 | 0.8913 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(8,)` | TensorStudio | 0.0032 | 0.0031 | 0.0020 | 0.0037 | 0.0006 | 0.7161 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0023 | 0.0025 | 0.0014 | 0.0038 | 0.0008 | 0.7161 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0020 | 0.0021 | 0.0016 | 0.0027 | 0.0004 | 3.9838 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0079 | 0.0077 | 0.0049 | 0.0096 | 0.0017 | 3.9838 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(8,)` | TensorStudio | 0.0020 | 0.0022 | 0.0016 | 0.0028 | 0.0004 | 0.8679 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0018 | 0.0020 | 0.0013 | 0.0034 | 0.0008 | 0.8679 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(8,)` | TensorStudio | 0.0024 | 0.0022 | 0.0017 | 0.0027 | 0.0004 | 0.9591 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0023 | 0.0024 | 0.0016 | 0.0033 | 0.0005 | 0.9591 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(8,)` | TensorStudio | 0.0022 | 0.0025 | 0.0017 | 0.0038 | 0.0007 | 0.8078 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0018 | 0.0020 | 0.0016 | 0.0028 | 0.0004 | 0.8078 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(32,)` | TensorStudio | 0.0026 | 0.0027 | 0.0021 | 0.0035 | 0.0005 | 1.1175 | n/a | n/a | n/a | win vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0029 | 0.0027 | 0.0015 | 0.0035 | 0.0008 | 1.1175 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0032 | 0.0035 | 0.0026 | 0.0046 | 0.0007 | 2.3370 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0074 | 0.0076 | 0.0058 | 0.0097 | 0.0014 | 2.3370 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(32,)` | TensorStudio | 0.0040 | 0.0038 | 0.0032 | 0.0042 | 0.0004 | 0.4608 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0019 | 0.0021 | 0.0016 | 0.0030 | 0.0005 | 0.4608 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(32,)` | TensorStudio | 0.0035 | 0.0034 | 0.0024 | 0.0043 | 0.0007 | 0.7645 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0027 | 0.0027 | 0.0018 | 0.0035 | 0.0006 | 0.7645 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(32,)` | TensorStudio | 0.0030 | 0.0030 | 0.0025 | 0.0037 | 0.0004 | 0.7808 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0024 | 0.0021 | 0.0015 | 0.0025 | 0.0004 | 0.7808 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(128,)` | TensorStudio | 0.0046 | 0.0046 | 0.0037 | 0.0056 | 0.0006 | 0.7249 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0033 | 0.0033 | 0.0025 | 0.0038 | 0.0005 | 0.7249 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0064 | 0.0065 | 0.0057 | 0.0078 | 0.0007 | 0.9464 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0060 | 0.0063 | 0.0054 | 0.0075 | 0.0008 | 0.9464 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(128,)` | TensorStudio | 0.0071 | 0.0072 | 0.0060 | 0.0083 | 0.0009 | 0.4161 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0030 | 0.0031 | 0.0024 | 0.0038 | 0.0005 | 0.4161 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(128,)` | TensorStudio | 0.0064 | 0.0062 | 0.0049 | 0.0076 | 0.0009 | 0.3728 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0024 | 0.0027 | 0.0016 | 0.0036 | 0.0008 | 0.3728 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(128,)` | TensorStudio | 0.0052 | 0.0058 | 0.0042 | 0.0076 | 0.0014 | 0.6373 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0033 | 0.0032 | 0.0026 | 0.0034 | 0.0003 | 0.6373 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(1024,)` | TensorStudio | 0.0183 | 0.0210 | 0.0168 | 0.0273 | 0.0042 | 0.2023 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0037 | 0.0034 | 0.0020 | 0.0046 | 0.0011 | 0.2023 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0301 | 0.0295 | 0.0278 | 0.0304 | 0.0010 | 0.3655 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0110 | 0.0108 | 0.0073 | 0.0157 | 0.0028 | 0.3655 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(1024,)` | TensorStudio | 0.0412 | 0.0419 | 0.0398 | 0.0465 | 0.0024 | 0.1149 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0047 | 0.0050 | 0.0039 | 0.0059 | 0.0008 | 0.1149 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(1024,)` | TensorStudio | 0.0356 | 0.0389 | 0.0221 | 0.0725 | 0.0180 | 0.2055 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0073 | 0.0078 | 0.0071 | 0.0091 | 0.0007 | 0.2055 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(1024,)` | TensorStudio | 0.0385 | 0.0393 | 0.0273 | 0.0545 | 0.0088 | 0.0947 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0039 | 0.0001 | 0.0947 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(4096,)` | TensorStudio | 0.1019 | 0.0948 | 0.0723 | 0.1112 | 0.0143 | 0.0345 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0035 | 0.0037 | 0.0031 | 0.0044 | 0.0005 | 0.0345 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.1296 | 0.1323 | 0.1214 | 0.1490 | 0.0096 | 0.1490 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0193 | 0.0193 | 0.0148 | 0.0230 | 0.0027 | 0.1490 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(4096,)` | TensorStudio | 0.1659 | 0.1656 | 0.1541 | 0.1761 | 0.0094 | 0.0993 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0165 | 0.0166 | 0.0142 | 0.0190 | 0.0015 | 0.0993 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(4096,)` | TensorStudio | 0.1112 | 0.1113 | 0.0960 | 0.1232 | 0.0088 | 0.0763 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0085 | 0.0092 | 0.0077 | 0.0111 | 0.0013 | 0.0763 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(4096,)` | TensorStudio | 0.1054 | 0.1025 | 0.0942 | 0.1116 | 0.0070 | 0.1141 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0120 | 0.0126 | 0.0084 | 0.0164 | 0.0029 | 0.1141 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(16384,)` | TensorStudio | 0.2782 | 0.2776 | 0.2631 | 0.2884 | 0.0083 | 0.0308 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0086 | 0.0098 | 0.0079 | 0.0151 | 0.0027 | 0.0308 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.4516 | 0.4499 | 0.4114 | 0.4829 | 0.0241 | 0.0769 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0347 | 0.0401 | 0.0319 | 0.0511 | 0.0087 | 0.0769 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(16384,)` | TensorStudio | 0.6521 | 0.6442 | 0.5956 | 0.6822 | 0.0297 | 0.0895 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0584 | 0.0558 | 0.0422 | 0.0652 | 0.0078 | 0.0895 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(16384,)` | TensorStudio | 0.4104 | 0.4062 | 0.3772 | 0.4293 | 0.0168 | 0.0774 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0318 | 0.0329 | 0.0253 | 0.0385 | 0.0047 | 0.0774 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(16384,)` | TensorStudio | 0.4045 | 0.3836 | 0.3274 | 0.4340 | 0.0402 | 0.0701 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0284 | 0.0319 | 0.0272 | 0.0383 | 0.0052 | 0.0701 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1,)` | TensorStudio | 0.0018 | 0.0017 | 0.0013 | 0.0021 | 0.0003 | 1.4376 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0025 | 0.0024 | 0.0017 | 0.0031 | 0.0005 | 1.4376 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1,)` | TensorStudio | 0.0023 | 0.0025 | 0.0015 | 0.0037 | 0.0009 | 6.1703 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0142 | 0.0154 | 0.0094 | 0.0226 | 0.0054 | 6.1703 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(8,)` | TensorStudio | 0.0019 | 0.0017 | 0.0014 | 0.0020 | 0.0002 | 1.7447 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0033 | 0.0035 | 0.0027 | 0.0046 | 0.0007 | 1.7447 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(8,)` | TensorStudio | 0.0020 | 0.0020 | 0.0017 | 0.0023 | 0.0002 | 5.9456 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0116 | 0.0114 | 0.0085 | 0.0133 | 0.0016 | 5.9456 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(32,)` | TensorStudio | 0.0022 | 0.0026 | 0.0016 | 0.0038 | 0.0008 | 1.5693 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0034 | 0.0032 | 0.0020 | 0.0040 | 0.0008 | 1.5693 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(32,)` | TensorStudio | 0.0026 | 0.0029 | 0.0019 | 0.0043 | 0.0009 | 5.2469 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0138 | 0.0131 | 0.0111 | 0.0149 | 0.0014 | 5.2469 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(128,)` | TensorStudio | 0.0042 | 0.0043 | 0.0033 | 0.0057 | 0.0009 | 0.6398 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0027 | 0.0029 | 0.0017 | 0.0043 | 0.0008 | 0.6398 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(128,)` | TensorStudio | 0.0041 | 0.0039 | 0.0031 | 0.0048 | 0.0006 | 3.1699 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0129 | 0.0130 | 0.0113 | 0.0143 | 0.0010 | 3.1699 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1024,)` | TensorStudio | 0.0123 | 0.0135 | 0.0112 | 0.0184 | 0.0026 | 0.2192 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0027 | 0.0027 | 0.0021 | 0.0034 | 0.0005 | 0.2192 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1024,)` | TensorStudio | 0.0162 | 0.0162 | 0.0148 | 0.0181 | 0.0012 | 0.9849 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0160 | 0.0153 | 0.0116 | 0.0193 | 0.0028 | 0.9849 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(4096,)` | TensorStudio | 0.0580 | 0.0572 | 0.0542 | 0.0598 | 0.0023 | 0.0983 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0057 | 0.0055 | 0.0039 | 0.0065 | 0.0009 | 0.0983 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(4096,)` | TensorStudio | 0.0545 | 0.0542 | 0.0518 | 0.0567 | 0.0016 | 0.1989 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0108 | 0.0151 | 0.0104 | 0.0220 | 0.0056 | 0.1989 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(16384,)` | TensorStudio | 0.1959 | 0.1942 | 0.1578 | 0.2146 | 0.0206 | 0.0628 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0123 | 0.0124 | 0.0120 | 0.0133 | 0.0005 | 0.0628 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(16384,)` | TensorStudio | 0.2117 | 0.2090 | 0.1744 | 0.2287 | 0.0186 | 0.1327 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0281 | 0.0279 | 0.0259 | 0.0301 | 0.0014 | 0.1327 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0981 | 0.0904 | 0.0683 | 0.1085 | 0.0154 | 0.0250 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0024 | 0.0029 | 0.0021 | 0.0046 | 0.0009 | 0.0250 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | TensorStudio | 3.6827 | 3.6899 | 3.5308 | 3.9782 | 0.1625 | 0.0050 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0184 | 0.0178 | 0.0148 | 0.0206 | 0.0021 | 0.0050 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | TensorStudio | 27.8409 | 28.3092 | 25.0195 | 31.8084 | 2.2378 | 0.0064 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1768 | 0.1724 | 0.1583 | 0.1823 | 0.0100 | 0.0064 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | TensorStudio | 216.6774 | 221.7577 | 211.8041 | 242.6430 | 11.0279 | 0.0019 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4039 | 0.4179 | 0.3282 | 0.5662 | 0.0827 | 0.0019 | n/a | n/a | n/a | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0143 | 0.0151 | 0.0131 | 0.0180 | 0.0017 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0514 | 0.0758 | 0.0316 | 0.1974 | 0.0614 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.2916 | 0.2891 | 0.1994 | 0.3943 | 0.0665 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.4099 | 0.4341 | 0.3855 | 0.5403 | 0.0571 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.7491 | 1.6861 | 1.4853 | 1.8745 | 0.1509 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 3.7885 | 3.7370 | 3.5346 | 3.8192 | 0.1052 | n/a | n/a | n/a | n/a | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 2.3466 | 2.4081 | 1.8871 | 2.7451 | 0.3145 | 0.1347 | n/a | n/a | n/a | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.3160 | 0.3172 | 0.1474 | 0.4415 | 0.0977 | 0.1347 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
