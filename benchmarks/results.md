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

- TensorStudio wins versus NumPy: `11`
- TensorStudio losses versus NumPy: `78`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0027 | 0.0002 | 0.2986 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2986 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(1,)` | TensorStudio | 0.0024 | 0.0025 | 0.0023 | 0.0027 | 0.0002 | 0.2769 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2769 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3179 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3179 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(1,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3220 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3220 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0086 | 0.0086 | 0.0084 | 0.0087 | 0.0001 | 0.4568 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0039 | 0.0041 | 0.0037 | 0.0053 | 0.0006 | 0.4568 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3122 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.3122 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3063 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3063 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3190 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3190 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(8,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3238 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.3238 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0085 | 0.0085 | 0.0084 | 0.0087 | 0.0001 | 0.4674 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0040 | 0.0040 | 0.0038 | 0.0041 | 0.0001 | 0.4674 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0021 | 0.0025 | 0.0001 | 0.2817 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0006 | 0.0007 | 0.0006 | 0.0007 | 0.0000 | 0.2817 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(32,)` | TensorStudio | 0.0022 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.3083 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3083 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(32,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0025 | 0.0001 | 0.3167 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3167 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(32,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0023 | 0.0001 | 0.3227 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3227 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0085 | 0.0085 | 0.0085 | 0.0086 | 0.0001 | 0.4626 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0039 | 0.0040 | 0.0039 | 0.0042 | 0.0001 | 0.4626 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(128,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0023 | 0.0001 | 0.2932 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0009 | 0.0001 | 0.2932 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0022 | 0.0025 | 0.0001 | 0.2840 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2840 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3047 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3047 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0026 | 0.0002 | 0.3179 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3179 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0087 | 0.0087 | 0.0087 | 0.0089 | 0.0001 | 0.4498 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0041 | 0.0001 | 0.4498 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(1024,)` | TensorStudio | 0.0028 | 0.0028 | 0.0026 | 0.0029 | 0.0001 | 0.3621 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3621 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0026 | 0.0027 | 0.0026 | 0.0030 | 0.0002 | 0.3469 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0009 | 0.0009 | 0.0009 | 0.0009 | 0.0000 | 0.3469 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0025 | 0.0026 | 0.0025 | 0.0029 | 0.0001 | 0.3761 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0009 | 0.0010 | 0.0009 | 0.0012 | 0.0001 | 0.3761 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(1024,)` | TensorStudio | 0.0029 | 0.0028 | 0.0028 | 0.0029 | 0.0000 | 0.3557 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3557 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0108 | 0.0108 | 0.0107 | 0.0110 | 0.0001 | 0.4988 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0054 | 0.0055 | 0.0052 | 0.0061 | 0.0003 | 0.4988 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(4096,)` | TensorStudio | 0.0041 | 0.0041 | 0.0040 | 0.0041 | 0.0000 | 0.3427 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0018 | 0.0002 | 0.3427 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0041 | 0.0042 | 0.0041 | 0.0046 | 0.0002 | 0.3884 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0014 | 0.0020 | 0.0002 | 0.3884 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0040 | 0.0044 | 0.0040 | 0.0059 | 0.0007 | 0.3808 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.3808 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(4096,)` | TensorStudio | 0.0051 | 0.0052 | 0.0048 | 0.0061 | 0.0005 | 0.3174 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.3174 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0186 | 0.0185 | 0.0183 | 0.0188 | 0.0002 | 0.4458 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0083 | 0.0084 | 0.0079 | 0.0088 | 0.0003 | 0.4458 | n/a | n/a | n/a | NumPy baseline |
| elementwise | add | `(16384,)` | TensorStudio | 0.0100 | 0.0100 | 0.0099 | 0.0103 | 0.0001 | 0.3682 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0039 | 0.0001 | 0.3682 | n/a | n/a | n/a | NumPy baseline |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0088 | 0.0088 | 0.0088 | 0.0089 | 0.0000 | 0.4285 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0038 | 0.0039 | 0.0038 | 0.0042 | 0.0002 | 0.4285 | n/a | n/a | n/a | NumPy baseline |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0089 | 0.0090 | 0.0089 | 0.0093 | 0.0002 | 0.4795 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0043 | 0.0045 | 0.0042 | 0.0053 | 0.0004 | 0.4795 | n/a | n/a | n/a | NumPy baseline |
| elementwise | div | `(16384,)` | TensorStudio | 0.0127 | 0.0135 | 0.0122 | 0.0155 | 0.0013 | 0.3295 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0042 | 0.0043 | 0.0041 | 0.0047 | 0.0002 | 0.3295 | n/a | n/a | n/a | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0426 | 0.0430 | 0.0414 | 0.0442 | 0.0011 | 0.4318 | n/a | n/a | n/a | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0184 | 0.0185 | 0.0183 | 0.0189 | 0.0002 | 0.4318 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(1,)` | TensorStudio | 0.0020 | 0.0020 | 0.0015 | 0.0027 | 0.0004 | 0.7606 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0015 | 0.0015 | 0.0013 | 0.0017 | 0.0001 | 0.7606 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0016 | 0.0017 | 0.0015 | 0.0020 | 0.0002 | 2.2170 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0036 | 0.0035 | 0.0034 | 0.0036 | 0.0001 | 2.2170 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.7607 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0014 | 0.0001 | 0.7607 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(1,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.8335 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0013 | 0.0000 | 0.8335 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0000 | 0.8020 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0013 | 0.0017 | 0.0012 | 0.0027 | 0.0006 | 0.8020 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(8,)` | TensorStudio | 0.0014 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.9750 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.9750 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 2.3229 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0036 | 0.0036 | 0.0034 | 0.0039 | 0.0002 | 2.3229 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(8,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.7821 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0012 | 0.0012 | 0.0000 | 0.7821 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.7932 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.7932 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.8105 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0012 | 0.0015 | 0.0001 | 0.8105 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(32,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 1.0566 | n/a | n/a | n/a | win vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0016 | 0.0016 | 0.0014 | 0.0018 | 0.0001 | 1.0566 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0025 | 0.0024 | 0.0019 | 0.0031 | 0.0004 | 1.7981 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0045 | 0.0046 | 0.0036 | 0.0054 | 0.0007 | 1.7981 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(32,)` | TensorStudio | 0.0021 | 0.0024 | 0.0020 | 0.0037 | 0.0007 | 0.5880 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0018 | 0.0002 | 0.5880 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(32,)` | TensorStudio | 0.0021 | 0.0021 | 0.0018 | 0.0025 | 0.0003 | 0.5832 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0013 | 0.0000 | 0.5832 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0021 | 0.0001 | 0.6945 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0012 | 0.0013 | 0.0012 | 0.0016 | 0.0001 | 0.6945 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(128,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 0.9583 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0015 | 0.0000 | 0.9583 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0025 | 0.0031 | 0.0024 | 0.0052 | 0.0010 | 1.4598 | n/a | n/a | n/a | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0041 | 0.0002 | 1.4598 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(128,)` | TensorStudio | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0000 | 0.5713 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0017 | 0.0001 | 0.5713 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.5928 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0017 | 0.0001 | 0.5928 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0028 | 0.0001 | 0.5656 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 0.5656 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(1024,)` | TensorStudio | 0.0021 | 0.0021 | 0.0021 | 0.0021 | 0.0000 | 0.9402 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0020 | 0.0021 | 0.0019 | 0.0027 | 0.0003 | 0.9402 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0089 | 0.0089 | 0.0088 | 0.0091 | 0.0001 | 0.6493 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0058 | 0.0058 | 0.0057 | 0.0061 | 0.0001 | 0.6493 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(1024,)` | TensorStudio | 0.0134 | 0.0135 | 0.0132 | 0.0142 | 0.0004 | 0.2720 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0000 | 0.2720 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(1024,)` | TensorStudio | 0.0079 | 0.0080 | 0.0078 | 0.0083 | 0.0002 | 0.3266 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0026 | 0.0026 | 0.0027 | 0.0000 | 0.3266 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(1024,)` | TensorStudio | 0.0092 | 0.0103 | 0.0087 | 0.0136 | 0.0019 | 0.3175 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0029 | 0.0030 | 0.0029 | 0.0030 | 0.0000 | 0.3175 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(4096,)` | TensorStudio | 0.0045 | 0.0045 | 0.0044 | 0.0045 | 0.0001 | 0.6832 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0031 | 0.0031 | 0.0031 | 0.0033 | 0.0001 | 0.6832 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0317 | 0.0319 | 0.0312 | 0.0326 | 0.0005 | 0.3446 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0109 | 0.0109 | 0.0104 | 0.0116 | 0.0004 | 0.3446 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(4096,)` | TensorStudio | 0.0584 | 0.0589 | 0.0576 | 0.0616 | 0.0014 | 0.1775 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0104 | 0.0103 | 0.0101 | 0.0105 | 0.0002 | 0.1775 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(4096,)` | TensorStudio | 0.0269 | 0.0270 | 0.0264 | 0.0279 | 0.0005 | 0.2017 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0054 | 0.0057 | 0.0054 | 0.0062 | 0.0003 | 0.2017 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(4096,)` | TensorStudio | 0.0312 | 0.0312 | 0.0297 | 0.0334 | 0.0014 | 0.2385 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0074 | 0.0076 | 0.0072 | 0.0086 | 0.0005 | 0.2385 | n/a | n/a | n/a | NumPy baseline |
| activations | relu | `(16384,)` | TensorStudio | 0.0147 | 0.0150 | 0.0147 | 0.0160 | 0.0005 | 0.5256 | n/a | n/a | n/a | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0078 | 0.0078 | 0.0077 | 0.0078 | 0.0000 | 0.5256 | n/a | n/a | n/a | NumPy baseline |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.1271 | 0.1268 | 0.1205 | 0.1322 | 0.0046 | 0.2316 | n/a | n/a | n/a | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0294 | 0.0296 | 0.0292 | 0.0303 | 0.0004 | 0.2316 | n/a | n/a | n/a | NumPy baseline |
| activations | tanh | `(16384,)` | TensorStudio | 0.2454 | 0.2435 | 0.2345 | 0.2500 | 0.0054 | 0.1615 | n/a | n/a | n/a | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0396 | 0.0397 | 0.0381 | 0.0422 | 0.0015 | 0.1615 | n/a | n/a | n/a | NumPy baseline |
| activations | exp | `(16384,)` | TensorStudio | 0.1055 | 0.1073 | 0.0988 | 0.1151 | 0.0063 | 0.1737 | n/a | n/a | n/a | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0183 | 0.0193 | 0.0183 | 0.0219 | 0.0014 | 0.1737 | n/a | n/a | n/a | NumPy baseline |
| activations | log | `(16384,)` | TensorStudio | 0.1181 | 0.1181 | 0.1103 | 0.1229 | 0.0044 | 0.2035 | n/a | n/a | n/a | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0240 | 0.0244 | 0.0238 | 0.0261 | 0.0009 | 0.2035 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1,)` | TensorStudio | 0.0013 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 1.3348 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 1.3348 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1,)` | TensorStudio | 0.0014 | 0.0014 | 0.0013 | 0.0017 | 0.0002 | 5.8562 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0081 | 0.0087 | 0.0073 | 0.0103 | 0.0013 | 5.8562 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(8,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0021 | 0.0002 | 1.2241 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0019 | 0.0020 | 0.0017 | 0.0022 | 0.0002 | 1.2241 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 5.0005 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0079 | 0.0088 | 0.0075 | 0.0128 | 0.0020 | 5.0005 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(32,)` | TensorStudio | 0.0022 | 0.0022 | 0.0021 | 0.0024 | 0.0001 | 0.8346 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0018 | 0.0019 | 0.0018 | 0.0020 | 0.0001 | 0.8346 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0026 | 0.0002 | 3.6133 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0082 | 0.0081 | 0.0076 | 0.0085 | 0.0003 | 3.6133 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(128,)` | TensorStudio | 0.0050 | 0.0055 | 0.0048 | 0.0066 | 0.0008 | 0.3658 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0018 | 0.0019 | 0.0017 | 0.0021 | 0.0001 | 0.3658 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(128,)` | TensorStudio | 0.0050 | 0.0052 | 0.0048 | 0.0063 | 0.0005 | 1.5891 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0079 | 0.0093 | 0.0077 | 0.0142 | 0.0025 | 1.5891 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1024,)` | TensorStudio | 0.0293 | 0.0312 | 0.0291 | 0.0380 | 0.0034 | 0.0960 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0028 | 0.0026 | 0.0022 | 0.0031 | 0.0004 | 0.0960 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1024,)` | TensorStudio | 0.0298 | 0.0299 | 0.0293 | 0.0310 | 0.0006 | 0.3237 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0097 | 0.0118 | 0.0089 | 0.0189 | 0.0038 | 0.3237 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(4096,)` | TensorStudio | 0.1320 | 0.1314 | 0.1269 | 0.1342 | 0.0027 | 0.0222 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0029 | 0.0032 | 0.0029 | 0.0041 | 0.0005 | 0.0222 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(4096,)` | TensorStudio | 0.1127 | 0.1127 | 0.1114 | 0.1137 | 0.0008 | 0.0780 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0088 | 0.0090 | 0.0084 | 0.0098 | 0.0005 | 0.0780 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(16384,)` | TensorStudio | 0.4395 | 0.4436 | 0.4370 | 0.4621 | 0.0093 | 0.0136 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0060 | 0.0060 | 0.0058 | 0.0062 | 0.0001 | 0.0136 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(16384,)` | TensorStudio | 0.4422 | 0.4420 | 0.4400 | 0.4434 | 0.0012 | 0.0275 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0122 | 0.0127 | 0.0120 | 0.0148 | 0.0010 | 0.0275 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0036 | 0.0036 | 0.0036 | 0.0036 | 0.0000 | 0.5125 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0018 | 0.0018 | 0.0018 | 0.0019 | 0.0000 | 0.5125 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0998 | 0.0888 | 0.0696 | 0.1037 | 0.0157 | 0.1019 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0102 | 0.0102 | 0.0101 | 0.0104 | 0.0001 | 0.1019 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.5097 | 0.5417 | 0.5058 | 0.6347 | 0.0497 | 0.3071 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.1565 | 0.1586 | 0.1511 | 0.1685 | 0.0058 | 0.3071 | n/a | n/a | n/a | NumPy baseline |
| matmul | matmul | `(256, 256)` | TensorStudio | 4.0247 | 4.1022 | 3.9061 | 4.3467 | 0.1919 | 0.1001 | n/a | n/a | n/a | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4028 | 0.4202 | 0.3744 | 0.4762 | 0.0384 | 0.1001 | n/a | n/a | n/a | NumPy baseline |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0146 | 0.0143 | 0.0130 | 0.0158 | 0.0009 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0313 | 0.0320 | 0.0296 | 0.0369 | 0.0027 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1595 | 0.1626 | 0.1540 | 0.1813 | 0.0097 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3438 | 0.3691 | 0.3432 | 0.4583 | 0.0449 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.1494 | 1.1570 | 1.1326 | 1.2113 | 0.0283 | n/a | n/a | n/a | n/a | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.5540 | 2.5624 | 2.5125 | 2.6198 | 0.0352 | n/a | n/a | n/a | n/a | no NumPy reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.5001 | 1.5055 | 1.4532 | 1.5548 | 0.0330 | 0.1053 | n/a | n/a | n/a | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1580 | 0.1739 | 0.1408 | 0.2563 | 0.0429 | 0.1053 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
