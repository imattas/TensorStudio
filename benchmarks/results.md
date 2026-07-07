# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.0.1`
- NumPy: `2.2.6`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: available (2.12.1+cpu)
- JAX CPU dispatch: unavailable (not installed)

## Summary

- TensorStudio wins versus NumPy: `13`
- TensorStudio losses versus NumPy: `76`
- TensorStudio wins versus PyTorch CPU: `71`
- TensorStudio losses versus PyTorch CPU: `23`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.3058 | n/a | 1.5355 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3058 | n/a | 1.5355 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | PyTorch CPU | 0.0035 | 0.0035 | 0.0035 | 0.0036 | 0.0000 | 0.3058 | n/a | 1.5355 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.3027 | n/a | 1.5383 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3027 | n/a | 1.5383 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0035 | 0.0040 | 0.0002 | 0.3027 | n/a | 1.5383 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0026 | 0.0001 | 0.3215 | n/a | 1.5291 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3215 | n/a | 1.5291 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | PyTorch CPU | 0.0034 | 0.0035 | 0.0034 | 0.0035 | 0.0001 | 0.3215 | n/a | 1.5291 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.3143 | n/a | 1.5286 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3143 | n/a | 1.5286 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0035 | 0.0039 | 0.0001 | 0.3143 | n/a | 1.5286 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0084 | 0.0085 | 0.0083 | 0.0087 | 0.0001 | 0.4545 | n/a | 6.7278 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0038 | 0.0038 | 0.0038 | 0.0038 | 0.0000 | 0.4545 | n/a | 6.7278 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | PyTorch CPU | 0.0567 | 0.0573 | 0.0558 | 0.0604 | 0.0016 | 0.4545 | n/a | 6.7278 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0023 | 0.0030 | 0.0023 | 0.0058 | 0.0014 | 0.2950 | n/a | 1.5542 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2950 | n/a | 1.5542 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0034 | 0.0039 | 0.0002 | 0.2950 | n/a | 1.5542 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0023 | 0.0026 | 0.0023 | 0.0036 | 0.0005 | 0.9722 | n/a | 2.1344 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0022 | 0.0022 | 0.0021 | 0.0022 | 0.0000 | 0.9722 | n/a | 2.1344 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | PyTorch CPU | 0.0049 | 0.0073 | 0.0036 | 0.0121 | 0.0039 | 0.9722 | n/a | 2.1344 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0023 | 0.0000 | 0.3361 | n/a | 1.5391 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3361 | n/a | 1.5391 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | PyTorch CPU | 0.0034 | 0.0055 | 0.0034 | 0.0089 | 0.0026 | 0.3361 | n/a | 1.5391 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0023 | 0.0023 | 0.0022 | 0.0026 | 0.0001 | 0.3430 | n/a | 1.6369 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0010 | 0.0001 | 0.3430 | n/a | 1.6369 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | PyTorch CPU | 0.0037 | 0.0047 | 0.0034 | 0.0080 | 0.0017 | 0.3430 | n/a | 1.6369 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0084 | 0.0084 | 0.0082 | 0.0087 | 0.0002 | 1.5663 | n/a | 6.7659 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0131 | 0.0102 | 0.0038 | 0.0132 | 0.0038 | 1.5663 | n/a | 6.7659 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | PyTorch CPU | 0.0566 | 0.0565 | 0.0559 | 0.0572 | 0.0005 | 1.5663 | n/a | 6.7659 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.3027 | n/a | 1.5081 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3027 | n/a | 1.5081 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0042 | 0.0003 | 0.3027 | n/a | 1.5081 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0024 | 0.0000 | 0.3146 | n/a | 1.7389 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0008 | 0.0000 | 0.3146 | n/a | 1.7389 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | PyTorch CPU | 0.0039 | 0.0040 | 0.0035 | 0.0045 | 0.0004 | 0.3146 | n/a | 1.7389 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0025 | 0.0001 | 0.3183 | n/a | 1.4792 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3183 | n/a | 1.4792 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0034 | 0.0000 | 0.3183 | n/a | 1.4792 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.3371 | n/a | 1.4958 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3371 | n/a | 1.4958 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | PyTorch CPU | 0.0034 | 0.0034 | 0.0033 | 0.0036 | 0.0001 | 0.3371 | n/a | 1.4958 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0084 | 0.0084 | 0.0082 | 0.0086 | 0.0001 | 0.4512 | n/a | 6.8504 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0040 | 0.0001 | 0.4512 | n/a | 6.8504 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | PyTorch CPU | 0.0574 | 0.0579 | 0.0557 | 0.0624 | 0.0023 | 0.4512 | n/a | 6.8504 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0027 | 0.0001 | 0.2943 | n/a | 1.4947 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2943 | n/a | 1.4947 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0039 | 0.0002 | 0.2943 | n/a | 1.4947 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0000 | 0.2984 | n/a | 1.4802 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.2984 | n/a | 1.4802 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | PyTorch CPU | 0.0036 | 0.0036 | 0.0035 | 0.0039 | 0.0001 | 0.2984 | n/a | 1.4802 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0023 | 0.0023 | 0.0023 | 0.0023 | 0.0000 | 0.3186 | n/a | 1.5171 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0007 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.3186 | n/a | 1.5171 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0034 | 0.0037 | 0.0001 | 0.3186 | n/a | 1.5171 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0024 | 0.0000 | 0.3106 | n/a | 1.4542 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0007 | 0.0007 | 0.0007 | 0.0007 | 0.0000 | 0.3106 | n/a | 1.4542 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | PyTorch CPU | 0.0035 | 0.0036 | 0.0035 | 0.0038 | 0.0001 | 0.3106 | n/a | 1.4542 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0086 | 0.0088 | 0.0085 | 0.0096 | 0.0004 | 0.4488 | n/a | 6.5977 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0039 | 0.0039 | 0.0038 | 0.0039 | 0.0000 | 0.4488 | n/a | 6.5977 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | PyTorch CPU | 0.0567 | 0.0567 | 0.0562 | 0.0572 | 0.0003 | 0.4488 | n/a | 6.5977 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0030 | 0.0029 | 0.0028 | 0.0030 | 0.0001 | 0.3661 | n/a | 1.2947 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0009 | 0.0013 | 0.0001 | 0.3661 | n/a | 1.2947 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | PyTorch CPU | 0.0038 | 0.0038 | 0.0038 | 0.0039 | 0.0000 | 0.3661 | n/a | 1.2947 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0028 | 0.0029 | 0.0028 | 0.0031 | 0.0001 | 0.3470 | n/a | 1.3807 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0010 | 0.0000 | 0.3470 | n/a | 1.3807 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | PyTorch CPU | 0.0038 | 0.0039 | 0.0038 | 0.0045 | 0.0003 | 0.3470 | n/a | 1.3807 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0027 | 0.0027 | 0.0027 | 0.0028 | 0.0001 | 0.3748 | n/a | 1.3884 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0010 | 0.0010 | 0.0010 | 0.0010 | 0.0000 | 0.3748 | n/a | 1.3884 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | PyTorch CPU | 0.0037 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 0.3748 | n/a | 1.3884 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0000 | 0.4720 | n/a | 1.2626 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0014 | 0.0013 | 0.0011 | 0.0015 | 0.0002 | 0.4720 | n/a | 1.2626 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | PyTorch CPU | 0.0038 | 0.0038 | 0.0037 | 0.0038 | 0.0001 | 0.4720 | n/a | 1.2626 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0104 | 0.0104 | 0.0102 | 0.0107 | 0.0002 | 0.5479 | n/a | 5.4353 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0057 | 0.0057 | 0.0056 | 0.0061 | 0.0002 | 0.5479 | n/a | 5.4353 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | PyTorch CPU | 0.0566 | 0.0568 | 0.0564 | 0.0572 | 0.0004 | 0.5479 | n/a | 5.4353 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0044 | 0.0044 | 0.0044 | 0.0045 | 0.0000 | 0.3358 | n/a | 1.0370 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0014 | 0.0016 | 0.0001 | 0.3358 | n/a | 1.0370 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | PyTorch CPU | 0.0045 | 0.0045 | 0.0045 | 0.0046 | 0.0000 | 0.3358 | n/a | 1.0370 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0044 | 0.0044 | 0.0043 | 0.0045 | 0.0001 | 0.3274 | n/a | 1.0682 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0017 | 0.0001 | 0.3274 | n/a | 1.0682 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | PyTorch CPU | 0.0047 | 0.0047 | 0.0047 | 0.0048 | 0.0001 | 0.3274 | n/a | 1.0682 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0043 | 0.0043 | 0.0042 | 0.0043 | 0.0000 | 0.3433 | n/a | 1.0249 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.3433 | n/a | 1.0249 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | PyTorch CPU | 0.0044 | 0.0044 | 0.0043 | 0.0046 | 0.0001 | 0.3433 | n/a | 1.0249 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0050 | 0.0051 | 0.0050 | 0.0052 | 0.0001 | 0.3256 | n/a | 0.9052 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.3256 | n/a | 0.9052 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | PyTorch CPU | 0.0045 | 0.0047 | 0.0043 | 0.0055 | 0.0005 | 0.3256 | n/a | 0.9052 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0181 | 0.0182 | 0.0177 | 0.0187 | 0.0003 | 0.4555 | n/a | 3.3225 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0082 | 0.0083 | 0.0081 | 0.0086 | 0.0002 | 0.4555 | n/a | 3.3225 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | PyTorch CPU | 0.0601 | 0.0602 | 0.0595 | 0.0617 | 0.0008 | 0.4555 | n/a | 3.3225 | n/a | no | n/a | yes | n/a | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0103 | 0.0103 | 0.0103 | 0.0104 | 0.0000 | 0.3406 | n/a | 0.7126 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0035 | 0.0035 | 0.0035 | 0.0035 | 0.0000 | 0.3406 | n/a | 0.7126 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | PyTorch CPU | 0.0073 | 0.0074 | 0.0073 | 0.0076 | 0.0001 | 0.3406 | n/a | 0.7126 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0103 | 0.0106 | 0.0100 | 0.0120 | 0.0007 | 0.3550 | n/a | 0.7356 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0037 | 0.0037 | 0.0037 | 0.0039 | 0.0001 | 0.3550 | n/a | 0.7356 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | PyTorch CPU | 0.0076 | 0.0076 | 0.0075 | 0.0079 | 0.0001 | 0.3550 | n/a | 0.7356 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0094 | 0.0096 | 0.0094 | 0.0101 | 0.0003 | 0.4154 | n/a | 0.7713 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0039 | 0.0039 | 0.0039 | 0.0040 | 0.0000 | 0.4154 | n/a | 0.7713 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | PyTorch CPU | 0.0073 | 0.0073 | 0.0072 | 0.0074 | 0.0000 | 0.4154 | n/a | 0.7713 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0128 | 0.0128 | 0.0128 | 0.0129 | 0.0001 | 0.3672 | n/a | 0.5639 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0047 | 0.0046 | 0.0041 | 0.0053 | 0.0004 | 0.3672 | n/a | 0.5639 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | PyTorch CPU | 0.0072 | 0.0078 | 0.0072 | 0.0088 | 0.0008 | 0.3672 | n/a | 0.5639 | n/a | no | n/a | no | n/a | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0472 | 0.0471 | 0.0464 | 0.0477 | 0.0004 | 0.3937 | n/a | 1.4671 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0186 | 0.0188 | 0.0185 | 0.0195 | 0.0004 | 0.3937 | n/a | 1.4671 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | PyTorch CPU | 0.0692 | 0.0696 | 0.0679 | 0.0716 | 0.0013 | 0.3937 | n/a | 1.4671 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.9205 | n/a | 14.7032 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0015 | 0.0000 | 0.9205 | n/a | 14.7032 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1,)` | PyTorch CPU | 0.0215 | 0.0215 | 0.0214 | 0.0217 | 0.0001 | 0.9205 | n/a | 14.7032 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 2.3812 | n/a | 38.2361 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(1,)` | NumPy | 0.0036 | 0.0036 | 0.0036 | 0.0038 | 0.0001 | 2.3812 | n/a | 38.2361 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(1,)` | PyTorch CPU | 0.0583 | 0.0590 | 0.0578 | 0.0617 | 0.0014 | 2.3812 | n/a | 38.2361 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(1,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0020 | 0.0002 | 0.9994 | n/a | 14.4346 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1,)` | NumPy | 0.0015 | 0.0015 | 0.0013 | 0.0019 | 0.0002 | 0.9994 | n/a | 14.4346 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1,)` | PyTorch CPU | 0.0220 | 0.0220 | 0.0217 | 0.0222 | 0.0002 | 0.9994 | n/a | 14.4346 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 0.8357 | n/a | 14.0936 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.8357 | n/a | 14.0936 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1,)` | PyTorch CPU | 0.0218 | 0.0217 | 0.0215 | 0.0218 | 0.0001 | 0.8357 | n/a | 14.0936 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 0.8598 | n/a | 13.7699 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 0.8598 | n/a | 13.7699 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0217 | 0.0221 | 0.0002 | 0.8598 | n/a | 13.7699 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(8,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0015 | 0.0000 | 0.9678 | n/a | 14.5779 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(8,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.9678 | n/a | 14.5779 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(8,)` | PyTorch CPU | 0.0215 | 0.0215 | 0.0214 | 0.0217 | 0.0001 | 0.9678 | n/a | 14.5779 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0019 | 0.0001 | 2.3228 | n/a | 37.6984 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(8,)` | NumPy | 0.0036 | 0.0036 | 0.0035 | 0.0038 | 0.0001 | 2.3228 | n/a | 37.6984 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(8,)` | PyTorch CPU | 0.0585 | 0.0583 | 0.0577 | 0.0586 | 0.0003 | 2.3228 | n/a | 37.6984 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0001 | 0.8201 | n/a | 13.6486 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.8201 | n/a | 13.6486 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(8,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0216 | 0.0219 | 0.0001 | 0.8201 | n/a | 13.6486 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0001 | 0.7981 | n/a | 13.7008 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(8,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.7981 | n/a | 13.7008 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(8,)` | PyTorch CPU | 0.0218 | 0.0218 | 0.0216 | 0.0219 | 0.0001 | 0.7981 | n/a | 13.7008 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(8,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.8394 | n/a | 13.7200 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(8,)` | NumPy | 0.0013 | 0.0014 | 0.0013 | 0.0014 | 0.0000 | 0.8394 | n/a | 13.7200 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(8,)` | PyTorch CPU | 0.0221 | 0.0233 | 0.0218 | 0.0265 | 0.0019 | 0.8394 | n/a | 13.7200 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(32,)` | TensorStudio | 0.0020 | 0.0022 | 0.0019 | 0.0027 | 0.0003 | 0.8404 | n/a | 11.1212 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(32,)` | NumPy | 0.0017 | 0.0016 | 0.0014 | 0.0018 | 0.0001 | 0.8404 | n/a | 11.1212 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(32,)` | PyTorch CPU | 0.0218 | 0.0222 | 0.0215 | 0.0234 | 0.0007 | 0.8404 | n/a | 11.1212 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0021 | 0.0001 | 2.0414 | n/a | 32.2110 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(32,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0039 | 0.0001 | 2.0414 | n/a | 32.2110 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(32,)` | PyTorch CPU | 0.0575 | 0.0577 | 0.0573 | 0.0586 | 0.0004 | 2.0414 | n/a | 32.2110 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(32,)` | TensorStudio | 0.0019 | 0.0019 | 0.0018 | 0.0021 | 0.0001 | 0.7268 | n/a | 11.9268 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0014 | 0.0000 | 0.7268 | n/a | 11.9268 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(32,)` | PyTorch CPU | 0.0221 | 0.0220 | 0.0216 | 0.0225 | 0.0003 | 0.7268 | n/a | 11.9268 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0017 | 0.0019 | 0.0001 | 0.7450 | n/a | 12.2009 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(32,)` | NumPy | 0.0013 | 0.0013 | 0.0013 | 0.0013 | 0.0000 | 0.7450 | n/a | 12.2009 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(32,)` | PyTorch CPU | 0.0217 | 0.0217 | 0.0216 | 0.0218 | 0.0000 | 0.7450 | n/a | 12.2009 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(32,)` | TensorStudio | 0.0018 | 0.0018 | 0.0018 | 0.0019 | 0.0000 | 0.7883 | n/a | 12.3266 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 0.7883 | n/a | 12.3266 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(32,)` | PyTorch CPU | 0.0219 | 0.0219 | 0.0216 | 0.0221 | 0.0002 | 0.7883 | n/a | 12.3266 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(128,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0019 | 0.0001 | 0.9910 | n/a | 13.2944 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(128,)` | NumPy | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 0.9910 | n/a | 13.2944 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(128,)` | PyTorch CPU | 0.0216 | 0.0216 | 0.0214 | 0.0217 | 0.0001 | 0.9910 | n/a | 13.2944 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0027 | 0.0001 | 1.4691 | n/a | 23.8436 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| activations | sigmoid | `(128,)` | NumPy | 0.0036 | 0.0037 | 0.0036 | 0.0038 | 0.0001 | 1.4691 | n/a | 23.8436 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| activations | sigmoid | `(128,)` | PyTorch CPU | 0.0591 | 0.0589 | 0.0580 | 0.0597 | 0.0006 | 1.4691 | n/a | 23.8436 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| activations | tanh | `(128,)` | TensorStudio | 0.0026 | 0.0027 | 0.0026 | 0.0029 | 0.0001 | 0.5877 | n/a | 8.5720 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(128,)` | NumPy | 0.0016 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 0.5877 | n/a | 8.5720 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(128,)` | PyTorch CPU | 0.0226 | 0.0226 | 0.0221 | 0.0228 | 0.0002 | 0.5877 | n/a | 8.5720 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(128,)` | TensorStudio | 0.0024 | 0.0024 | 0.0024 | 0.0025 | 0.0000 | 0.5784 | n/a | 9.2210 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(128,)` | NumPy | 0.0014 | 0.0015 | 0.0014 | 0.0017 | 0.0001 | 0.5784 | n/a | 9.2210 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(128,)` | PyTorch CPU | 0.0222 | 0.0223 | 0.0219 | 0.0234 | 0.0006 | 0.5784 | n/a | 9.2210 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(128,)` | TensorStudio | 0.0025 | 0.0025 | 0.0024 | 0.0029 | 0.0002 | 0.6150 | n/a | 8.9709 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(128,)` | NumPy | 0.0015 | 0.0016 | 0.0015 | 0.0018 | 0.0001 | 0.6150 | n/a | 8.9709 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(128,)` | PyTorch CPU | 0.0221 | 0.0222 | 0.0221 | 0.0224 | 0.0001 | 0.6150 | n/a | 8.9709 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(1024,)` | TensorStudio | 0.0022 | 0.0022 | 0.0022 | 0.0022 | 0.0000 | 0.9562 | n/a | 10.0434 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(1024,)` | NumPy | 0.0021 | 0.0021 | 0.0020 | 0.0025 | 0.0002 | 0.9562 | n/a | 10.0434 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(1024,)` | PyTorch CPU | 0.0217 | 0.0227 | 0.0215 | 0.0265 | 0.0019 | 0.9562 | n/a | 10.0434 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(1024,)` | TensorStudio | 0.0091 | 0.0091 | 0.0089 | 0.0092 | 0.0001 | 0.6639 | n/a | 6.6525 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(1024,)` | NumPy | 0.0060 | 0.0060 | 0.0059 | 0.0061 | 0.0001 | 0.6639 | n/a | 6.6525 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(1024,)` | PyTorch CPU | 0.0603 | 0.0602 | 0.0599 | 0.0606 | 0.0002 | 0.6639 | n/a | 6.6525 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(1024,)` | TensorStudio | 0.0132 | 0.0133 | 0.0129 | 0.0143 | 0.0005 | 0.2893 | n/a | 1.8539 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(1024,)` | NumPy | 0.0038 | 0.0038 | 0.0037 | 0.0039 | 0.0001 | 0.2893 | n/a | 1.8539 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | tanh | `(1024,)` | PyTorch CPU | 0.0244 | 0.0243 | 0.0240 | 0.0244 | 0.0002 | 0.2893 | n/a | 1.8539 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | exp | `(1024,)` | TensorStudio | 0.0081 | 0.0080 | 0.0078 | 0.0082 | 0.0001 | 0.3185 | n/a | 2.7751 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(1024,)` | NumPy | 0.0026 | 0.0028 | 0.0026 | 0.0031 | 0.0002 | 0.3185 | n/a | 2.7751 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(1024,)` | PyTorch CPU | 0.0224 | 0.0225 | 0.0222 | 0.0233 | 0.0004 | 0.3185 | n/a | 2.7751 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(1024,)` | TensorStudio | 0.0084 | 0.0087 | 0.0083 | 0.0099 | 0.0006 | 0.3564 | n/a | 2.6589 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | log | `(1024,)` | NumPy | 0.0030 | 0.0031 | 0.0030 | 0.0033 | 0.0001 | 0.3564 | n/a | 2.6589 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | log | `(1024,)` | PyTorch CPU | 0.0224 | 0.0225 | 0.0223 | 0.0228 | 0.0002 | 0.3564 | n/a | 2.6589 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | relu | `(4096,)` | TensorStudio | 0.0044 | 0.0044 | 0.0044 | 0.0045 | 0.0001 | 0.7315 | n/a | 5.1134 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(4096,)` | NumPy | 0.0032 | 0.0032 | 0.0032 | 0.0034 | 0.0001 | 0.7315 | n/a | 5.1134 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(4096,)` | PyTorch CPU | 0.0224 | 0.0224 | 0.0221 | 0.0225 | 0.0001 | 0.7315 | n/a | 5.1134 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(4096,)` | TensorStudio | 0.0317 | 0.0322 | 0.0317 | 0.0334 | 0.0007 | 0.3377 | n/a | 2.9378 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(4096,)` | NumPy | 0.0107 | 0.0108 | 0.0106 | 0.0111 | 0.0002 | 0.3377 | n/a | 2.9378 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(4096,)` | PyTorch CPU | 0.0933 | 0.0886 | 0.0737 | 0.0994 | 0.0110 | 0.3377 | n/a | 2.9378 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | tanh | `(4096,)` | TensorStudio | 0.0914 | 0.0827 | 0.0602 | 0.1019 | 0.0157 | 0.1181 | n/a | 0.4750 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | tanh | `(4096,)` | NumPy | 0.0108 | 0.0109 | 0.0107 | 0.0111 | 0.0001 | 0.1181 | n/a | 0.4750 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | tanh | `(4096,)` | PyTorch CPU | 0.0434 | 0.0408 | 0.0323 | 0.0486 | 0.0058 | 0.1181 | n/a | 0.4750 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | exp | `(4096,)` | TensorStudio | 0.0352 | 0.0392 | 0.0336 | 0.0483 | 0.0058 | 0.2984 | n/a | 1.0055 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | exp | `(4096,)` | NumPy | 0.0105 | 0.0096 | 0.0063 | 0.0123 | 0.0024 | 0.2984 | n/a | 1.0055 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | exp | `(4096,)` | PyTorch CPU | 0.0354 | 0.0388 | 0.0333 | 0.0460 | 0.0052 | 0.2984 | n/a | 1.0055 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | log | `(4096,)` | TensorStudio | 0.0568 | 0.0517 | 0.0350 | 0.0624 | 0.0104 | 0.1504 | n/a | 0.6110 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(4096,)` | NumPy | 0.0085 | 0.0085 | 0.0083 | 0.0088 | 0.0002 | 0.1504 | n/a | 0.6110 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(4096,)` | PyTorch CPU | 0.0347 | 0.0383 | 0.0293 | 0.0552 | 0.0098 | 0.1504 | n/a | 0.6110 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | relu | `(16384,)` | TensorStudio | 0.0287 | 0.0266 | 0.0175 | 0.0305 | 0.0047 | 0.6988 | n/a | 1.8646 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| activations | relu | `(16384,)` | NumPy | 0.0201 | 0.0199 | 0.0189 | 0.0207 | 0.0007 | 0.6988 | n/a | 1.8646 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| activations | relu | `(16384,)` | PyTorch CPU | 0.0536 | 0.0527 | 0.0488 | 0.0567 | 0.0033 | 0.6988 | n/a | 1.8646 | n/a | no | n/a | yes | n/a | NumPy | reference |
| activations | sigmoid | `(16384,)` | TensorStudio | 0.2634 | 0.2749 | 0.2390 | 0.3305 | 0.0335 | 0.1233 | n/a | 0.3978 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | sigmoid | `(16384,)` | NumPy | 0.0325 | 0.0332 | 0.0301 | 0.0394 | 0.0032 | 0.1233 | n/a | 0.3978 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | sigmoid | `(16384,)` | PyTorch CPU | 0.1048 | 0.1117 | 0.0914 | 0.1357 | 0.0160 | 0.1233 | n/a | 0.3978 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | tanh | `(16384,)` | TensorStudio | 0.3057 | 0.3119 | 0.2577 | 0.3761 | 0.0475 | 0.1610 | n/a | 0.1405 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| activations | tanh | `(16384,)` | NumPy | 0.0492 | 0.0470 | 0.0422 | 0.0506 | 0.0036 | 0.1610 | n/a | 0.1405 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| activations | tanh | `(16384,)` | PyTorch CPU | 0.0430 | 0.0484 | 0.0357 | 0.0648 | 0.0117 | 0.1610 | n/a | 0.1405 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| activations | exp | `(16384,)` | TensorStudio | 0.1429 | 0.1625 | 0.1256 | 0.2063 | 0.0356 | 0.1415 | n/a | 0.3455 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | exp | `(16384,)` | NumPy | 0.0202 | 0.0212 | 0.0197 | 0.0246 | 0.0018 | 0.1415 | n/a | 0.3455 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | exp | `(16384,)` | PyTorch CPU | 0.0494 | 0.0490 | 0.0450 | 0.0539 | 0.0031 | 0.1415 | n/a | 0.3455 | n/a | no | n/a | no | n/a | NumPy | reference |
| activations | log | `(16384,)` | TensorStudio | 0.1376 | 0.1547 | 0.1253 | 0.2359 | 0.0412 | 0.2168 | n/a | 0.2303 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| activations | log | `(16384,)` | NumPy | 0.0298 | 0.0330 | 0.0265 | 0.0481 | 0.0080 | 0.2168 | n/a | 0.2303 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| activations | log | `(16384,)` | PyTorch CPU | 0.0317 | 0.0322 | 0.0302 | 0.0362 | 0.0021 | 0.2168 | n/a | 0.2303 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(1,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 2.0758 | n/a | 3.4341 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0033 | 0.0033 | 0.0025 | 0.0038 | 0.0005 | 2.0758 | n/a | 3.4341 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | PyTorch CPU | 0.0054 | 0.0059 | 0.0047 | 0.0072 | 0.0010 | 2.0758 | n/a | 3.4341 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0015 | 0.0021 | 0.0015 | 0.0030 | 0.0007 | 5.6559 | n/a | 7.1355 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0088 | 0.0103 | 0.0070 | 0.0142 | 0.0032 | 5.6559 | n/a | 7.1355 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | PyTorch CPU | 0.0110 | 0.0110 | 0.0109 | 0.0112 | 0.0001 | 5.6559 | n/a | 7.1355 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 1.2076 | n/a | 3.1824 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0001 | 1.2076 | n/a | 3.1824 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | PyTorch CPU | 0.0045 | 0.0045 | 0.0042 | 0.0047 | 0.0002 | 1.2076 | n/a | 3.1824 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0014 | 0.0014 | 0.0014 | 0.0015 | 0.0000 | 5.0749 | n/a | 7.8487 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0072 | 0.0072 | 0.0072 | 0.0073 | 0.0001 | 5.0749 | n/a | 7.8487 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | PyTorch CPU | 0.0111 | 0.0112 | 0.0109 | 0.0116 | 0.0002 | 5.0749 | n/a | 7.8487 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0015 | 0.0015 | 0.0015 | 0.0016 | 0.0000 | 1.1159 | n/a | 2.8363 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0001 | 1.1159 | n/a | 2.8363 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | sum | `(32,)` | PyTorch CPU | 0.0044 | 0.0045 | 0.0042 | 0.0050 | 0.0003 | 1.1159 | n/a | 2.8363 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0016 | 0.0016 | 0.0016 | 0.0016 | 0.0000 | 4.4051 | n/a | 7.1262 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0069 | 0.0069 | 0.0068 | 0.0070 | 0.0001 | 4.4051 | n/a | 7.1262 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | PyTorch CPU | 0.0112 | 0.0113 | 0.0109 | 0.0119 | 0.0003 | 4.4051 | n/a | 7.1262 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0022 | 0.0023 | 0.0022 | 0.0023 | 0.0000 | 0.7867 | n/a | 1.9714 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0018 | 0.0018 | 0.0017 | 0.0020 | 0.0001 | 0.7867 | n/a | 1.9714 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| reductions | sum | `(128,)` | PyTorch CPU | 0.0044 | 0.0046 | 0.0044 | 0.0048 | 0.0002 | 0.7867 | n/a | 1.9714 | n/a | no | n/a | yes | n/a | NumPy | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0021 | 0.0021 | 0.0021 | 0.0023 | 0.0001 | 3.3265 | n/a | 5.3193 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0070 | 0.0079 | 0.0069 | 0.0113 | 0.0017 | 3.3265 | n/a | 5.3193 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | PyTorch CPU | 0.0112 | 0.0113 | 0.0110 | 0.0115 | 0.0002 | 3.3265 | n/a | 5.3193 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0071 | 0.0072 | 0.0070 | 0.0078 | 0.0003 | 0.2925 | n/a | 0.6239 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0021 | 0.0021 | 0.0021 | 0.0022 | 0.0000 | 0.2925 | n/a | 0.6239 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | PyTorch CPU | 0.0044 | 0.0047 | 0.0044 | 0.0054 | 0.0004 | 0.2925 | n/a | 0.6239 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0070 | 0.0071 | 0.0069 | 0.0073 | 0.0001 | 1.0715 | n/a | 1.5903 | n/a | yes | n/a | yes | n/a | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0075 | 0.0076 | 0.0074 | 0.0078 | 0.0002 | 1.0715 | n/a | 1.5903 | n/a | yes | n/a | yes | n/a | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | PyTorch CPU | 0.0112 | 0.0118 | 0.0110 | 0.0129 | 0.0009 | 1.0715 | n/a | 1.5903 | n/a | yes | n/a | yes | n/a | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0241 | 0.0241 | 0.0240 | 0.0242 | 0.0001 | 0.1214 | n/a | 0.2111 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0029 | 0.0029 | 0.0029 | 0.0030 | 0.0000 | 0.1214 | n/a | 0.2111 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | PyTorch CPU | 0.0051 | 0.0051 | 0.0050 | 0.0054 | 0.0001 | 0.1214 | n/a | 0.2111 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0247 | 0.0284 | 0.0242 | 0.0442 | 0.0079 | 0.3375 | n/a | 0.4857 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0083 | 0.0083 | 0.0082 | 0.0086 | 0.0001 | 0.3375 | n/a | 0.4857 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | PyTorch CPU | 0.0120 | 0.0120 | 0.0117 | 0.0123 | 0.0002 | 0.3375 | n/a | 0.4857 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.0926 | 0.0943 | 0.0924 | 0.1004 | 0.0031 | 0.0709 | n/a | 0.0736 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0066 | 0.0064 | 0.0060 | 0.0067 | 0.0003 | 0.0709 | n/a | 0.0736 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | PyTorch CPU | 0.0068 | 0.0070 | 0.0068 | 0.0078 | 0.0004 | 0.0709 | n/a | 0.0736 | n/a | no | n/a | no | n/a | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.0934 | 0.0930 | 0.0921 | 0.0939 | 0.0008 | 0.1292 | n/a | 0.1534 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0121 | 0.0121 | 0.0120 | 0.0121 | 0.0000 | 0.1292 | n/a | 0.1534 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | PyTorch CPU | 0.0143 | 0.0145 | 0.0140 | 0.0157 | 0.0006 | 0.1292 | n/a | 0.1534 | n/a | no | n/a | no | n/a | NumPy | reference |
| matmul | matmul | `(16, 16)` | TensorStudio | 0.0030 | 0.0030 | 0.0030 | 0.0031 | 0.0000 | 0.6303 | n/a | 1.3850 | n/a | no | n/a | yes | n/a | NumPy | loss vs NumPy |
| matmul | matmul | `(16, 16)` | NumPy | 0.0019 | 0.0019 | 0.0019 | 0.0019 | 0.0000 | 0.6303 | n/a | 1.3850 | n/a | no | n/a | yes | n/a | NumPy | NumPy baseline |
| matmul | matmul | `(16, 16)` | PyTorch CPU | 0.0042 | 0.0042 | 0.0042 | 0.0043 | 0.0000 | 0.6303 | n/a | 1.3850 | n/a | no | n/a | yes | n/a | NumPy | reference |
| matmul | matmul | `(64, 64)` | TensorStudio | 0.0341 | 0.0352 | 0.0337 | 0.0398 | 0.0023 | 0.3045 | n/a | 0.3494 | n/a | no | n/a | no | n/a | NumPy | loss vs NumPy |
| matmul | matmul | `(64, 64)` | NumPy | 0.0104 | 0.0104 | 0.0103 | 0.0104 | 0.0000 | 0.3045 | n/a | 0.3494 | n/a | no | n/a | no | n/a | NumPy | NumPy baseline |
| matmul | matmul | `(64, 64)` | PyTorch CPU | 0.0119 | 0.0124 | 0.0075 | 0.0172 | 0.0033 | 0.3045 | n/a | 0.3494 | n/a | no | n/a | no | n/a | NumPy | reference |
| matmul | matmul | `(128, 128)` | TensorStudio | 0.2781 | 0.2814 | 0.2747 | 0.2922 | 0.0071 | 0.7835 | n/a | 0.1685 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(128, 128)` | NumPy | 0.2179 | 0.2052 | 0.1570 | 0.2372 | 0.0283 | 0.7835 | n/a | 0.1685 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(128, 128)` | PyTorch CPU | 0.0469 | 0.0410 | 0.0239 | 0.0618 | 0.0147 | 0.7835 | n/a | 0.1685 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| matmul | matmul | `(256, 256)` | TensorStudio | 2.4215 | 2.3500 | 2.1725 | 2.5032 | 0.1416 | 0.1663 | n/a | 0.0343 | n/a | no | n/a | no | n/a | PyTorch CPU | loss vs NumPy |
| matmul | matmul | `(256, 256)` | NumPy | 0.4027 | 0.4114 | 0.3903 | 0.4603 | 0.0256 | 0.1663 | n/a | 0.0343 | n/a | no | n/a | no | n/a | PyTorch CPU | NumPy baseline |
| matmul | matmul | `(256, 256)` | PyTorch CPU | 0.0832 | 0.0845 | 0.0778 | 0.0950 | 0.0057 | 0.1663 | n/a | 0.0343 | n/a | no | n/a | no | n/a | PyTorch CPU | reference |
| autograd | scalar_backward | `(1,)` | TensorStudio | 0.0250 | 0.0219 | 0.0138 | 0.0288 | 0.0061 | n/a | n/a | 4.8413 | n/a | n/a | n/a | yes | n/a | TensorStudio | no NumPy reference |
| autograd | scalar_backward | `(1,)` | PyTorch CPU | 0.1211 | 0.1377 | 0.1072 | 0.1802 | 0.0277 | n/a | n/a | 4.8413 | n/a | n/a | n/a | yes | n/a | TensorStudio | reference |
| autograd | elementwise_backward | `(1,)` | TensorStudio | 0.0317 | 0.0327 | 0.0289 | 0.0396 | 0.0038 | n/a | n/a | 6.4136 | n/a | n/a | n/a | yes | n/a | TensorStudio | no NumPy reference |
| autograd | elementwise_backward | `(1,)` | PyTorch CPU | 0.2036 | 0.2069 | 0.1935 | 0.2248 | 0.0109 | n/a | n/a | 6.4136 | n/a | n/a | n/a | yes | n/a | TensorStudio | reference |
| autograd | scalar_backward | `(128,)` | TensorStudio | 0.1401 | 0.1410 | 0.1372 | 0.1486 | 0.0040 | n/a | n/a | 0.7135 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | scalar_backward | `(128,)` | PyTorch CPU | 0.1000 | 0.1002 | 0.0979 | 0.1021 | 0.0016 | n/a | n/a | 0.7135 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | elementwise_backward | `(128,)` | TensorStudio | 0.3210 | 0.3237 | 0.3184 | 0.3357 | 0.0062 | n/a | n/a | 0.5820 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | elementwise_backward | `(128,)` | PyTorch CPU | 0.1868 | 0.1864 | 0.1833 | 0.1897 | 0.0022 | n/a | n/a | 0.5820 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | scalar_backward | `(1024,)` | TensorStudio | 1.0056 | 1.0068 | 0.9988 | 1.0143 | 0.0060 | n/a | n/a | 0.0999 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | scalar_backward | `(1024,)` | PyTorch CPU | 0.1004 | 0.1029 | 0.0987 | 0.1106 | 0.0045 | n/a | n/a | 0.0999 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| autograd | elementwise_backward | `(1024,)` | TensorStudio | 2.3527 | 2.3644 | 2.3458 | 2.4036 | 0.0218 | n/a | n/a | 0.0802 | n/a | n/a | n/a | no | n/a | PyTorch CPU | no NumPy reference |
| autograd | elementwise_backward | `(1024,)` | PyTorch CPU | 0.1887 | 0.1916 | 0.1866 | 0.1979 | 0.0048 | n/a | n/a | 0.0802 | n/a | n/a | n/a | no | n/a | PyTorch CPU | reference |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | TensorStudio | 1.4970 | 1.5332 | 1.4927 | 1.6214 | 0.0508 | 0.0958 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | loss vs NumPy |
| training_loop | tiny_linear_regression_10_steps | `(32,)` | NumPy | 0.1435 | 0.1416 | 0.1296 | 0.1527 | 0.0095 | 0.0958 | n/a | n/a | n/a | no | n/a | n/a | n/a | NumPy | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
