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

- TensorStudio wins versus NumPy: `8`
- TensorStudio losses versus NumPy: `6`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| reductions | sum | `(1,)` | TensorStudio | 0.0017 | 0.0019 | 0.0016 | 0.0026 | 0.0004 | 1.0312 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0017 | 0.0017 | 0.0017 | 0.0017 | 0.0000 | 1.0312 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1,)` | TensorStudio | 0.0017 | 0.0017 | 0.0017 | 0.0018 | 0.0000 | 4.0986 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0070 | 0.0070 | 0.0068 | 0.0072 | 0.0001 | 4.0986 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(8,)` | TensorStudio | 0.0016 | 0.0017 | 0.0016 | 0.0018 | 0.0001 | 1.0138 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0016 | 0.0017 | 0.0016 | 0.0020 | 0.0001 | 1.0138 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(8,)` | TensorStudio | 0.0014 | 0.0014 | 0.0014 | 0.0014 | 0.0000 | 4.9512 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0070 | 0.0074 | 0.0069 | 0.0087 | 0.0007 | 4.9512 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(32,)` | TensorStudio | 0.0016 | 0.0016 | 0.0015 | 0.0016 | 0.0000 | 1.0867 | n/a | n/a | n/a | win vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0017 | 0.0019 | 0.0017 | 0.0028 | 0.0005 | 1.0867 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(32,)` | TensorStudio | 0.0015 | 0.0016 | 0.0015 | 0.0017 | 0.0001 | 4.6612 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0072 | 0.0072 | 0.0069 | 0.0077 | 0.0003 | 4.6612 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(128,)` | TensorStudio | 0.0020 | 0.0020 | 0.0020 | 0.0021 | 0.0000 | 0.8636 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0017 | 0.0018 | 0.0017 | 0.0022 | 0.0002 | 0.8636 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(128,)` | TensorStudio | 0.0021 | 0.0021 | 0.0020 | 0.0021 | 0.0000 | 3.5539 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0073 | 0.0073 | 0.0071 | 0.0074 | 0.0001 | 3.5539 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(1024,)` | TensorStudio | 0.0075 | 0.0080 | 0.0072 | 0.0094 | 0.0008 | 0.3221 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0024 | 0.0024 | 0.0021 | 0.0030 | 0.0003 | 0.3221 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(1024,)` | TensorStudio | 0.0075 | 0.0074 | 0.0070 | 0.0077 | 0.0002 | 1.1545 | n/a | n/a | n/a | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0086 | 0.0087 | 0.0081 | 0.0098 | 0.0006 | 1.1545 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(4096,)` | TensorStudio | 0.0244 | 0.0245 | 0.0240 | 0.0253 | 0.0004 | 0.1191 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0029 | 0.0030 | 0.0029 | 0.0033 | 0.0001 | 0.1191 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(4096,)` | TensorStudio | 0.0241 | 0.0242 | 0.0240 | 0.0245 | 0.0002 | 0.3482 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0084 | 0.0084 | 0.0082 | 0.0088 | 0.0002 | 0.3482 | n/a | n/a | n/a | NumPy baseline |
| reductions | sum | `(16384,)` | TensorStudio | 0.0936 | 0.0971 | 0.0922 | 0.1128 | 0.0079 | 0.0660 | n/a | n/a | n/a | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0062 | 0.0064 | 0.0061 | 0.0070 | 0.0004 | 0.0660 | n/a | n/a | n/a | NumPy baseline |
| reductions | mean | `(16384,)` | TensorStudio | 0.0933 | 0.0932 | 0.0927 | 0.0938 | 0.0004 | 0.1335 | n/a | n/a | n/a | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0125 | 0.0130 | 0.0120 | 0.0151 | 0.0012 | 0.1335 | n/a | n/a | n/a | NumPy baseline |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
