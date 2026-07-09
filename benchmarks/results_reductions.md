# TensorStudio Benchmark Results

These benchmarks are local references, not marketing claims.
TensorStudio is compared on CPU only and uses the same shapes and dtype.

## Environment

- Platform: `Windows-10-10.0.26200-SP0`
- Processor: `Intel64 Family 6 Model 158 Stepping 10, GenuineIntel`
- Python: `3.10.11`
- TensorStudio: `1.4.0`
- NumPy: `1.26.4`
- TensorFlow CPU eager: unavailable (not installed)
- PyTorch CPU: unavailable (not installed)
- JAX CPU dispatch: available (0.6.2)

## Summary

- TensorStudio wins versus NumPy: `13`
- TensorStudio losses versus NumPy: `9`
- TensorStudio wins versus JAX CPU dispatch: `17`
- TensorStudio losses versus JAX CPU dispatch: `5`

TensorStudio beat NumPy on at least one local benchmark case. See the detailed table before making any narrow performance claim.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| reductions | sum | `(1,)` | TensorStudio | 0.0032 | 0.0030 | 0.0026 | 0.0033 | 0.0002 | 1.1095 | n/a | n/a | 6.1520 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(1,)` | NumPy | 0.0035 | 0.0036 | 0.0026 | 0.0051 | 0.0008 | 1.1095 | n/a | n/a | 6.1520 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(1,)` | JAX CPU dispatch | 0.0195 | 0.0235 | 0.0165 | 0.0409 | 0.0089 | 1.1095 | n/a | n/a | 6.1520 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(1,)` | TensorStudio | 0.0040 | 0.0044 | 0.0033 | 0.0056 | 0.0009 | 4.2465 | n/a | n/a | 3.8494 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1,)` | NumPy | 0.0172 | 0.0173 | 0.0151 | 0.0191 | 0.0013 | 4.2465 | n/a | n/a | 3.8494 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1,)` | JAX CPU dispatch | 0.0156 | 0.0157 | 0.0134 | 0.0180 | 0.0015 | 4.2465 | n/a | n/a | 3.8494 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(8,)` | TensorStudio | 0.0029 | 0.0029 | 0.0027 | 0.0031 | 0.0001 | 1.0902 | n/a | n/a | 6.3141 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(8,)` | NumPy | 0.0032 | 0.0034 | 0.0028 | 0.0043 | 0.0006 | 1.0902 | n/a | n/a | 6.3141 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(8,)` | JAX CPU dispatch | 0.0183 | 0.0190 | 0.0158 | 0.0239 | 0.0028 | 1.0902 | n/a | n/a | 6.3141 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(8,)` | TensorStudio | 0.0031 | 0.0030 | 0.0028 | 0.0032 | 0.0001 | 3.8937 | n/a | n/a | 5.1562 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(8,)` | NumPy | 0.0120 | 0.0124 | 0.0097 | 0.0159 | 0.0022 | 3.8937 | n/a | n/a | 5.1562 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(8,)` | JAX CPU dispatch | 0.0159 | 0.0162 | 0.0139 | 0.0185 | 0.0017 | 3.8937 | n/a | n/a | 5.1562 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(32,)` | TensorStudio | 0.0052 | 0.0049 | 0.0038 | 0.0055 | 0.0006 | 0.5795 | n/a | n/a | 3.5412 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(32,)` | NumPy | 0.0030 | 0.0037 | 0.0027 | 0.0053 | 0.0011 | 0.5795 | n/a | n/a | 3.5412 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(32,)` | JAX CPU dispatch | 0.0182 | 0.0183 | 0.0175 | 0.0194 | 0.0006 | 0.5795 | n/a | n/a | 3.5412 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(32,)` | TensorStudio | 0.0037 | 0.0041 | 0.0029 | 0.0052 | 0.0009 | 3.5782 | n/a | n/a | 5.1342 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(32,)` | NumPy | 0.0134 | 0.0139 | 0.0111 | 0.0179 | 0.0022 | 3.5782 | n/a | n/a | 5.1342 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(32,)` | JAX CPU dispatch | 0.0192 | 0.0197 | 0.0176 | 0.0228 | 0.0021 | 3.5782 | n/a | n/a | 5.1342 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(128,)` | TensorStudio | 0.0048 | 0.0047 | 0.0041 | 0.0052 | 0.0004 | 1.2393 | n/a | n/a | 4.0971 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum | `(128,)` | NumPy | 0.0059 | 0.0055 | 0.0029 | 0.0067 | 0.0014 | 1.2393 | n/a | n/a | 4.0971 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum | `(128,)` | JAX CPU dispatch | 0.0195 | 0.0194 | 0.0182 | 0.0212 | 0.0011 | 1.2393 | n/a | n/a | 4.0971 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean | `(128,)` | TensorStudio | 0.0050 | 0.0050 | 0.0038 | 0.0067 | 0.0010 | 2.6741 | n/a | n/a | 4.4301 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(128,)` | NumPy | 0.0134 | 0.0136 | 0.0119 | 0.0153 | 0.0011 | 2.6741 | n/a | n/a | 4.4301 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(128,)` | JAX CPU dispatch | 0.0222 | 0.0209 | 0.0182 | 0.0229 | 0.0020 | 2.6741 | n/a | n/a | 4.4301 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(1024,)` | TensorStudio | 0.0180 | 0.0169 | 0.0107 | 0.0219 | 0.0048 | 0.3840 | n/a | n/a | 1.8108 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum | `(1024,)` | NumPy | 0.0069 | 0.0067 | 0.0046 | 0.0083 | 0.0012 | 0.3840 | n/a | n/a | 1.8108 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum | `(1024,)` | JAX CPU dispatch | 0.0325 | 0.0325 | 0.0300 | 0.0363 | 0.0022 | 0.3840 | n/a | n/a | 1.8108 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean | `(1024,)` | TensorStudio | 0.0184 | 0.0200 | 0.0168 | 0.0240 | 0.0028 | 1.2201 | n/a | n/a | 1.2968 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean | `(1024,)` | NumPy | 0.0224 | 0.0201 | 0.0134 | 0.0249 | 0.0048 | 1.2201 | n/a | n/a | 1.2968 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean | `(1024,)` | JAX CPU dispatch | 0.0238 | 0.0235 | 0.0211 | 0.0274 | 0.0023 | 1.2201 | n/a | n/a | 1.2968 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum | `(4096,)` | TensorStudio | 0.0357 | 0.0397 | 0.0336 | 0.0501 | 0.0063 | 0.2886 | n/a | n/a | 0.9312 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(4096,)` | NumPy | 0.0103 | 0.0103 | 0.0089 | 0.0114 | 0.0008 | 0.2886 | n/a | n/a | 0.9312 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(4096,)` | JAX CPU dispatch | 0.0332 | 0.0331 | 0.0325 | 0.0337 | 0.0005 | 0.2886 | n/a | n/a | 0.9312 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(4096,)` | TensorStudio | 0.0373 | 0.0375 | 0.0349 | 0.0412 | 0.0022 | 0.3699 | n/a | n/a | 0.5393 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(4096,)` | NumPy | 0.0138 | 0.0144 | 0.0126 | 0.0182 | 0.0020 | 0.3699 | n/a | n/a | 0.5393 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(4096,)` | JAX CPU dispatch | 0.0201 | 0.0205 | 0.0170 | 0.0228 | 0.0022 | 0.3699 | n/a | n/a | 0.5393 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum | `(16384,)` | TensorStudio | 0.1444 | 0.1545 | 0.1307 | 0.2129 | 0.0298 | 0.0619 | n/a | n/a | 0.2395 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum | `(16384,)` | NumPy | 0.0089 | 0.0090 | 0.0085 | 0.0098 | 0.0005 | 0.0619 | n/a | n/a | 0.2395 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum | `(16384,)` | JAX CPU dispatch | 0.0346 | 0.0343 | 0.0262 | 0.0444 | 0.0066 | 0.0619 | n/a | n/a | 0.2395 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean | `(16384,)` | TensorStudio | 0.1428 | 0.1499 | 0.1301 | 0.1924 | 0.0220 | 0.1110 | n/a | n/a | 0.2781 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | mean | `(16384,)` | NumPy | 0.0158 | 0.0185 | 0.0156 | 0.0251 | 0.0037 | 0.1110 | n/a | n/a | 0.2781 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | mean | `(16384,)` | JAX CPU dispatch | 0.0397 | 0.0442 | 0.0392 | 0.0574 | 0.0070 | 0.1110 | n/a | n/a | 0.2781 | no | n/a | n/a | no | NumPy | reference |
| reductions | sum_axis1 | `(16, 16)` | TensorStudio | 0.0033 | 0.0033 | 0.0032 | 0.0035 | 0.0001 | 1.0183 | n/a | n/a | 4.7128 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | sum_axis1 | `(16, 16)` | NumPy | 0.0034 | 0.0034 | 0.0033 | 0.0035 | 0.0001 | 1.0183 | n/a | n/a | 4.7128 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | sum_axis1 | `(16, 16)` | JAX CPU dispatch | 0.0156 | 0.0154 | 0.0148 | 0.0162 | 0.0005 | 1.0183 | n/a | n/a | 4.7128 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | mean_axis0 | `(16, 16)` | TensorStudio | 0.0029 | 0.0031 | 0.0029 | 0.0035 | 0.0002 | 4.5387 | n/a | n/a | 7.0448 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(16, 16)` | NumPy | 0.0134 | 0.0150 | 0.0120 | 0.0220 | 0.0036 | 4.5387 | n/a | n/a | 7.0448 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(16, 16)` | JAX CPU dispatch | 0.0208 | 0.0217 | 0.0157 | 0.0269 | 0.0043 | 4.5387 | n/a | n/a | 7.0448 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(64, 64)` | TensorStudio | 0.0076 | 0.0075 | 0.0067 | 0.0079 | 0.0004 | 0.6742 | n/a | n/a | 2.7905 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(64, 64)` | NumPy | 0.0051 | 0.0053 | 0.0050 | 0.0059 | 0.0003 | 0.6742 | n/a | n/a | 2.7905 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(64, 64)` | JAX CPU dispatch | 0.0212 | 0.0209 | 0.0163 | 0.0262 | 0.0038 | 0.6742 | n/a | n/a | 2.7905 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(64, 64)` | TensorStudio | 0.0059 | 0.0061 | 0.0044 | 0.0084 | 0.0015 | 3.6931 | n/a | n/a | 3.5082 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(64, 64)` | NumPy | 0.0219 | 0.0218 | 0.0157 | 0.0277 | 0.0045 | 3.6931 | n/a | n/a | 3.5082 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(64, 64)` | JAX CPU dispatch | 0.0208 | 0.0214 | 0.0166 | 0.0279 | 0.0044 | 3.6931 | n/a | n/a | 3.5082 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(128, 128)` | TensorStudio | 0.0206 | 0.0207 | 0.0202 | 0.0213 | 0.0004 | 0.4510 | n/a | n/a | 1.2457 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(128, 128)` | NumPy | 0.0093 | 0.0094 | 0.0091 | 0.0098 | 0.0003 | 0.4510 | n/a | n/a | 1.2457 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(128, 128)` | JAX CPU dispatch | 0.0257 | 0.0263 | 0.0221 | 0.0316 | 0.0034 | 0.4510 | n/a | n/a | 1.2457 | no | n/a | n/a | yes | NumPy | reference |
| reductions | mean_axis0 | `(128, 128)` | TensorStudio | 0.0162 | 0.0132 | 0.0078 | 0.0165 | 0.0039 | 1.1447 | n/a | n/a | 1.7099 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(128, 128)` | NumPy | 0.0186 | 0.0220 | 0.0175 | 0.0331 | 0.0059 | 1.1447 | n/a | n/a | 1.7099 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(128, 128)` | JAX CPU dispatch | 0.0277 | 0.0282 | 0.0245 | 0.0329 | 0.0034 | 1.1447 | n/a | n/a | 1.7099 | yes | n/a | n/a | yes | TensorStudio | reference |
| reductions | sum_axis1 | `(256, 256)` | TensorStudio | 0.0889 | 0.0892 | 0.0827 | 0.0930 | 0.0037 | 0.3381 | n/a | n/a | 0.8619 | no | n/a | n/a | no | NumPy | loss vs NumPy |
| reductions | sum_axis1 | `(256, 256)` | NumPy | 0.0301 | 0.0315 | 0.0255 | 0.0431 | 0.0064 | 0.3381 | n/a | n/a | 0.8619 | no | n/a | n/a | no | NumPy | NumPy baseline |
| reductions | sum_axis1 | `(256, 256)` | JAX CPU dispatch | 0.0767 | 0.0880 | 0.0752 | 0.1102 | 0.0149 | 0.3381 | n/a | n/a | 0.8619 | no | n/a | n/a | no | NumPy | reference |
| reductions | mean_axis0 | `(256, 256)` | TensorStudio | 0.0233 | 0.0246 | 0.0212 | 0.0334 | 0.0045 | 1.1715 | n/a | n/a | 1.4057 | yes | n/a | n/a | yes | TensorStudio | win vs NumPy |
| reductions | mean_axis0 | `(256, 256)` | NumPy | 0.0273 | 0.0306 | 0.0268 | 0.0392 | 0.0048 | 1.1715 | n/a | n/a | 1.4057 | yes | n/a | n/a | yes | TensorStudio | NumPy baseline |
| reductions | mean_axis0 | `(256, 256)` | JAX CPU dispatch | 0.0328 | 0.0282 | 0.0199 | 0.0342 | 0.0065 | 1.1715 | n/a | n/a | 1.4057 | yes | n/a | n/a | yes | TensorStudio | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
