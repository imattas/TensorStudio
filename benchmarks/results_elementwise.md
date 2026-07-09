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

- TensorStudio wins versus NumPy: `0`
- TensorStudio losses versus NumPy: `35`
- TensorStudio wins versus JAX CPU dispatch: `35`
- TensorStudio losses versus JAX CPU dispatch: `0`

TensorStudio did not beat NumPy on this machine for the available benchmark set. Performance remains a blocker for broad performance claims.

## Detailed Results

| category | operation | shape | library | median ms | mean ms | min ms | max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | win vs NumPy | win vs TensorFlow | win vs PyTorch | win vs JAX | fastest library | result |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|
| elementwise | add | `(1,)` | TensorStudio | 0.0035 | 0.0035 | 0.0032 | 0.0039 | 0.0002 | 0.2340 | n/a | n/a | 5.6144 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0014 | 0.0002 | 0.2340 | n/a | n/a | 5.6144 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1,)` | JAX CPU dispatch | 0.0196 | 0.0203 | 0.0170 | 0.0237 | 0.0023 | 0.2340 | n/a | n/a | 5.6144 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1,)` | TensorStudio | 0.0054 | 0.0053 | 0.0044 | 0.0064 | 0.0007 | 0.2668 | n/a | n/a | 3.0069 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1,)` | NumPy | 0.0015 | 0.0016 | 0.0010 | 0.0025 | 0.0005 | 0.2668 | n/a | n/a | 3.0069 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1,)` | JAX CPU dispatch | 0.0164 | 0.0164 | 0.0143 | 0.0184 | 0.0013 | 0.2668 | n/a | n/a | 3.0069 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1,)` | TensorStudio | 0.0035 | 0.0037 | 0.0032 | 0.0046 | 0.0005 | 0.5340 | n/a | n/a | 4.9424 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1,)` | NumPy | 0.0019 | 0.0017 | 0.0009 | 0.0027 | 0.0007 | 0.5340 | n/a | n/a | 4.9424 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1,)` | JAX CPU dispatch | 0.0172 | 0.0171 | 0.0162 | 0.0180 | 0.0007 | 0.5340 | n/a | n/a | 4.9424 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1,)` | TensorStudio | 0.0036 | 0.0045 | 0.0034 | 0.0062 | 0.0012 | 0.2903 | n/a | n/a | 3.5865 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1,)` | NumPy | 0.0010 | 0.0010 | 0.0009 | 0.0011 | 0.0001 | 0.2903 | n/a | n/a | 3.5865 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1,)` | JAX CPU dispatch | 0.0130 | 0.0134 | 0.0112 | 0.0171 | 0.0020 | 0.2903 | n/a | n/a | 3.5865 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | TensorStudio | 0.0140 | 0.0140 | 0.0119 | 0.0156 | 0.0012 | 0.6532 | n/a | n/a | 9.7185 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | NumPy | 0.0091 | 0.0098 | 0.0081 | 0.0122 | 0.0015 | 0.6532 | n/a | n/a | 9.7185 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1,)` | JAX CPU dispatch | 0.1359 | 0.1336 | 0.1280 | 0.1393 | 0.0046 | 0.6532 | n/a | n/a | 9.7185 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(8,)` | TensorStudio | 0.0064 | 0.0066 | 0.0048 | 0.0088 | 0.0013 | 0.1856 | n/a | n/a | 4.2520 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(8,)` | NumPy | 0.0012 | 0.0013 | 0.0008 | 0.0018 | 0.0004 | 0.1856 | n/a | n/a | 4.2520 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(8,)` | JAX CPU dispatch | 0.0274 | 0.0276 | 0.0257 | 0.0290 | 0.0012 | 0.1856 | n/a | n/a | 4.2520 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(8,)` | TensorStudio | 0.0051 | 0.0048 | 0.0038 | 0.0059 | 0.0009 | 0.2958 | n/a | n/a | 4.3480 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(8,)` | NumPy | 0.0015 | 0.0014 | 0.0011 | 0.0017 | 0.0002 | 0.2958 | n/a | n/a | 4.3480 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(8,)` | JAX CPU dispatch | 0.0224 | 0.0231 | 0.0210 | 0.0255 | 0.0017 | 0.2958 | n/a | n/a | 4.3480 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(8,)` | TensorStudio | 0.0058 | 0.0059 | 0.0050 | 0.0072 | 0.0009 | 0.3859 | n/a | n/a | 2.9726 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(8,)` | NumPy | 0.0022 | 0.0023 | 0.0018 | 0.0030 | 0.0004 | 0.3859 | n/a | n/a | 2.9726 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(8,)` | JAX CPU dispatch | 0.0171 | 0.0171 | 0.0161 | 0.0184 | 0.0008 | 0.3859 | n/a | n/a | 2.9726 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(8,)` | TensorStudio | 0.0036 | 0.0037 | 0.0034 | 0.0040 | 0.0003 | 0.3440 | n/a | n/a | 3.4893 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(8,)` | NumPy | 0.0012 | 0.0012 | 0.0010 | 0.0014 | 0.0001 | 0.3440 | n/a | n/a | 3.4893 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(8,)` | JAX CPU dispatch | 0.0126 | 0.0124 | 0.0114 | 0.0129 | 0.0005 | 0.3440 | n/a | n/a | 3.4893 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | TensorStudio | 0.0170 | 0.0173 | 0.0137 | 0.0225 | 0.0031 | 0.4502 | n/a | n/a | 7.7102 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | NumPy | 0.0077 | 0.0074 | 0.0069 | 0.0077 | 0.0003 | 0.4502 | n/a | n/a | 7.7102 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(8,)` | JAX CPU dispatch | 0.1311 | 0.1318 | 0.1241 | 0.1419 | 0.0058 | 0.4502 | n/a | n/a | 7.7102 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(32,)` | TensorStudio | 0.0037 | 0.0039 | 0.0032 | 0.0051 | 0.0007 | 0.3717 | n/a | n/a | 4.4691 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(32,)` | NumPy | 0.0014 | 0.0014 | 0.0013 | 0.0016 | 0.0001 | 0.3717 | n/a | n/a | 4.4691 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(32,)` | JAX CPU dispatch | 0.0164 | 0.0168 | 0.0155 | 0.0184 | 0.0013 | 0.3717 | n/a | n/a | 4.4691 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(32,)` | TensorStudio | 0.0035 | 0.0035 | 0.0032 | 0.0038 | 0.0002 | 0.2158 | n/a | n/a | 4.1070 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(32,)` | NumPy | 0.0008 | 0.0008 | 0.0007 | 0.0008 | 0.0000 | 0.2158 | n/a | n/a | 4.1070 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(32,)` | JAX CPU dispatch | 0.0143 | 0.0144 | 0.0135 | 0.0159 | 0.0008 | 0.2158 | n/a | n/a | 4.1070 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(32,)` | TensorStudio | 0.0037 | 0.0040 | 0.0033 | 0.0054 | 0.0008 | 0.2378 | n/a | n/a | 4.8273 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(32,)` | NumPy | 0.0009 | 0.0009 | 0.0008 | 0.0013 | 0.0002 | 0.2378 | n/a | n/a | 4.8273 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(32,)` | JAX CPU dispatch | 0.0177 | 0.0177 | 0.0153 | 0.0200 | 0.0020 | 0.2378 | n/a | n/a | 4.8273 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(32,)` | TensorStudio | 0.0047 | 0.0050 | 0.0045 | 0.0065 | 0.0007 | 0.2605 | n/a | n/a | 2.7422 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(32,)` | NumPy | 0.0012 | 0.0014 | 0.0011 | 0.0020 | 0.0003 | 0.2605 | n/a | n/a | 2.7422 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(32,)` | JAX CPU dispatch | 0.0130 | 0.0132 | 0.0115 | 0.0164 | 0.0017 | 0.2605 | n/a | n/a | 2.7422 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | TensorStudio | 0.0128 | 0.0126 | 0.0118 | 0.0133 | 0.0006 | 0.5580 | n/a | n/a | 9.6573 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | NumPy | 0.0072 | 0.0075 | 0.0066 | 0.0094 | 0.0010 | 0.5580 | n/a | n/a | 9.6573 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(32,)` | JAX CPU dispatch | 0.1239 | 0.1226 | 0.1172 | 0.1249 | 0.0029 | 0.5580 | n/a | n/a | 9.6573 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(128,)` | TensorStudio | 0.0043 | 0.0047 | 0.0041 | 0.0055 | 0.0006 | 0.2015 | n/a | n/a | 3.8395 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(128,)` | NumPy | 0.0009 | 0.0009 | 0.0007 | 0.0014 | 0.0002 | 0.2015 | n/a | n/a | 3.8395 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(128,)` | JAX CPU dispatch | 0.0167 | 0.0165 | 0.0158 | 0.0171 | 0.0005 | 0.2015 | n/a | n/a | 3.8395 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(128,)` | TensorStudio | 0.0040 | 0.0039 | 0.0033 | 0.0044 | 0.0004 | 0.2927 | n/a | n/a | 4.4791 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(128,)` | NumPy | 0.0012 | 0.0012 | 0.0010 | 0.0014 | 0.0001 | 0.2927 | n/a | n/a | 4.4791 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(128,)` | JAX CPU dispatch | 0.0179 | 0.0173 | 0.0152 | 0.0195 | 0.0016 | 0.2927 | n/a | n/a | 4.4791 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(128,)` | TensorStudio | 0.0042 | 0.0042 | 0.0034 | 0.0049 | 0.0006 | 0.1918 | n/a | n/a | 4.1876 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(128,)` | NumPy | 0.0008 | 0.0009 | 0.0008 | 0.0010 | 0.0001 | 0.1918 | n/a | n/a | 4.1876 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(128,)` | JAX CPU dispatch | 0.0177 | 0.0189 | 0.0160 | 0.0220 | 0.0024 | 0.1918 | n/a | n/a | 4.1876 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(128,)` | TensorStudio | 0.0055 | 0.0055 | 0.0041 | 0.0071 | 0.0009 | 0.3806 | n/a | n/a | 2.7351 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(128,)` | NumPy | 0.0021 | 0.0021 | 0.0019 | 0.0022 | 0.0001 | 0.3806 | n/a | n/a | 2.7351 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(128,)` | JAX CPU dispatch | 0.0150 | 0.0142 | 0.0113 | 0.0163 | 0.0020 | 0.3806 | n/a | n/a | 2.7351 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | TensorStudio | 0.0155 | 0.0159 | 0.0151 | 0.0174 | 0.0008 | 0.6719 | n/a | n/a | 9.0466 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | NumPy | 0.0104 | 0.0102 | 0.0091 | 0.0110 | 0.0007 | 0.6719 | n/a | n/a | 9.0466 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(128,)` | JAX CPU dispatch | 0.1400 | 0.1318 | 0.1100 | 0.1440 | 0.0133 | 0.6719 | n/a | n/a | 9.0466 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(1024,)` | TensorStudio | 0.0039 | 0.0042 | 0.0038 | 0.0056 | 0.0007 | 0.4864 | n/a | n/a | 4.1159 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(1024,)` | NumPy | 0.0019 | 0.0023 | 0.0014 | 0.0034 | 0.0009 | 0.4864 | n/a | n/a | 4.1159 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(1024,)` | JAX CPU dispatch | 0.0162 | 0.0164 | 0.0145 | 0.0189 | 0.0014 | 0.4864 | n/a | n/a | 4.1159 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(1024,)` | TensorStudio | 0.0041 | 0.0040 | 0.0039 | 0.0042 | 0.0001 | 0.2862 | n/a | n/a | 4.3485 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(1024,)` | NumPy | 0.0012 | 0.0012 | 0.0011 | 0.0013 | 0.0001 | 0.2862 | n/a | n/a | 4.3485 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(1024,)` | JAX CPU dispatch | 0.0176 | 0.0176 | 0.0155 | 0.0207 | 0.0019 | 0.2862 | n/a | n/a | 4.3485 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(1024,)` | TensorStudio | 0.0040 | 0.0041 | 0.0034 | 0.0049 | 0.0005 | 0.2812 | n/a | n/a | 4.6579 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(1024,)` | NumPy | 0.0011 | 0.0011 | 0.0011 | 0.0012 | 0.0000 | 0.2812 | n/a | n/a | 4.6579 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(1024,)` | JAX CPU dispatch | 0.0187 | 0.0190 | 0.0172 | 0.0209 | 0.0014 | 0.2812 | n/a | n/a | 4.6579 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(1024,)` | TensorStudio | 0.0050 | 0.0051 | 0.0041 | 0.0070 | 0.0011 | 0.3404 | n/a | n/a | 2.9114 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(1024,)` | NumPy | 0.0017 | 0.0020 | 0.0013 | 0.0038 | 0.0009 | 0.3404 | n/a | n/a | 2.9114 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(1024,)` | JAX CPU dispatch | 0.0146 | 0.0151 | 0.0125 | 0.0178 | 0.0020 | 0.3404 | n/a | n/a | 2.9114 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | TensorStudio | 0.0156 | 0.0185 | 0.0151 | 0.0245 | 0.0039 | 0.5937 | n/a | n/a | 8.4558 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | NumPy | 0.0093 | 0.0105 | 0.0091 | 0.0149 | 0.0022 | 0.5937 | n/a | n/a | 8.4558 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(1024,)` | JAX CPU dispatch | 0.1321 | 0.1339 | 0.1272 | 0.1483 | 0.0074 | 0.5937 | n/a | n/a | 8.4558 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(4096,)` | TensorStudio | 0.0081 | 0.0082 | 0.0077 | 0.0088 | 0.0005 | 0.1989 | n/a | n/a | 2.1019 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(4096,)` | NumPy | 0.0016 | 0.0016 | 0.0016 | 0.0017 | 0.0000 | 0.1989 | n/a | n/a | 2.1019 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(4096,)` | JAX CPU dispatch | 0.0170 | 0.0177 | 0.0157 | 0.0209 | 0.0020 | 0.1989 | n/a | n/a | 2.1019 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(4096,)` | TensorStudio | 0.0060 | 0.0061 | 0.0054 | 0.0076 | 0.0007 | 0.3959 | n/a | n/a | 2.8976 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(4096,)` | NumPy | 0.0024 | 0.0024 | 0.0023 | 0.0025 | 0.0001 | 0.3959 | n/a | n/a | 2.8976 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(4096,)` | JAX CPU dispatch | 0.0173 | 0.0192 | 0.0159 | 0.0245 | 0.0032 | 0.3959 | n/a | n/a | 2.8976 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(4096,)` | TensorStudio | 0.0074 | 0.0074 | 0.0054 | 0.0101 | 0.0017 | 0.3066 | n/a | n/a | 2.6655 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(4096,)` | NumPy | 0.0023 | 0.0023 | 0.0022 | 0.0024 | 0.0001 | 0.3066 | n/a | n/a | 2.6655 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(4096,)` | JAX CPU dispatch | 0.0196 | 0.0195 | 0.0161 | 0.0227 | 0.0023 | 0.3066 | n/a | n/a | 2.6655 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(4096,)` | TensorStudio | 0.0071 | 0.0076 | 0.0066 | 0.0095 | 0.0011 | 0.2959 | n/a | n/a | 1.8648 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(4096,)` | NumPy | 0.0021 | 0.0024 | 0.0020 | 0.0034 | 0.0005 | 0.2959 | n/a | n/a | 1.8648 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(4096,)` | JAX CPU dispatch | 0.0132 | 0.0149 | 0.0126 | 0.0186 | 0.0024 | 0.2959 | n/a | n/a | 1.8648 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | TensorStudio | 0.0311 | 0.0319 | 0.0262 | 0.0398 | 0.0050 | 0.4502 | n/a | n/a | 4.8381 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | NumPy | 0.0140 | 0.0147 | 0.0124 | 0.0188 | 0.0022 | 0.4502 | n/a | n/a | 4.8381 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(4096,)` | JAX CPU dispatch | 0.1505 | 0.1559 | 0.1424 | 0.1810 | 0.0132 | 0.4502 | n/a | n/a | 4.8381 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | add | `(16384,)` | TensorStudio | 0.0124 | 0.0127 | 0.0119 | 0.0144 | 0.0009 | 0.4305 | n/a | n/a | 2.6284 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | add | `(16384,)` | NumPy | 0.0053 | 0.0056 | 0.0048 | 0.0067 | 0.0007 | 0.4305 | n/a | n/a | 2.6284 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | add | `(16384,)` | JAX CPU dispatch | 0.0325 | 0.0314 | 0.0221 | 0.0400 | 0.0058 | 0.4305 | n/a | n/a | 2.6284 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | sub | `(16384,)` | TensorStudio | 0.0123 | 0.0123 | 0.0122 | 0.0125 | 0.0001 | 0.4709 | n/a | n/a | 1.3331 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | sub | `(16384,)` | NumPy | 0.0058 | 0.0062 | 0.0054 | 0.0073 | 0.0008 | 0.4709 | n/a | n/a | 1.3331 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | sub | `(16384,)` | JAX CPU dispatch | 0.0164 | 0.0201 | 0.0152 | 0.0270 | 0.0053 | 0.4709 | n/a | n/a | 1.3331 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | mul | `(16384,)` | TensorStudio | 0.0119 | 0.0124 | 0.0115 | 0.0135 | 0.0008 | 0.4677 | n/a | n/a | 1.6331 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | mul | `(16384,)` | NumPy | 0.0056 | 0.0056 | 0.0055 | 0.0057 | 0.0001 | 0.4677 | n/a | n/a | 1.6331 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | mul | `(16384,)` | JAX CPU dispatch | 0.0194 | 0.0200 | 0.0157 | 0.0233 | 0.0029 | 0.4677 | n/a | n/a | 1.6331 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | div | `(16384,)` | TensorStudio | 0.0159 | 0.0162 | 0.0158 | 0.0168 | 0.0004 | 0.3475 | n/a | n/a | 1.5602 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | div | `(16384,)` | NumPy | 0.0055 | 0.0063 | 0.0050 | 0.0093 | 0.0015 | 0.3475 | n/a | n/a | 1.5602 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | div | `(16384,)` | JAX CPU dispatch | 0.0248 | 0.0241 | 0.0168 | 0.0281 | 0.0040 | 0.3475 | n/a | n/a | 1.5602 | no | n/a | n/a | yes | NumPy | reference |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | TensorStudio | 0.0760 | 0.0797 | 0.0643 | 0.0969 | 0.0123 | 0.4358 | n/a | n/a | 3.9182 | no | n/a | n/a | yes | NumPy | loss vs NumPy |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | NumPy | 0.0331 | 0.0315 | 0.0264 | 0.0376 | 0.0042 | 0.4358 | n/a | n/a | 3.9182 | no | n/a | n/a | yes | NumPy | NumPy baseline |
| elementwise | chain_relu((x*2+1)/3) | `(16384,)` | JAX CPU dispatch | 0.2977 | 0.2836 | 0.2304 | 0.3308 | 0.0345 | 0.4358 | n/a | n/a | 3.9182 | no | n/a | n/a | yes | NumPy | reference |

Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean
TensorStudio was faster for that specific case.
Win columns say whether that same speedup is above 1.0 for the competitor.
