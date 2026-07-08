# Quantization And Custom Kernels

TensorStudio `1.16.0` adds research helpers for affine quantization and a small
custom-kernel registry.

## Quantization

```python
import tensorstudio as ts

x = ts.tensor([-1.0, 0.0, 1.0])
config = ts.quantization.QuantizationConfig(num_bits=8, symmetric=True)
packed = ts.quantization.quantize_tensor(x, config)
restored = packed.dequantize()

print(packed.qvalues.tolist())
print(restored.tolist())
```

Supported helpers:

- `quantize_tensor`
- `dequantize_tensor`
- `fake_quantize`
- `quantize_state_dict`
- `dequantize_state_dict`
- `quantization_report`

These helpers store quantized values in regular TensorStudio integer tensors.
They do not yet provide int8 GEMM or convolution kernels.

## Custom Kernels

```python
import tensorstudio as ts

ts.register_kernel("double", lambda x: x * 2.0, overwrite=True)
print(ts.call_kernel("double", ts.ones((2,))).tolist())
print(ts.kernel_info("double"))
ts.unregister_kernel("double")
```

The registry is useful for plugins, experiments, and native-extension callables.
It does not bypass TensorStudio's normal safety boundaries.
