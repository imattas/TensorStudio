# Broadcasting

TensorStudio binary elementwise operations use NumPy-style broadcasting.

## Rules

Shapes are compared from the trailing dimension backward:

- Equal dimensions are compatible.
- A dimension of `1` can expand to match the other dimension.
- Missing leading dimensions behave like `1`.
- Any other mismatch raises a shape error.

Examples:

```python
import tensorstudio as ts

x = ts.ones((2, 3))
b = ts.tensor([10.0, 20.0, 30.0])
print((x + b).shape)  # (2, 3)
```

Scalar tensors broadcast to any compatible shape:

```python
print((ts.ones((2, 2)) * 3.0).tolist())
```

## Covered Operations

Broadcasting is implemented in C++ for:

- Arithmetic.
- Comparisons.
- `maximum` and `minimum`.
- `where` condition and branch tensors.

Reduction gradients also unbroadcast accumulated gradients back to the original
operand shapes.

## Error Behavior

Incompatible shapes raise `ShapeError` with the relevant shapes. TensorStudio
does not silently reshape tensors.

```python
ts.ones((2, 3)) + ts.ones((4,))
```

This is invalid because trailing dimensions `3` and `4` are incompatible.

