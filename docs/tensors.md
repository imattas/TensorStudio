# Tensors

TensorStudio tensors are eager CPU tensors backed by C++ storage. A tensor can
own storage or view existing storage through shape, stride, and offset metadata.

## Metadata

Each tensor tracks:

- Shared storage.
- DType.
- Shape.
- Strides.
- Offset.
- Device descriptor.
- `requires_grad`.
- Optional gradient.
- Optional autograd metadata.

```python
import tensorstudio as ts

x = ts.arange(6).reshape((2, 3))

print(x.shape)          # (2, 3)
print(x.strides)        # (3, 1)
print(x.device)         # cpu
print(x.is_contiguous)  # True
```

The v1 release candidate exposes a CPU device abstraction only. Unsupported
devices raise errors instead of silently falling back.

## DTypes

Supported dtypes:

- `float32`
- `float64`
- `int32`
- `int64`
- `bool`

Autograd is supported for floating point tensors only.

```python
ts.tensor([1, 2, 3]).dtype
ts.tensor([1.0, 2.0]).dtype
ts.tensor([True, False]).dtype
ts.tensor([1.0], dtype="float64").dtype
```

## Creation

```python
ts.tensor([[1.0, 2.0], [3.0, 4.0]])
ts.from_numpy(array)
ts.zeros((2, 3))
ts.zeros_like(x)
ts.empty((2, 3))
ts.ones((2, 3))
ts.ones_like(x)
ts.full((2, 3), 5.0)
ts.full_like(x, 5.0)
ts.rand((2, 3), seed=123)
ts.rand_like(x, seed=123)
ts.randn((2, 3), seed=123)
ts.randn_like(x, seed=123)
ts.arange(0, 10, 2)
ts.eye(3)
ts.linspace(0.0, 1.0, 5)
```

`empty` does not guarantee any particular values. Current storage may appear
zeroed on some platforms, but code should fill it before use.

The `*_like` helpers copy shape, dtype, and the default `requires_grad` flag
from an existing tensor unless an override is provided:

```python
base = ts.ones((2, 3), dtype="float64", requires_grad=True)
ts.zeros_like(base)
ts.full_like(base, 5.0, requires_grad=False)
```

## Broadcasting

Binary elementwise operations use NumPy-style trailing-dimension broadcasting.

```python
a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ts.tensor([10.0, 20.0])

print((a + b).tolist())
```

Rules:

- Dimensions are matched from the end.
- Equal dimensions are compatible.
- A dimension of `1` can broadcast to the other size.
- Missing leading dimensions behave like `1`.
- Incompatible shapes raise `ShapeError`.

Broadcasting is also reduced correctly in autograd for supported operations.

## Views

Supported views:

- `reshape`
- `flatten`
- 2D `transpose`
- 2D `T` property

```python
x = ts.arange(6).reshape((2, 3))
print(x.flatten().tolist())
print(x.T.shape)
```

`reshape` currently requires a contiguous tensor. `transpose` returns a strided
view.

## Math

Elementwise:

- `add`, `sub`, `mul`, `div`
- `neg`
- scalar `pow`
- `relu`, `sigmoid`, `tanh`
- `exp`, `log`, `sqrt`, `abs`
- `clamp` and `clip`

Comparisons:

- `==`, `!=`, `<`, `<=`, `>`, `>=`

Matrix and reductions:

- `matmul` / `@` for 2D tensors
- `sum`
- `mean`
- `max`
- `min`

Reductions currently operate over all elements.

## Conversion

```python
array = x.numpy()
values = x.tolist()
scalar = x.sum().item()
```

NumPy interop copies data in both directions. Mutating the NumPy array returned
by `numpy()` does not mutate the TensorStudio tensor.

## Mutation

General in-place tensor math is intentionally limited in this release candidate.
`+=` raises a clear error. Optimizers use private assignment helpers to update
parameters after gradients are computed.
