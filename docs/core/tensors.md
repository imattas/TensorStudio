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

The v1 release exposes a CPU device abstraction only. Unsupported
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

## DType Promotion

TensorStudio uses a compact promotion order for arithmetic:

```text
bool < int32 < int64 < float32 < float64
```

`add`, `sub`, `mul`, and `matmul` return the wider dtype from that order.
`div` always returns a floating dtype: `float64` if either input is `float64`,
otherwise `float32`. Comparisons always return `bool`.

| left | right | arithmetic result |
|---|---|---|
| `bool` | `bool` | `bool` |
| `bool` | `int32` | `int32` |
| `bool` | `int64` | `int64` |
| `bool` | `float32` | `float32` |
| `bool` | `float64` | `float64` |
| `int32` | `int32` | `int32` |
| `int32` | `int64` | `int64` |
| `int32` | `float32` | `float32` |
| `int32` | `float64` | `float64` |
| `int64` | `int64` | `int64` |
| `int64` | `float32` | `float32` |
| `int64` | `float64` | `float64` |
| `float32` | `float32` | `float32` |
| `float32` | `float64` | `float64` |
| `float64` | `float64` | `float64` |

Use `promote_types` and `result_type` to inspect the rule:

```python
ts.promote_types("int32", "float32")        # "float32"
ts.result_type("int64", "int32", op="div") # "float32"
ts.result_type("int64", "float32", op="gt") # "bool"
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
- `transpose()` for reversed axes
- `transpose(axis0, axis1)` for axis swaps
- `permute`
- `squeeze`
- `unsqueeze`
- `T` property
- Basic indexing and slicing with integers, slices, tuples, ellipsis, and
  `None`/newaxis

```python
x = ts.arange(6).reshape((2, 3))
print(x.flatten().tolist())
print(x.T.shape)
print(x[0, :].tolist())

y = ts.arange(24).reshape((2, 3, 4))
print(y.permute(2, 0, 1).shape)
print(y.transpose(0, 2).shape)
print(y.unsqueeze(0).squeeze(0).shape)
```

`reshape` currently requires a contiguous tensor. `transpose`, `permute`,
`squeeze`, `unsqueeze`, and basic slicing return metadata views that share the
same storage where possible.

Indexing supports common NumPy-style read cases:

```python
x = ts.arange(24).reshape((2, 3, 4))

x[0]              # integer index, drops the first dimension
x[:, 1:]          # slice
x[..., -1]        # ellipsis
x[None, :, 0, :]  # new leading axis
x[::-1, :, ::2]   # negative and stepped slices
x[1, 2, 3].item() # scalar zero-dimensional tensor
```

Advanced list, tensor, and boolean-mask indexing are not implemented yet.
Use comparison tensors with `where` for elementwise selection.

## Math

Elementwise:

- `add`, `sub`, `mul`, `div`
- `neg`
- scalar `pow`
- `relu`, `sigmoid`, `tanh`
- `exp`, `log`, `log1p`, `sqrt`, `rsqrt`, `abs`
- `sin`, `cos`, `tan`
- `asin`, `acos`, `atan`
- `maximum` and `minimum`
- `where`
- `clamp` and `clip`

Comparisons:

- `==`, `!=`, `<`, `<=`, `>`, `>=`
- `equal`, `not_equal`, `less`, `less_equal`, `greater`, `greater_equal`

Selection ops are native C++ tensor kernels. They accept Python scalars or
tensors, use NumPy-style broadcasting, and keep gradients for floating point
branches:

```python
x = ts.tensor([[-2.0, 0.5, 3.0]], requires_grad=True)
floor = ts.maximum(x, 0.0)
capped = floor.minimum(2.0)
positive_or_zero = ts.where(x.greater(0.0), x, 0.0)

loss = (capped + positive_or_zero).sum()
loss.backward()
```

`where` requires a boolean condition. It does not differentiate through that
condition, but it does route gradients into the selected branch values.

Matrix and reductions:

- `matmul` / `@` for 2D tensors
- `sum`
- `mean`
- `max`
- `min`
- `argmax`
- `argmin`

Reductions operate over all elements by default and support an int axis or a
tuple/list of axes with `keepdims`:

```python
x = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
x.sum(axis=1)
x.mean(axis=0, keepdims=True)
x.max(axis=-1)
x.reshape((1, 2, 3)).sum(axis=(0, 2))
x.argmax()
x.argmin(axis=1)
```

Higher-level helpers in `tensorstudio.math` include variance, standard
deviation, norms, square, and reciprocal:

```python
ts.math.variance(x)
ts.math.std(x, axis=0)
ts.math.norm(x, ord=2)
```

Image-style functional operations:

- `conv2d`
- `max_pool2d`
- `avg_pool2d`

These functions expect CPU NCHW tensors. The current implementation focuses on
portable eager kernels rather than CUDA/cuDNN-style throughput.

## Casting And Combining

```python
x = ts.tensor([[1.2, 2.8]])
x.astype("int32")
x.to("float64")

ts.concat([x, x], axis=0)
ts.stack([x, x], axis=1)
```

`concat` requires tensors with the same dtype and compatible non-concatenated
dimensions. `stack` requires identical shapes and inserts a new axis.

## Conversion

```python
array = x.numpy()
values = x.tolist()
scalar = x.sum().item()
```

NumPy interop copies data in both directions. Mutating the NumPy array returned
by `numpy()` does not mutate the TensorStudio tensor.

## Mutation

General in-place tensor math is intentionally limited in v1.
`+=` raises a clear error. Optimizers use private assignment helpers to update
parameters after gradients are computed.
