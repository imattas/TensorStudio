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

## DType Casting Policy

TensorStudio exposes explicit casting modes through `can_cast`, `cast`,
`astype`, and `to`:

```python
ts.can_cast("int32", "int64", casting="safe")      # True
ts.can_cast("int32", "float32", casting="safe")    # False
ts.can_cast("float64", "float32", casting="same_kind") # True

x = ts.tensor([1.2, 2.8], dtype="float64")
x.astype("float32", casting="same_kind")
x.to("int32")  # default casting="unsafe", matching NumPy-style astype behavior
```

Supported modes:

- `no` and `equiv`: exact canonical dtype matches only.
- `safe`: every value representable by the source dtype must be exactly
  representable by the target dtype.
- `same_kind`: `safe` casts plus casts within integer or floating dtype
  families.
- `unsafe`: any supported TensorStudio dtype conversion.

The native C++ kernel performs the actual conversion; the Python policy layer
decides whether a requested conversion is allowed.

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

Creation helpers accept `device=`. When it is omitted, TensorStudio uses the
thread-local `current_device()` placement:

```python
with ts.device_scope("cpu"):
    x = ts.zeros((2, 3))

y = ts.ones((2, 3), device="cpu")
```

The current runtime only supports CPU tensor storage. Non-CPU placement raises
`DeviceError` instead of silently falling back.

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
- Basic indexing and slicing with integers, slices, tuples, ellipsis, and
  `None`/newaxis
- One-axis Python integer-list gather indexing, materialized as a tensor copy
- One-axis Python boolean-list mask indexing, materialized as a tensor copy
- One-axis TensorStudio integer-index tensor indexing, materialized as a tensor
  copy and preserving the selector shape
- One-axis 1D TensorStudio boolean-mask tensor indexing, materialized as a
  tensor copy
- Full-shape TensorStudio boolean-mask indexing, materialized as a flattened
  tensor copy
- Prefix-shape TensorStudio boolean-mask indexing, materialized as selected
  leading blocks with trailing dimensions preserved
- Tuple-position partial TensorStudio boolean-mask indexing, materialized as
  selected blocks at the mask position
- Adjacent multi-axis 1D integer-list or integer-tensor indexing, paired
  element-by-element like NumPy advanced indexing
- Adjacent multi-axis 1D boolean-list, boolean-tensor, and mixed
  boolean/integer indexing, paired through the same native vectorized gather
  path
- Adjacent multi-axis integer-tensor indexing with broadcasted selector shapes
- Non-adjacent multi-axis integer-list or integer-tensor indexing, with the
  broadcasted selector shape placed before the remaining basic dimensions
- Non-adjacent mixed one-dimensional boolean/integer advanced axes, with the
  broadcasted selector shape placed before the remaining basic dimensions
- Tuple-position partial boolean masks mixed with integer or one-dimensional
  boolean advanced selectors

```python
x = ts.arange(6).reshape((2, 3))
print(x.flatten().tolist())
print(x.T.shape)
print(x[0, :].tolist())
```

`reshape` currently requires a contiguous tensor. `transpose` returns a strided
view. Basic slicing also returns a view where possible.

Indexing supports common NumPy-style read cases:

```python
x = ts.arange(24).reshape((2, 3, 4))

x[0]              # integer index, drops the first dimension
x[:, 1:]          # slice
x[..., -1]        # ellipsis
x[None, :, 0, :]  # new leading axis
x[::-1, :, ::2]   # negative and stepped slices
x[1, 2, 3].item() # scalar zero-dimensional tensor
x[:, [2, 0], :]   # one integer-list gather axis, returns a copy
x[[True, False]]  # one boolean-list mask axis, returns a copy
x[ts.tensor([1, 0], dtype="int64")]        # one integer tensor gather axis
x[ts.tensor([[1, 0], [0, 1]], dtype="int64")] # selector shape is preserved
x[ts.tensor([True, False], dtype="bool")]  # one tensor mask axis
x[x.greater(2)]   # full-shape boolean tensor mask, returns a flat copy

y = ts.arange(24).reshape((2, 3, 4))
y[ts.tensor([[True, False, True], [False, True, False]], dtype="bool")]
y[:, ts.tensor([[True, False, True, False],
                [False, True, False, True],
                [True, False, False, False]], dtype="bool")]
y[[1, 0], [2, 1]] # paired row/column advanced indexing
y[[True, False], [False, True, True]] # paired boolean masks
y[ts.tensor([True, False], dtype="bool"), ts.tensor([1, 2], dtype="int64")]
y[ts.tensor([[1], [0]], dtype="int64"), ts.tensor([[2, 0]], dtype="int64")]
y[[1, 0], :, [3, 1]] # separated advanced axes move selector shape first
y[[True, False], :, [1, 3]] # separated mixed boolean/integer axes
y[:, ts.tensor([[True, False, False, True],
                [False, True, False, False],
                [True, False, True, False]], dtype="bool"),
  ts.tensor([0, 2, 3, 1, 0], dtype="int64")]
```

Integer-list gather supports negative entries, repeated entries, and empty
lists. Integer tensor gathers support `int32` and `int64` 1D index tensors.
Tensor integer gathers support `int32` and `int64` selectors with rank one or
higher for a single advanced axis. Boolean-list and tensor masks must match the
length of the axis they consume.
Full-shape boolean tensor masks must match the complete tensor shape and return
a flattened tensor of selected values. Prefix-shape boolean tensor masks must
match the leading dimensions of the indexed tensor and return selected leading
blocks with trailing dimensions preserved. Multi-dimensional boolean tensor
masks can also appear inside a tuple, such as `x[:, mask]`, where the mask
shape must match the consecutive tensor axes at that tuple position and the
output keeps one selected dimension for the mask.
Adjacent multi-axis integer and one-dimensional boolean selectors are broadcast
across their advanced selector shapes. Selectors with the same shape are paired
element-by-element, and singleton selector dimensions broadcast across the
other selector shapes. When advanced integer or one-dimensional boolean
selectors are separated by basic indexing components, their broadcasted
selector shape is placed before the remaining basic output dimensions,
matching the common NumPy advanced-indexing layout.
Tuple-position partial boolean masks are expanded to coordinate gathers, so
they can participate in those same vectorized advanced-index broadcasts with
integer and one-dimensional boolean selectors.
When differentiable, backward scatters gradients back to the source tensor and
accumulates repeated gather indices. Full NumPy advanced-indexing parity is
still incomplete; use comparison tensors with `where` for elementwise
selection.

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
ts.cast(x, "float64", casting="safe")
x.astype("int32")
x.to("float64", casting="same_kind")

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
same_storage = x.to_device("cpu")
copied = ts.to_device(x, "cpu", copy=True)
```

NumPy interop copies data in both directions. Mutating the NumPy array returned
by `numpy()` does not mutate the TensorStudio tensor.

`to_device("cpu", copy=False)` returns a CPU tensor alias that shares storage.
`copy=True` returns a clone. Non-CPU targets raise `DeviceError` until real
backend storage and copy kernels are implemented.

## Mutation

General in-place tensor math is intentionally limited in v1.
`+=` raises a clear error. Optimizers use private assignment helpers to update
parameters after gradients are computed.
