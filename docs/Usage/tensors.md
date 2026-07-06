# Tensor Basics

TensorStudio tensors are eager CPU tensors backed by C++ storage. A tensor owns
or views shared storage and carries metadata that describes how to interpret the
values.

## Tensor Metadata

Each tensor stores:

- shared storage
- dtype
- shape
- strides
- offset
- CPU device descriptor
- `requires_grad`
- optional gradient
- optional autograd metadata

This is the same broad model used by larger tensor frameworks: storage holds
bytes, while shape and stride metadata describe the logical tensor.

## Dtypes

Supported dtypes:

- `float32`
- `float64`
- `int32`
- `int64`
- `bool`

Autograd is supported for floating point tensors only.

```python
import tensorstudio as ts

ts.tensor([1, 2, 3]).dtype          # int64
ts.tensor([1.0, 2.0]).dtype         # float32
ts.tensor([1.0], dtype="float64")   # float64
```

## Shapes

Shapes are tuples of non-negative dimensions.

```python
x = ts.tensor([[1.0, 2.0], [3.0, 4.0]])

print(x.shape)  # (2, 2)
print(x.ndim)   # 2
print(x.size)   # 4
```

A scalar tensor has shape `()`.

```python
s = ts.tensor(3.14)
print(s.shape)
print(s.item())
```

## Creation Functions

```python
ts.zeros((2, 3))
ts.ones((2, 3), dtype="float64")
ts.full((2, 3), 5.0)
ts.randn((2, 3), seed=123)
ts.arange(0, 10, 2)
```

`randn` accepts a seed for deterministic examples and tests.

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
- Dimension `1` broadcasts to the other size.
- Missing leading dimensions behave like `1`.
- Incompatible shapes raise `ShapeError`.

## Views

TensorStudio v0.1.0 supports:

- `reshape`
- `flatten`
- 2D `transpose`
- `T` as a transpose property

```python
x = ts.arange(6).reshape((2, 3))
print(x.flatten().tolist())
print(x.T.tolist())
```

Reshape currently requires a contiguous tensor. Transpose creates a strided view.

## Reductions

Reductions operate over all elements and return scalar tensors.

```python
x = ts.tensor([1.0, 2.0, 3.0])
print(x.sum().item())
print(x.mean().item())
```

Axis-specific reductions are future work.

## Conversion

```python
array = x.numpy()   # NumPy copy
values = x.tolist()
scalar = x.sum().item()
```

NumPy interop copies data in both directions. Mutating the NumPy array returned
by `numpy()` does not mutate the TensorStudio tensor.
