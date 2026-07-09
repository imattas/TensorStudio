# NumPy Interop

TensorStudio depends on NumPy for interop, tests, examples, and benchmark
reference results. Core tensor math is implemented in C++ rather than by
delegating to NumPy.

## From NumPy To TensorStudio

```python
import numpy as np
import tensorstudio as ts

array = np.array([[1.0, 2.0]], dtype=np.float32)
x = ts.from_numpy(array)
```

`from_numpy` creates a copy. Later changes to `array` do not affect `x`.

```python
array[0, 0] = 99.0
print(x.tolist())  # [[1.0, 2.0]]
```

## From TensorStudio To NumPy

```python
x = ts.tensor([1.0, 2.0, 3.0])
array = x.numpy()
```

`numpy()` returns a copy. Mutating the returned array does not affect the tensor.

## Dtype Mapping

TensorStudio supports these NumPy dtype roundtrips:

- `np.float32` <-> `float32`
- `np.float64` <-> `float64`
- `np.int32` <-> `int32`
- `np.int64` <-> `int64`
- `np.bool_` <-> `bool`

Unsupported NumPy dtypes raise `DTypeError`.

## Testing Pattern

Tests should compare TensorStudio output with NumPy when a NumPy equivalent is
straightforward.

```python
np.testing.assert_allclose((a @ b).numpy(), a.numpy() @ b.numpy())
```

This keeps the implementation honest without making NumPy part of the execution
path for core ops.
