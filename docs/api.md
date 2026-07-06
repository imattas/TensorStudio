# API Overview

The public package is imported as:

```python
import tensorstudio as ts
```

The native extension module is `tensorstudio._C`. Most users should use the
Python API, which is a thin layer over the C++ tensor core plus Python-level
modules, optimizers, data utilities, and serialization helpers.

## Top-Level Exports

- `Tensor`
- `tensor`
- `from_numpy`
- `zeros`
- `ones`
- `empty`
- `full`
- `rand`
- `randn`
- `arange`
- `eye`
- `linspace`
- `manual_seed`
- `save`
- `load`
- `no_grad`
- `is_grad_enabled`
- `set_grad_enabled`
- `data`
- `nn`
- `optim`
- `__version__`

## Tensor Creation

```python
ts.tensor(data, dtype=None, requires_grad=False)
ts.from_numpy(array, requires_grad=False)
ts.zeros(shape, dtype="float32")
ts.ones(shape, dtype="float32")
ts.empty(shape, dtype="float32")
ts.full(shape, fill_value, dtype="float32")
ts.rand(shape, dtype="float32", seed=None)
ts.randn(shape, dtype="float32", seed=None)
ts.arange(start, stop=None, step=1, dtype="float32")
ts.eye(n, m=None, dtype="float32")
ts.linspace(start, stop, steps, dtype="float32")
```

Nested Python data must be rectangular. Ragged lists raise `ShapeError`.

## Tensor Properties

- `shape`
- `strides`
- `dtype`
- `device`
- `ndim`
- `size`
- `requires_grad`
- `grad`
- `is_contiguous`
- `T`

## Tensor Methods

Conversion:

- `numpy()`
- `tolist()`
- `item()`

Autograd and copying:

- `backward(gradient=None)`
- `zero_grad()`
- `clone()`
- `detach()`

Views:

- `reshape(*shape)` or `reshape(shape)`
- `flatten()`
- `transpose()`

Math:

- `sum()`
- `mean()`
- `max()`
- `min()`
- `relu()`
- `sigmoid()`
- `tanh()`
- `exp()`
- `log()`
- `sqrt()`
- `abs()`
- `clamp(min_value, max_value)`
- `clip(min_value, max_value)`

## Operators

- `x + y`
- `x - y`
- `x * y`
- `x / y`
- `-x`
- `x ** exponent`
- `x @ y`
- `x == y`
- `x != y`
- `x < y`
- `x <= y`
- `x > y`
- `x >= y`

`+=` is intentionally unsupported and raises a clear error.

## `tensorstudio.nn`

- `Module`
- `Parameter`
- `Linear`
- `Sequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `Dropout`
- `Flatten`
- `MSELoss`
- `L1Loss`
- `BCELoss`
- Functional equivalents for activations and losses.

Module methods:

- `parameters()`
- `named_parameters()`
- `modules()`
- `train()`
- `eval()`
- `zero_grad()`
- `state_dict()`
- `load_state_dict()`

## `tensorstudio.optim`

- `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)`
- `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`
- `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`

Optimizers implement `zero_grad`, `step`, `state_dict`, and `load_state_dict`.

## `tensorstudio.data`

- `Dataset`
- `TensorDataset`
- `DataLoader`

`DataLoader` supports batching, shuffle, `drop_last`, and deterministic seed.

## Error Types

Import native exception aliases from `tensorstudio.errors`:

- `TensorStudioError`
- `ShapeError`
- `DTypeError`
- `DeviceError`
- `AutogradError`

## Serialization

```python
ts.save(obj, "object.tsmodel")
loaded = ts.load("object.tsmodel")
```

Serialization uses pickle. Only load files from trusted sources.
