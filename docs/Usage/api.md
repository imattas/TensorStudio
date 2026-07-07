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
- `zeros_like`
- `ones`
- `ones_like`
- `empty`
- `empty_like`
- `full`
- `full_like`
- `rand`
- `rand_like`
- `randn`
- `randn_like`
- `arange`
- `eye`
- `linspace`
- `conv2d`
- `max_pool2d`
- `avg_pool2d`
- `astype`
- `concat`
- `stack`
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
ts.zeros(shape, dtype="float32", requires_grad=False)
ts.zeros_like(input, dtype=None, requires_grad=None)
ts.ones(shape, dtype="float32", requires_grad=False)
ts.ones_like(input, dtype=None, requires_grad=None)
ts.empty(shape, dtype="float32", requires_grad=False)
ts.empty_like(input, dtype=None, requires_grad=None)
ts.full(shape, fill_value, dtype="float32", requires_grad=False)
ts.full_like(input, fill_value, dtype=None, requires_grad=None)
ts.rand(shape, dtype="float32", seed=None, requires_grad=False)
ts.rand_like(input, dtype=None, seed=None, requires_grad=None)
ts.randn(shape, dtype="float32", seed=None, requires_grad=False)
ts.randn_like(input, dtype=None, seed=None, requires_grad=None)
ts.arange(start, stop=None, step=1, dtype="float32", requires_grad=False)
ts.eye(n, m=None, dtype="float32", requires_grad=False)
ts.linspace(start, stop, steps, dtype="float32", requires_grad=False)
```

Nested Python data must be rectangular. Ragged lists raise `ShapeError`.
`requires_grad=True` is accepted for floating point tensors and rejected for
integer and boolean tensors.

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

- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `max(axis=None, keepdims=False)`
- `min(axis=None, keepdims=False)`
- `relu()`
- `sigmoid()`
- `tanh()`
- `exp()`
- `log()`
- `sqrt()`
- `abs()`
- `clamp(min_value, max_value)`
- `clip(min_value, max_value)`
- `astype(dtype)`
- `to(dtype)`

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

## Functional Ops

```python
ts.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1)
ts.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1)
ts.avg_pool2d(input, kernel_size, stride=None, padding=0, count_include_pad=False)
```

`conv2d` expects NCHW input with shape `(batch, in_channels, height, width)`
and weight shape `(out_channels, in_channels, kernel_h, kernel_w)`. Stride,
padding, and dilation accept either an integer or a pair of integers. The
current implementation is CPU-only and supports autograd for input, weight, and
bias.

`max_pool2d` and `avg_pool2d` also expect NCHW input. `kernel_size`, `stride`,
`padding`, and `dilation` accept an integer or a pair where supported. If
`stride=None`, pooling defaults to non-overlapping windows with
`stride=kernel_size`. `max_pool2d` supports dilation; `avg_pool2d` supports
`count_include_pad`.

```python
ts.astype(input, dtype)
ts.concat(tensors, axis=0)
ts.stack(tensors, axis=0)
```

`concat` joins same-dtype tensors along an existing axis. `stack` inserts a new
axis and requires identical input shapes.

## `tensorstudio.nn`

- `Module`
- `Parameter`
- `Linear`
- `Conv2d`
- `MaxPool2d`
- `AvgPool2d`
- `Sequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `Dropout`
- `Flatten`
- `Identity`
- `LeakyReLU`
- `Softplus`
- `MSELoss`
- `L1Loss`
- `BCELoss`
- `BCEWithLogitsLoss`
- `CrossEntropyLoss`
- `HuberLoss`
- Functional equivalents for linear layers, convolution, pooling, activations,
  softmax/log-softmax, and losses.

Module methods:

- `parameters()`
- `named_parameters()`
- `trainable_parameters()`
- `children()`
- `named_children()`
- `modules()`
- `named_modules()`
- `train()`
- `eval()`
- `zero_grad()`
- `requires_grad_()`
- `freeze()`
- `unfreeze()`
- `parameter_count()`
- `apply()`
- `state_dict()`
- `load_state_dict()`

## `tensorstudio.optim`

- `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)`
- `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`
- `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`
- `clip_grad_norm_(params, max_norm)`
- `clip_grad_value_(params, clip_value)`
- `StepLR(optimizer, step_size, gamma=0.1)`
- `ExponentialLR(optimizer, gamma)`

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
