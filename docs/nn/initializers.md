# Initializers

TensorStudio `1.9.0` includes `tensorstudio.nn.init`, a small collection of
in-place parameter initializers. They are designed for module construction and
experiments where deterministic seeds matter.

## Basic Fill And Random Initializers

```python
from tensorstudio import nn

layer = nn.Linear(32, 16)
nn.init.zeros_(layer.bias)
nn.init.ones_(layer.weight)
nn.init.uniform_(layer.weight, a=-0.1, b=0.1, seed=7)
nn.init.normal_(layer.weight, mean=0.0, std=0.02, seed=7)
```

Initializer functions return the tensor they mutate. Internally they assign
under `tensorstudio.no_grad()` so initialization does not become part of an
autograd graph.

## Xavier Initializers

Use Xavier/Glorot initialization for layers where preserving variance between
forward and backward passes is a good default.

```python
nn.init.xavier_uniform_(layer.weight, gain=1.0, seed=7)
nn.init.xavier_normal_(layer.weight, gain=1.0, seed=7)
```

For a matrix shaped `(out_features, in_features)`, TensorStudio uses:

- `fan_in = in_features`
- `fan_out = out_features`

For convolution weights, it multiplies the channel counts by the receptive
field size.

## Kaiming Initializers

Use Kaiming/He initialization for ReLU-like networks.

```python
conv = nn.Conv2d(3, 32, kernel_size=3, padding=1)
nn.init.kaiming_uniform_(conv.weight, nonlinearity="relu", seed=11)
nn.init.kaiming_normal_(conv.weight, nonlinearity="leaky_relu", a=0.1, seed=11)
```

Supported gain names:

- `linear`
- `sigmoid`
- `tanh`
- `relu`
- `leaky_relu`
- `selu`
- `conv1d`
- `conv2d`
- `conv_transpose2d`

Unsupported gain names raise `ValueError` rather than guessing.

## Limitations

- Initializers currently operate on dense CPU tensors.
- Sparse, quantized, and device-specific initialization are future work.
- The formulas are standard defaults, not a guarantee that a particular model
  will train well.
