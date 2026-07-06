# Neural Network Modules

The `tensorstudio.nn` package is a Python-level module system built on top of
C++ TensorStudio tensors.

## Module

Subclass `nn.Module` and implement `forward`.

```python
from tensorstudio import nn

class TinyBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 4)
        self.activation = nn.Tanh()

    def forward(self, x):
        return self.activation(self.linear(x))
```

Calling a module delegates to `forward`.

```python
y = block(x)
```

## Parameters

`nn.Parameter` creates or marks tensors with `requires_grad=True`.

```python
weight = nn.Parameter([[1.0, 2.0], [3.0, 4.0]])
```

Module parameter discovery walks attributes, nested modules, lists, tuples, and
dictionaries.

```python
params = model.parameters()
```

## Linear

```python
layer = nn.Linear(in_features=3, out_features=2)
y = layer(x)
```

`Linear` computes:

```text
input @ weight.T + bias
```

The weight shape is `(out_features, in_features)`. The bias shape is
`(out_features,)`.

## Sequential

```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 1),
)
```

`Sequential` applies modules in order.

## Activations

Available activation modules:

- `nn.ReLU`
- `nn.Sigmoid`
- `nn.Tanh`

Functional versions live in `tensorstudio.nn.functional`.

## Losses

`nn.MSELoss` computes mean squared error:

```python
loss_fn = nn.MSELoss()
loss = loss_fn(prediction, target)
```

## Train And Eval Modes

Modules expose:

```python
model.train()
model.eval()
```

These currently toggle the `training` flag recursively. v0.1.0 modules do not
include dropout or batch normalization, so the flag is mostly scaffolding for
future modules.
