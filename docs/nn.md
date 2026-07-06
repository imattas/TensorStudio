# Neural Networks

`tensorstudio.nn` is a small Python-level neural network system built on top of
TensorStudio tensor operations.

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
block = TinyBlock()
y = block(x)
```

## Parameters

`nn.Parameter` creates or marks a tensor with `requires_grad=True`.

```python
weight = nn.Parameter([[1.0, 2.0], [3.0, 4.0]])
```

Module parameter discovery walks attributes, nested modules, lists, tuples, and
dictionaries.

```python
model.parameters()
model.named_parameters()
```

`named_parameters()` returns stable dotted names such as `0.weight` for
`Sequential` children.

## State Dictionaries

Modules expose detached tensor copies through `state_dict()`.

```python
state = model.state_dict()
model.load_state_dict(state)
```

`load_state_dict` checks missing and unexpected keys by default. Pass
`strict=False` to load only matching keys.

## Layers And Containers

Available modules:

- `nn.Linear`
- `nn.Sequential`
- `nn.ReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Dropout`
- `nn.Flatten`

```python
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Dropout(p=0.1),
    nn.Linear(8, 1),
)
```

`Dropout` uses inverted dropout in training mode and is a no-op in eval mode.

```python
model.train()
model.eval()
```

## Losses

Available loss modules:

- `nn.MSELoss`
- `nn.L1Loss`
- `nn.BCELoss`

Functional equivalents live in `tensorstudio.nn.functional`:

- `relu`
- `sigmoid`
- `tanh`
- `mse_loss`
- `l1_loss`
- `binary_cross_entropy`

`BCELoss` expects probabilities, not logits. Inputs are clamped with a small
epsilon for numerical stability.

## Training Loop

```python
import tensorstudio as ts
from tensorstudio import nn, optim

ts.manual_seed(0)

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05)

for _ in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
```

## Repr

Modules implement a compact `repr` that shows nested module structure. It is
intended for debugging, not as a serialization format.
