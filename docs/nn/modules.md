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

## Conv2d

```python
layer = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
x = ts.ones((4, 1, 28, 28))
y = layer(x)
```

`Conv2d` uses NCHW input and weight shape
`(out_channels, in_channels, kernel_h, kernel_w)`. `kernel_size`, `stride`,
`padding`, and `dilation` accept either an integer or a pair of integers.

The first implementation is a native CPU kernel with autograd for input,
weight, and optional bias. It is not a CUDA/cuDNN replacement and does not yet
include grouped convolution.

## Pooling

```python
pool = nn.MaxPool2d(kernel_size=2)
x = ts.ones((4, 8, 28, 28))
y = pool(x)
```

`MaxPool2d` and `AvgPool2d` are parameter-free NCHW pooling layers backed by
native CPU kernels. `kernel_size`, `stride`, and `padding` accept either an
integer or a pair of integers. `MaxPool2d` also supports dilation. `AvgPool2d`
supports `count_include_pad`.

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
- `nn.LeakyReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Softplus`

Functional versions live in `tensorstudio.nn.functional`.

## Losses

`nn.MSELoss` computes mean squared error:

```python
loss_fn = nn.MSELoss()
loss = loss_fn(prediction, target)
```

Classification losses include `BCELoss`, `BCEWithLogitsLoss`, and
`CrossEntropyLoss`.

```python
logits = ts.tensor([[1.0, 2.0, 3.0], [1.5, -0.5, 0.25]])
target = ts.tensor([2, 0])
loss = nn.CrossEntropyLoss()(logits, target)
```

## Train And Eval Modes

Modules expose:

```python
model.train()
model.eval()
```

These toggle the `training` flag recursively. `Dropout` changes behavior between
training and eval mode; other current modules are deterministic in both modes.
