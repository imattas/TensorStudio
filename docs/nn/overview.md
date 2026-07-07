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
model.trainable_parameters()
```

`named_parameters()` returns stable dotted names such as `0.weight` for
`Sequential` children.

`parameters()` returns all discovered parameters, including frozen parameters.
Use `trainable_parameters()` when you specifically want tensors with
`requires_grad=True`.

## Module Introspection

Modules include small PyTorch-style helpers for inspection and bulk updates:

```python
model.children()
model.named_children()
model.modules()
model.named_modules()
model.parameter_count()
model.parameter_count(trainable_only=True)
model.freeze()
model.unfreeze()
model.apply(lambda module: print(type(module).__name__))
```

`freeze()` and `unfreeze()` flip `requires_grad` on discovered parameters and
return the module for chaining.

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
- `nn.Conv1d`
- `nn.Conv2d`
- `nn.ConvTranspose2d`
- `nn.DepthwiseConv2d`
- `nn.MaxPool2d`
- `nn.AvgPool2d`
- `nn.AdaptiveAvgPool2d`
- `nn.AdaptiveMaxPool2d`
- `nn.GlobalAvgPool2d`
- `nn.GlobalMaxPool2d`
- `nn.BatchNorm1d`
- `nn.BatchNorm2d`
- `nn.LayerNorm`
- `nn.Embedding`
- `nn.Sequential`
- `nn.Identity`
- `nn.ReLU`
- `nn.LeakyReLU`
- `nn.Sigmoid`
- `nn.Tanh`
- `nn.Softplus`
- `nn.GELU`
- `nn.ELU`
- `nn.SELU`
- `nn.SiLU`
- `nn.Mish`
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

`Conv2d` expects NCHW input and stores weights as
`(out_channels, in_channels / groups, kernel_h, kernel_w)`.

```python
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
x = ts.ones((8, 1, 28, 28))
y = conv(x)
assert y.shape == (8, 4, 28, 28)
```

The convolution implementation is CPU-only and supports autograd for the input,
weight, and optional bias. Grouped convolution, depthwise convolution, `Conv1d`,
and `ConvTranspose2d` are available in `1.9.0`.

Pooling layers expect NCHW input and are parameter-free.

```python
pool = nn.MaxPool2d(kernel_size=2)
x = ts.ones((8, 4, 28, 28))
y = pool(x)
assert y.shape == (8, 4, 14, 14)
```

`nn.MaxPool2d` supports `kernel_size`, `stride`, `padding`, and `dilation`.
`nn.AvgPool2d` supports `kernel_size`, `stride`, `padding`, and
`count_include_pad`.

```python
model.train()
model.eval()
```

## Losses

Available loss modules:

- `nn.MSELoss`
- `nn.L1Loss`
- `nn.BCELoss`
- `nn.BCEWithLogitsLoss`
- `nn.CrossEntropyLoss`
- `nn.NLLLoss`
- `nn.FocalLoss`
- `nn.KLDivLoss`
- `nn.CosineEmbeddingLoss`
- `nn.HuberLoss`

Functional equivalents live in `tensorstudio.nn.functional`:

- `linear`
- `conv1d`
- `conv2d`
- `conv_transpose2d`
- `embedding`
- `max_pool2d`
- `avg_pool2d`
- `adaptive_avg_pool2d`
- `adaptive_max_pool2d`
- `global_avg_pool2d`
- `global_max_pool2d`
- `batch_norm`
- `layer_norm`
- `relu`
- `leaky_relu`
- `sigmoid`
- `tanh`
- `gelu`
- `elu`
- `selu`
- `silu`
- `mish`
- `softmax`
- `log_softmax`
- `softplus`
- `mse_loss`
- `l1_loss`
- `binary_cross_entropy`
- `binary_cross_entropy_with_logits`
- `cross_entropy`
- `nll_loss`
- `focal_loss`
- `kl_div`
- `cosine_embedding_loss`
- `huber_loss`

`BCELoss` expects probabilities, not logits. Inputs are clamped with a small
epsilon for numerical stability.

`BCEWithLogitsLoss` applies sigmoid before binary cross entropy. `HuberLoss` is
useful for regression tasks where large residuals should be less aggressive
than mean squared error.

`CrossEntropyLoss` expects logits with shape `(batch, classes)` and integer
class targets with shape `(batch,)`.

```python
logits = ts.tensor([[1.0, 2.0, 3.0], [1.5, -0.5, 0.25]])
target = ts.tensor([2, 0])
loss = nn.CrossEntropyLoss()(logits, target)
```

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
