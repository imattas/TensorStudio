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

## Convolution Layers

```python
layer = nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, groups=2)
x = ts.ones((4, 4, 28, 28))
y = layer(x)
```

`Conv2d` uses NCHW input and weight shape
`(out_channels, in_channels / groups, kernel_h, kernel_w)`. `kernel_size`,
`stride`, `padding`, and `dilation` accept either an integer or a pair of
integers.

The first implementation is a native CPU kernel with autograd for input,
weight, and optional bias. It supports grouped convolution and can express
depthwise convolution when `groups == in_channels`.

Additional convolution modules:

- `nn.Conv1d` for NCL sequence inputs. This is a thin wrapper over the native
  2D kernel with a singleton spatial dimension.
- `nn.DepthwiseConv2d` for depthwise image kernels with optional
  `depth_multiplier`.
- `nn.ConvTranspose2d` for NCHW transposed convolution with native CPU forward
  and backward kernels.

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

Adaptive and global pooling helpers:

- `nn.AdaptiveAvgPool2d`
- `nn.AdaptiveMaxPool2d`
- `nn.GlobalAvgPool2d`
- `nn.GlobalMaxPool2d`

Adaptive pooling divides the input spatial dimensions into deterministic bins
and reduces each bin. Global pooling reduces over the full NCHW spatial axes.

## Normalization

TensorStudio `1.9.0` adds:

- `nn.BatchNorm1d` for `(N, C)` and `(N, C, L)`.
- `nn.BatchNorm2d` for `(N, C, H, W)`.
- `nn.LayerNorm` for trailing normalized dimensions.

BatchNorm stores running mean and variance as module buffers. Buffers are saved
in `state_dict()` but are not returned by `parameters()`.

```python
bn = nn.BatchNorm2d(16)
print([name for name, _ in bn.named_buffers()])
```

## Embedding

```python
table = nn.Embedding(num_embeddings=1000, embedding_dim=64)
tokens = ts.tensor([[1, 4, 9]], dtype="int64")
vectors = table(tokens)
```

Embedding lookup is a native CPU operation. Gradients accumulate into repeated
rows of `table.weight`; integer indices are not differentiable.

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
- `nn.GELU`
- `nn.ELU`
- `nn.SELU`
- `nn.SiLU`
- `nn.Mish`

Functional versions live in `tensorstudio.nn.functional`.

## Losses

`nn.MSELoss` computes mean squared error:

```python
loss_fn = nn.MSELoss()
loss = loss_fn(prediction, target)
```

Classification losses include `BCELoss`, `BCEWithLogitsLoss`,
`CrossEntropyLoss`, `NLLLoss`, `FocalLoss`, `KLDivLoss`, and
`CosineEmbeddingLoss`. `CrossEntropyLoss` supports label smoothing.

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
training and eval mode. BatchNorm also changes behavior: training mode uses
batch statistics and updates running buffers, while eval mode uses the stored
running statistics.

## Model Summary

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.GlobalAvgPool2d(),
    nn.Flatten(),
    nn.Linear(8, 10),
)
info = nn.summary(model, input_shape=(1, 1, 28, 28))
print(info["total_parameters"])
```

The summary returns structured dictionaries with module names, parameter
counts, optional output shapes, and estimated parameter/buffer memory. It does
not claim runtime memory exactness.
