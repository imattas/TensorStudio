# Quickstart

This guide covers installation, tensor creation, operations, autograd, a tiny
neural network, data loading, and serialization.

## Requirements

- Python 3.10 or newer.
- A C++20 compiler only when building from source.
- NumPy, installed automatically as a runtime dependency.

Prebuilt wheels are the preferred installation path. Source builds use CMake,
pybind11, and scikit-build-core.

## Install

From PyPI:

```bash
python -m pip install tensorstudio
```

From a source checkout:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Optional file-interchange and image-input extras:

```bash
python -m pip install "tensorstudio[onnx,vision]"
```

Validate the install:

```bash
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
pytest -q
```

## Create Tensors

```python
import tensorstudio as ts

x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
y = ts.ones((2, 2))
z = ts.linspace(0.0, 1.0, 5)

print(x.shape)          # (2, 2)
print(x.strides)        # (2, 1)
print(x.dtype)          # float32
print(x.device)         # cpu
print(x.is_contiguous)  # True
print(z.tolist())
```

## Creation Helpers

```python
ts.zeros((2, 3))
ts.empty((2, 3))
ts.ones((2, 3), dtype="float64")
ts.full((2, 3), 7.0)
ts.zeros_like(x)
ts.ones_like(x)
ts.full_like(x, 7.0)
ts.rand((2, 3), seed=123)
ts.rand_like(x, seed=123)
ts.randn((2, 3), seed=123)
ts.randn_like(x, seed=123)
ts.uniform((2, 3), low=-1.0, high=1.0, seed=123)
ts.normal((2, 3), mean=5.0, stddev=0.5, seed=123)
ts.randint((4,), low=0, high=10, seed=123)
ts.bernoulli((4,), probability=0.25, seed=123)
ts.arange(0, 10, 2)
ts.eye(3)
ts.linspace(-1.0, 1.0, 5)
```

`manual_seed(seed)` seeds TensorStudio's process-local C++ random generator:

```python
ts.manual_seed(42)
sample = ts.randn((2, 2))
```

## Operations

TensorStudio supports Python operators for arithmetic and matrix multiplication:

```python
a = ts.tensor([[1.0, 2.0], [3.0, 4.0]])
b = ts.tensor([10.0, 20.0])

print((a + b).tolist())
print((a * 2 - 1).tolist())
print((a @ ts.eye(2)).tolist())
print(a.T.tolist())
print(a[0, :].tolist())
```

Supported math includes `sum`, `mean`, `var`, `std`, `norm`, `max`, `min`,
`all`, `any`, `softmax`, `log_softmax`, `logsumexp`, `relu`, `sigmoid`,
`tanh`, `exp`, `log`, `log1p`, `sqrt`, `rsqrt`, trigonometric and inverse
trigonometric functions, `abs`, and `clamp`.

```python
x = ts.tensor([-2.0, -1.0, 0.0, 4.0])
print(x.abs().tolist())
print(x.clamp(-1.0, 2.0).tolist())
print(ts.sin(x).tolist())
print(ts.math.std(x).item())
```

Use stable probability helpers for logits:

```python
logits = ts.tensor([[1000.0, 1001.0, 999.0], [1.0, 2.0, 3.0]])

print(logits.softmax(axis=1).tolist())
print(logits.log_softmax(axis=1).tolist())
print(logits.logsumexp(axis=1).tolist())
```

Batched matrix multiplication works through `@`, `ts.bmm`, and the supported
`einsum` subset:

```python
left = ts.randn((2, 3, 4), seed=1)
right = ts.randn((2, 4, 5), seed=2)

print((left @ right).shape)
print(ts.bmm(left, right).shape)
print(ts.einsum("bij,bjk->bik", left, right).shape)
```

Reductions can operate over all elements, one axis, or a tuple/list of axes:

```python
x = ts.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
print(x.sum().item())
print(x.mean(axis=0).tolist())
print(x.max(axis=1, keepdims=True).tolist())
print(ts.arange(24).reshape((2, 3, 4)).sum(axis=(0, 2)).tolist())
print(x.argmax().item())
```

Tensors can be cast, concatenated, and stacked:

```python
a = ts.ones((1, 2))
b = ts.full((1, 2), 3.0)

print(a.astype("float64").dtype)
print(ts.concat([a, b], axis=0).tolist())
print(ts.stack([a, b], axis=1).shape)
```

NCHW image-style tensors can use native CPU convolution and pooling helpers:

```python
image = ts.ones((1, 1, 4, 4))
kernel = ts.ones((1, 1, 3, 3))

features = ts.conv2d(image, kernel, padding=1)
pooled = ts.max_pool2d(features, kernel_size=2)

print(features.shape)  # (1, 1, 4, 4)
print(pooled.shape)    # (1, 1, 2, 2)
```

## Autograd

```python
x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).mean()
loss.backward()

print(loss.item())
print(x.grad.tolist())
```

For non-scalar outputs, pass an explicit gradient:

```python
y = x * 2.0
y.backward(ts.ones(y.shape))
```

Disable graph recording with `no_grad()`:

```python
with ts.no_grad():
    doubled = x * 2.0
```

## Neural Network

```python
from tensorstudio import nn, optim

ts.manual_seed(0)

x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])

model = nn.Sequential(nn.Linear(1, 8), nn.Tanh(), nn.Linear(8, 1))
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
scheduler = optim.StepLR(optimizer, step_size=50, gamma=0.5)

for _ in range(100):
    optimizer.zero_grad()
    prediction = model(x)
    loss = loss_fn(prediction, y)
    loss.backward()
    optim.clip_grad_norm_(model.parameters(), max_norm=10.0)
    optimizer.step()
    scheduler.step()

print(loss.item())
```

For multiclass logits, use `CrossEntropyLoss` or the matching functional helper:

```python
logits = ts.tensor([[1.0, 2.0, 3.0], [1.5, -0.5, 0.25]], requires_grad=True)
target = ts.tensor([2, 0])
loss = nn.CrossEntropyLoss()(logits, target)
loss.backward()
```

## Vision

Convert HWC images to channel-first tensors and run a compact CNN classifier:

```python
import numpy as np

image = np.zeros((8, 8, 3), dtype=np.uint8)
transform = ts.vision.Compose(
    [
        ts.vision.Resize((8, 8)),
        ts.vision.ToTensor(),
        ts.vision.Normalize(0.5, 0.5),
    ]
)
x = transform(image).reshape((1, 3, 8, 8))

model = ts.vision.ImageClassifier((3, 8, 8), num_classes=2, channels=(4,))
logits = model(x)
target = ts.tensor([1], dtype="int64")

print(logits.shape)
print(ts.vision.accuracy(logits, target))
```

For local image directories, use `ImageFolder`:

```python
dataset = ts.vision.ImageFolder("dataset", transform=transform)
loader = ts.data.DataLoader(dataset, batch_size=8, shuffle=True, seed=7)
```

## DataLoader

```python
from tensorstudio.data import DataLoader, TensorDataset

dataset = TensorDataset(ts.arange(6).reshape((6, 1)), ts.arange(6))
loader = DataLoader(dataset, batch_size=2, shuffle=True, seed=7)

for features, targets in loader:
    print(features.tolist(), targets.tolist())
```

The DataLoader is single-process by design in v1.

## Projects

Use `tensorstudio.project` when an example grows into a repeatable run with
config, folders, training history, and checkpoints:

```python
from tensorstudio.project import Project, ProjectConfig, Trainer, save_state_dict

project = Project("runs/linear", ProjectConfig(name="linear-regression", seed=7))
trainer = Trainer(model, optimizer, loss_fn)
history = trainer.fit(loader, epochs=20)
save_state_dict(model, project.checkpoint_path("weights"))

print(history.last)
```

## Save And Load

```python
checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
ts.save(checkpoint, "checkpoint.tsmodel")
loaded = ts.load("checkpoint.tsmodel")
```

Only load TensorStudio pickle files from trusted sources.

Use `.tsnpz` files for safer tensor and state_dict storage:

```python
ts.save_npz(model.state_dict(), "weights.tsnpz")
model.load_state_dict(ts.load_npz("weights.tsnpz"))
```

## ONNX Export

Install the ONNX extra, then export supported Sequential models:

```python
ts.export_onnx(model.model, "classifier.onnx", input_shape=(1, 3, 8, 8))
```

The exporter covers common TensorStudio modules such as `Linear`, `Conv2d`,
`Flatten`, activations, and 2D pooling. It does not trace arbitrary Python
control flow.
