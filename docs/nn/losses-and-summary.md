# Losses And Model Summaries

TensorStudio neural-network losses are thin Python modules over TensorStudio
tensor operations. They are eager, deterministic, and intentionally explicit.

## Classification Losses

```python
import tensorstudio as ts
from tensorstudio import nn

logits = ts.tensor([[1.0, 2.0, 3.0], [2.0, 0.0, -1.0]], requires_grad=True)
target = ts.tensor([2, 0], dtype="int64")

loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, target)
loss.backward()
```

Available classification and distribution losses:

- `nn.CrossEntropyLoss`, with optional label smoothing.
- `nn.NLLLoss`, for log-probability inputs.
- `nn.FocalLoss`, for class-imbalance experiments.
- `nn.KLDivLoss`, for log-probability distribution matching.
- `nn.BCELoss` and `nn.BCEWithLogitsLoss`, for binary targets.

Functional equivalents live in `tensorstudio.nn.functional`.

## Pairwise Losses

`nn.CosineEmbeddingLoss` accepts similar or dissimilar pair labels:

```python
left = ts.tensor([[1.0, 0.0], [1.0, 0.0]], requires_grad=True)
right = ts.tensor([[1.0, 0.0], [-1.0, 0.0]], requires_grad=True)
labels = ts.tensor([1.0, -1.0])

loss = nn.CosineEmbeddingLoss()(left, right, labels)
```

Targets greater than or equal to zero are treated as similar pairs; negative
targets are treated as dissimilar pairs.

## Regression Losses

Existing regression losses remain available:

- `nn.MSELoss`
- `nn.L1Loss`
- `nn.HuberLoss`

## Model Summary

`nn.summary()` returns structured information instead of printing a fixed table.
That makes it useful in tests, docs, and notebooks.

```python
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.GlobalAvgPool2d(),
    nn.Flatten(),
    nn.Linear(8, 10),
)

info = nn.summary(model, input_shape=(1, 1, 28, 28))
print(info["total_parameters"])
print(info["estimated_total_bytes"])
```

The summary includes:

- module rows with names and types
- optional output shapes for simple sequential models
- total and trainable parameter counts
- buffer counts
- estimated parameter and buffer bytes

The byte estimate covers stored tensors, not temporary tensors created during
forward or backward passes.
