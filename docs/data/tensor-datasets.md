# Tensor Datasets

TensorStudio data utilities are intentionally small. They are meant to cover
the common local cases: tensors already in memory, simple supervised samples,
and deterministic batching for examples and tests.

## `TensorDataset`

`TensorDataset` groups one or more tensors along the first dimension. Every
tensor must have the same leading length.

```python
import tensorstudio as ts
from tensorstudio.data import TensorDataset

x = ts.linspace(-1.0, 1.0, 32).reshape((32, 1))
y = 2.0 * x + 1.0

dataset = TensorDataset(x, y)
features, target = dataset[0]
print(len(dataset))
```

Use this for:

- Regression examples with feature and target tensors.
- Classification examples with image tensors and integer labels.
- Small benchmark inputs that should remain deterministic.

## Sample Shape Contract

The first dimension is the sample axis. A feature tensor with shape
`(32, 3, 28, 28)` contains 32 image samples. A label tensor with shape `(32,)`
contains 32 labels.

When a `DataLoader` batches samples from a `TensorDataset`, it stacks each field
back along a new leading batch axis.

## When To Write A Custom Dataset

Create a subclass of `Dataset` when samples are not already packed into a
single tensor:

```python
from tensorstudio.data import Dataset


class PairDataset(Dataset):
    def __init__(self, pairs):
        self._pairs = list(pairs)

    def __len__(self):
        return len(self._pairs)

    def __getitem__(self, index):
        return self._pairs[index]
```

Custom datasets should avoid expensive work in `__len__`. Put per-sample IO or
transforms in `__getitem__`, and keep randomness seedable when possible.

## Current Limits

`TensorDataset` is eager and CPU-only. It does not memory-map files, shard
across workers, or prefetch batches. Those capabilities belong later in the
roadmap after the core tensor and model APIs are stable.
