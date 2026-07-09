# Data

`tensorstudio.data` provides a minimal map-style dataset and dataloader layer.
It is deliberately single-process in v1 so it works
cleanly on Windows and in simple teaching environments.

## Dataset

Subclass `Dataset` for map-style data:

```python
from tensorstudio.data import Dataset

class MyDataset(Dataset):
    def __len__(self) -> int:
        return 100

    def __getitem__(self, index: int):
        return index, index * 2
```

## TensorDataset

`TensorDataset` wraps tensors with matching leading dimensions.

```python
import tensorstudio as ts
from tensorstudio.data import TensorDataset

features = ts.arange(6).reshape((6, 1))
targets = ts.arange(6)
dataset = TensorDataset(features, targets)
```

Each item is returned as a tuple.

## DataLoader

```python
from tensorstudio.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    drop_last=False,
    seed=42,
)

for batch_features, batch_targets in loader:
    print(batch_features.shape, batch_targets.shape)
```

Features:

- Batching.
- `len(loader)` returns the number of batches.
- Deterministic shuffle with `seed`.
- Optional `drop_last`.
- Simple collation for TensorStudio tensors, numbers, booleans, lists, and
  tuples.

## Windows Behavior

There is no multiprocessing worker pool in v1. That avoids
Windows process-spawning pitfalls and keeps examples runnable from a normal
PowerShell prompt.

## Limitations

- No multiprocessing workers.
- No pinned memory.
- No streaming dataset helpers.
- No advanced custom collation API yet.
