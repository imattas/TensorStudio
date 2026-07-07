# Vision Dataset Creation

TensorStudio already includes small local image dataset helpers, and the roadmap
extends this toward richer dataset creation tools.

## Current ImageFolder Layout

Use one directory per class:

```text
dataset/
  cats/
    0001.png
  dogs/
    0002.png
```

```python
import tensorstudio as ts

transform = ts.vision.Compose(
    [
        ts.vision.Resize((32, 32)),
        ts.vision.ToTensor(),
        ts.vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

dataset = ts.vision.ImageFolder("dataset", transform=transform)
loader = ts.data.DataLoader(dataset, batch_size=8, shuffle=True, seed=7)
```

## Explicit Image Lists

Use `ImageList` when paths and labels come from a CSV, database export, or
custom split file.

```python
items = [
    ("images/a.png", 0),
    ("images/b.png", 1),
]
dataset = ts.vision.ImageList(items, transform=transform)
```

## Planned Dataset Builders

The roadmap includes helpers that create datasets from:

- Image folders.
- Explicit image path lists.
- Tensor/label pairs.
- NumPy arrays.
- Tabular arrays with labels.

These builders should produce metadata summaries such as sample counts, class
names, shape hints, and deterministic train/validation splits.

