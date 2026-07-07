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

## DetectionFolder Layout

Detection datasets can use matching image and JSON annotation folders:

```text
dataset/
  images/
    frame-001.png
  annotations/
    frame-001.json
```

```json
{
  "boxes": [[0, 0, 32, 32]],
  "labels": [1]
}
```

```python
dataset = ts.vision.DetectionFolder("dataset")
image, target = dataset[0]
print(target["boxes"], target["labels"])
```

`tensorstudio.data.from_detection_folder()` provides the same dataset through
the data factory namespace.

## SegmentationFolder Layout

Segmentation datasets use matching image and mask stems:

```text
dataset/
  images/
    sample.png
  masks/
    sample.png
```

```python
dataset = ts.vision.SegmentationFolder("dataset")
image, mask = dataset[0]
```

`tensorstudio.data.from_segmentation_folder()` provides the same dataset through
the data factory namespace.

## Metadata And Splits

All dataset types work with `tensorstudio.data.dataset_summary()` where the
dataset exposes metadata and with deterministic `train_val_split()` for local
experiments.
