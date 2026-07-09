# Dataset Creation

TensorStudio `1.10.0` adds small dataset factories for common local workflows.
They keep data loading explicit and single-process, while covering the path from
arrays or image folders to repeatable train/validation splits.

## Arrays And Labels

Use `from_arrays` for tabular or small in-memory datasets:

```python
import tensorstudio as ts
from tensorstudio.data import DataLoader, from_arrays, train_val_split

features = [[0.0], [1.0], [2.0], [3.0]]
labels = [[1.0], [3.0], [5.0], [7.0]]

dataset = from_arrays(features, labels, dtype="float32")
train_data, val_data = train_val_split(dataset, val_fraction=0.25, seed=7)

for x, y in DataLoader(train_data, batch_size=2, shuffle=True, seed=7):
    print(x.shape, y.shape)
```

The factory copies data into TensorStudio tensors. That makes ownership simple:
later changes to the source NumPy array or Python list do not mutate the
dataset.

## TensorStudio Tensors

Use `from_tensors` when the features and targets are already TensorStudio
tensors:

```python
import tensorstudio as ts
from tensorstudio.data import from_tensors

x = ts.randn((8, 4), seed=11)
y = ts.randint((8,), low=0, high=3, dtype="int64", seed=12)
dataset = from_tensors(x, y)
```

All tensors must share the same leading dimension.

## Image Folders

Use `from_image_folder` for classification folders laid out as
`root/class_name/image.ext`:

```python
from tensorstudio.data import DataLoader, from_image_folder

dataset = from_image_folder("data/train")
loader = DataLoader(dataset, batch_size=16, shuffle=True, seed=7)

print(dataset.classes)
print(dataset.class_to_idx)
```

Images are loaded with Pillow, converted to CHW TensorStudio tensors, and labels
are class indices. Install the optional vision extra when using image files:

```bash
python -m pip install "tensorstudio[vision]"
```

## Deterministic Splits

`train_val_split` returns `Subset` views and accepts a seed:

```python
train_data, val_data = train_val_split(dataset, val_fraction=0.2, seed=42)
```

The same dataset length, fraction, and seed produce the same index split. The
splitter rejects empty or one-item datasets because both train and validation
sets need at least one item.

## Metadata Summaries

`dataset_summary` gives quick, inspectable metadata:

```python
from tensorstudio.data import dataset_summary

summary = dataset_summary(dataset)
print(summary)
```

For image datasets, summaries include class names and target-class indices when
the dataset exposes them.
