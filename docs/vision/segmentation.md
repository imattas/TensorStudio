# Segmentation Utilities

TensorStudio `1.12.0` includes mask helpers and a folder dataset for semantic
segmentation experiments.

## Mask Metrics

```python
import tensorstudio as ts

prediction = ts.tensor([[0, 1], [0, 2]], dtype="int64")
target = ts.tensor([[0, 1], [2, 2]], dtype="int64")

print(ts.vision.mask_iou(prediction, target, num_classes=3))
print(ts.vision.mean_mask_iou(prediction, target, num_classes=3))
```

## Mask Conversion

```python
one_hot = ts.vision.mask_to_one_hot(target.numpy(), num_classes=3)
labels = ts.vision.one_hot_to_mask(one_hot)
boxes = ts.vision.masks_to_boxes(labels == 2)
```

## Mask Transforms

```python
resized = ts.vision.resize_mask(labels, (128, 128))
center = ts.vision.center_crop_mask(labels, (96, 96))
random_crop = ts.vision.random_crop_mask(labels, (96, 96), seed=7)
```

Mask transforms use nearest-neighbor sampling and operate over the last two
dimensions.

## SegmentationFolder

`SegmentationFolder` expects matching image and mask stems:

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

Images are loaded as CHW TensorStudio tensors. Masks are loaded as integer
TensorStudio tensors.
