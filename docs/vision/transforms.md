# Vision Transforms

TensorStudio `1.14.0` includes `tensorstudio.vision` batch-aware image
transforms and practical augmentations for small local computer-vision
experiments.

## Batch Operations

```python
import tensorstudio as ts

images = ts.randn((8, 3, 32, 32), seed=7)
resized = ts.vision.batch_resize(images, (16, 16))
cropped = ts.vision.batch_center_crop(images, (12, 12))
normalized = ts.vision.batch_normalize(images, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
```

Batch helpers accept NCHW TensorStudio tensors and NHWC/NCHW NumPy arrays where
the transform can infer the channel axis.

## Augmentation Primitives

```python
transform = ts.vision.Compose(
    [
        ts.vision.RandomResizedCrop((32, 32), seed=1),
        ts.vision.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, seed=2),
        ts.vision.RandomRotation(10.0, seed=3),
        ts.vision.Cutout((8, 8), seed=4),
        ts.vision.ToTensor(),
    ]
)
```

Available additions:

- `ColorJitter`
- `RandomResizedCrop`
- `RandomRotation`
- `RandomAffine`
- `Cutout`
- functional `color_jitter`, `random_resized_crop`, `rotate`, `affine`, and
  `cutout`

The geometric transforms use deterministic nearest-neighbor sampling for
portable behavior. They are suitable for preprocessing, data augmentation, and
tests, not photogrammetry-grade resampling.

## Mixup And CutMix

```python
mixed, y_a, y_b, lam = ts.vision.mixup(images, labels, alpha=0.4, seed=7)
cut, y_a, y_b, lam = ts.vision.cutmix(images, labels, alpha=1.0, seed=8)
```

Both helpers return the augmented batch, the original target batch, the shuffled
target batch, and the mixing coefficient. Loss code can combine per-target
losses using `lam`.
