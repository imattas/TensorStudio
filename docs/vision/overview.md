# Vision

TensorStudio's vision package provides local image IO, preprocessing,
augmentation, datasets, metrics, detection helpers, segmentation helpers,
visualization helpers, compact CNN classifiers, and compact UNet-style models
for image-recognition and small segmentation experiments.

It is designed for small CPU-first workflows. It is not a replacement for
OpenCV, torchvision, or specialized production vision stacks.

## Install

NumPy image arrays work with the base package. Install the optional Pillow extra
for file IO and bounding-box drawing:

```bash
python -m pip install "tensorstudio[vision]"
```

## Image IO

```python
import tensorstudio as ts

image = ts.vision.load_image("cat.png", mode="RGB")
ts.vision.save_image(image, "copy.png")
```

`load_image` returns an HWC NumPy array. `save_image` accepts NumPy arrays or
TensorStudio tensors in HWC, CHW, NHWC, or NCHW style layouts.

## Transform Pipelines

```python
transform = ts.vision.Compose(
    [
        ts.vision.Resize((32, 32), interpolation="bilinear"),
        ts.vision.RandomHorizontalFlip(p=0.5, seed=7),
        ts.vision.CenterCrop((28, 28)),
        ts.vision.ToTensor(),
        ts.vision.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
)

x = transform(image)
print(x.shape)  # (3, 28, 28)
```

Available transform functions include:

- `to_tensor`
- `to_numpy_image`
- `normalize`
- `resize_nearest`
- `resize_bilinear`
- `center_crop`
- `random_crop`
- `horizontal_flip`
- `vertical_flip`
- `random_horizontal_flip`
- `batch_resize`
- `batch_center_crop`
- `batch_normalize`
- `color_jitter`
- `random_resized_crop`
- `rotate`
- `affine`
- `cutout`
- `mixup`
- `cutmix`
- `pad`
- `rgb_to_grayscale`
- `grayscale_to_rgb`

Callable transform classes include:

- `Compose`
- `ToTensor`
- `Normalize`
- `Resize`
- `CenterCrop`
- `RandomCrop`
- `RandomHorizontalFlip`
- `ColorJitter`
- `RandomResizedCrop`
- `RandomRotation`
- `RandomAffine`
- `Cutout`

## ImageFolder

`ImageFolder` reads directory trees laid out like:

```text
dataset/
  cat/
    image0.png
  dog/
    image1.png
```

```python
dataset = ts.vision.ImageFolder("dataset", transform=transform)
loader = ts.data.DataLoader(dataset, batch_size=8, shuffle=True, seed=0)

for images, labels in loader:
    print(images.shape, labels.shape)
```

`ImageList` is also available when you already have explicit `(path, target)`
pairs. `DetectionFolder` and `SegmentationFolder` cover simple detection and
segmentation folder layouts; see the dedicated pages for expected file
structure.

## Models

```python
from tensorstudio import nn, optim

model = ts.vision.ImageClassifier((3, 32, 32), num_classes=10, channels=(8, 16))
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for images, labels in loader:
    optimizer.zero_grad()
    logits = model(images)
    loss = loss_fn(logits, labels.astype("int64"))
    loss.backward()
    optimizer.step()
```

Available model helpers:

- `ConvBlock`
- `ResidualBlock`
- `DepthwiseSeparableBlock`
- `TinyConvClassifier`
- `ImageClassifier`
- `CompactUNet`
- `make_cnn_classifier`
- `make_image_classifier`
- `make_unet`

These models use TensorStudio native-backed `nn.Conv2d`, `nn.DepthwiseConv2d`,
`nn.ConvTranspose2d`, `nn.ReLU`, `nn.MaxPool2d`, `nn.Flatten`, and `nn.Linear`.

## Metrics

```python
acc = ts.vision.accuracy(logits, labels.astype("int64"))
top1, top5 = ts.vision.top_k_accuracy(logits, labels.astype("int64"), k=(1, 5))
matrix = ts.vision.confusion_matrix(logits, labels.astype("int64"), num_classes=10)
iou = ts.vision.box_iou([[0, 0, 10, 10]], [[5, 5, 15, 15]])
mask_iou = ts.vision.mask_iou(prediction_mask, target_mask, num_classes=3)
```

## Visualization

```python
grid = ts.vision.make_grid(images, nrow=4, padding=2)
boxed = ts.vision.draw_bounding_boxes(image, [[2, 2, 20, 20]], labels=["cat"])
predictions = ts.vision.draw_predictions(image, [[2, 2, 20, 20]], ["cat"], scores=[0.92])
masked = ts.vision.overlay_mask(image, mask)
features = ts.vision.feature_map_grid(model_features)
ts.vision.save_image(grid, "batch.png")
ts.vision.save_image(boxed, "boxed.png")
```

## ONNX Export

Vision classifiers export through their supported TensorStudio layer stacks:

```python
model = ts.vision.ImageClassifier((3, 32, 32), num_classes=10, channels=(8, 16))
ts.export_onnx(model, "classifier.onnx", input_shape=(1, 3, 32, 32))
```

## Current Limits

- CPU-only image models.
- No pretrained model zoo yet.
- Detection and segmentation helpers are utilities, not full training
  frameworks.
- No video IO.
- No GPU image kernels or accelerated image decode path.
- Transform operations that convert through NumPy are intended for input
  preprocessing, not differentiable augmentation.
