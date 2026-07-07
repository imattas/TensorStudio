# Vision

TensorStudio's vision package provides image-to-tensor preprocessing and small
CNN classifier helpers for lightweight image-recognition experiments.

The vision namespace is intentionally compact. Image decoding, augmentation
pipelines, pretrained model zoos, and production dataloading remain outside the
v1.2 scope.

## Install

NumPy image arrays work with the base package. Install the optional Pillow extra
when you want to pass Pillow images directly:

```bash
python -m pip install "tensorstudio[vision]"
```

## Image To Tensor

`to_tensor` accepts HWC, CHW, NHWC, NCHW, and grayscale image-like arrays. It
returns channel-first tensors and scales integer images to `[0, 1]` by default.

```python
import numpy as np
import tensorstudio as ts

image = np.zeros((8, 8, 3), dtype=np.uint8)
x = ts.vision.to_tensor(image)

print(x.shape)  # (3, 8, 8)
print(x.dtype)  # float32
```

## Preprocessing

```python
cropped = ts.vision.center_crop(image, (6, 6))
resized = ts.vision.resize_nearest(cropped, (8, 8))
x = ts.vision.to_tensor(resized).reshape((1, 3, 8, 8))
x = ts.vision.normalize(x, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
```

`normalize` works on CHW or NCHW TensorStudio tensors and keeps gradients flowing
when the input requires gradients.

## Tiny CNN Classifier

```python
from tensorstudio import nn, optim

model = ts.vision.TinyConvClassifier((3, 8, 8), num_classes=2, hidden_channels=4)
target = ts.tensor([1], dtype="int64")
optimizer = optim.Adam(model.parameters(), lr=0.01)

for _ in range(10):
    optimizer.zero_grad()
    logits = model(x)
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()
    optimizer.step()
```

The classifier uses:

- `nn.Conv2d`
- `nn.ReLU`
- `nn.MaxPool2d`
- `nn.Flatten`
- `nn.Linear`

Those layers run through TensorStudio tensor operations and autograd. The model
is small by design and meant for examples, tests, and experimentation.

## ONNX Export

The tiny classifier owns an exportable Sequential stack:

```python
ts.export_onnx(model, "tiny_classifier.onnx", input_shape=(1, 3, 8, 8))
```

See [ONNX Interchange](../Usage/interchange.md) for details and limits.
