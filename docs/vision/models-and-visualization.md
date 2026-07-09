# Models And Visualization

The vision model helpers are compact building blocks intended for learning,
experimentation, and small CPU examples.

## Model Blocks

```python
import tensorstudio as ts

residual = ts.vision.ResidualBlock(channels=8)
mobile = ts.vision.DepthwiseSeparableBlock(8, 16)
unet = ts.vision.CompactUNet(in_channels=1, num_classes=2, base_channels=8)
```

`ResidualBlock` is a small ResNet-style block. `DepthwiseSeparableBlock` follows
the MobileNet pattern of depthwise convolution plus pointwise projection.
`CompactUNet` is a one-level segmentation model using native convolution,
pooling, transposed convolution, and concatenation.

```python
x = ts.randn((1, 1, 64, 64), seed=7)
logits = unet(x)
print(logits.shape)  # (1, 2, 64, 64)
```

## Visualization

```python
import numpy as np

image = np.zeros((64, 64, 3), dtype=np.uint8)
boxes = [[8, 8, 40, 40]]

boxed = ts.vision.draw_predictions(image, boxes, labels=["object"], scores=[0.91])
masked = ts.vision.overlay_mask(image, np.ones((64, 64), dtype=np.uint8), alpha=0.4)
grid = ts.vision.feature_map_grid(ts.randn((1, 8, 16, 16), seed=3), nrow=4)
```

Visualization helpers return NumPy arrays and require Pillow only for text/box
drawing.

## Scope

These helpers do not claim to be pretrained production models. They are a
useful starting point for testing TensorStudio vision operations, building
examples, and experimenting with small models on CPU.
