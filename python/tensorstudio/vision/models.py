"""Small vision model builders."""

from __future__ import annotations

from tensorstudio.nn import (
    Conv2d,
    ConvTranspose2d,
    DepthwiseConv2d,
    Flatten,
    Linear,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from tensorstudio.ops import concat
from tensorstudio.tensor import Tensor


class ConvBlock(Module):
    """Conv2d + ReLU block with optional 2x2 max pooling."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] = 3,
        padding: int | tuple[int, int] = 1,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.pool = pool
        layers: list[Module] = [
            Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            ReLU(),
        ]
        if pool:
            layers.append(MaxPool2d(kernel_size=2))
        self.layers = Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, padding={self.padding}, pool={self.pool}"
        )


class ResidualBlock(Module):
    """Small ResNet-style residual block for NCHW tensors."""

    def __init__(self, channels: int, hidden_channels: int | None = None) -> None:
        super().__init__()
        if channels <= 0:
            raise ValueError("channels must be positive")
        hidden = channels if hidden_channels is None else hidden_channels
        if hidden <= 0:
            raise ValueError("hidden_channels must be positive")
        self.channels = channels
        self.hidden_channels = hidden
        self.layers = Sequential(
            Conv2d(channels, hidden, kernel_size=3, padding=1),
            ReLU(),
            Conv2d(hidden, channels, kernel_size=3, padding=1),
        )
        self.activation = ReLU()

    def forward(self, input: Tensor) -> Tensor:
        return self.activation(self.layers(input) + input)

    def extra_repr(self) -> str:
        return f"channels={self.channels}, hidden_channels={self.hidden_channels}"


class DepthwiseSeparableBlock(Module):
    """MobileNet-style depthwise convolution followed by pointwise projection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] = 1,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.layers = Sequential(
            DepthwiseConv2d(in_channels, kernel_size=3, stride=stride, padding=1),
            ReLU(),
            Conv2d(in_channels, out_channels, kernel_size=1),
            ReLU(),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.layers(input)

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"stride={self.stride}"
        )


class CompactUNet(Module):
    """Compact one-level UNet for small segmentation experiments."""

    def __init__(self, in_channels: int, num_classes: int, base_channels: int = 8) -> None:
        super().__init__()
        if in_channels <= 0 or num_classes <= 0 or base_channels <= 0:
            raise ValueError("in_channels, num_classes, and base_channels must be positive")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.enc1 = _double_conv(in_channels, base_channels)
        self.pool = MaxPool2d(kernel_size=2)
        self.bridge = _double_conv(base_channels, base_channels * 2)
        self.up = ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = _double_conv(base_channels * 2, base_channels)
        self.head = Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, input: Tensor) -> Tensor:
        skip = self.enc1(input)
        bridge = self.bridge(self.pool(skip))
        upsampled = self.up(bridge)
        if upsampled.shape[-2:] != skip.shape[-2:]:
            upsampled = upsampled[..., : skip.shape[-2], : skip.shape[-1]]
        merged = concat([upsampled, skip], axis=1)
        return self.head(self.dec1(merged))

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, num_classes={self.num_classes}, "
            f"base_channels={self.base_channels}"
        )


def _double_conv(in_channels: int, out_channels: int) -> Sequential:
    return Sequential(
        Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        ReLU(),
        Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        ReLU(),
    )


class ImageClassifier(Module):
    """Configurable small CNN image classifier."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        channels: tuple[int, ...] | list[int] = (8, 16),
    ) -> None:
        super().__init__()
        in_channels, height, width = (int(value) for value in input_shape)
        if in_channels <= 0 or height <= 1 or width <= 1:
            raise ValueError("input_shape must be (channels, height, width) with positive sizes")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        hidden_channels = [int(channel) for channel in channels]
        if not hidden_channels or any(channel <= 0 for channel in hidden_channels):
            raise ValueError("channels must contain positive integers")

        layers: list[Module] = []
        current_channels = in_channels
        current_h, current_w = height, width
        for out_channels in hidden_channels:
            if current_h < 2 or current_w < 2:
                raise ValueError("input spatial dimensions became too small for pooling")
            layers.append(ConvBlock(current_channels, out_channels, pool=True))
            current_channels = out_channels
            current_h = (current_h - 2) // 2 + 1
            current_w = (current_w - 2) // 2 + 1
        layers.extend(
            [
                Flatten(),
                Linear(current_channels * current_h * current_w, num_classes),
            ]
        )
        self.input_shape = (in_channels, height, width)
        self.num_classes = num_classes
        self.channels = tuple(hidden_channels)
        self.model = Sequential(*layers)

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def extra_repr(self) -> str:
        return (
            f"input_shape={self.input_shape}, num_classes={self.num_classes}, "
            f"channels={self.channels}"
        )


class TinyConvClassifier(Module):
    """A compact Conv2d image classifier for small image-recognition examples."""

    def __init__(
        self,
        input_shape: tuple[int, int, int],
        num_classes: int,
        hidden_channels: int = 8,
    ) -> None:
        super().__init__()
        channels, height, width = (int(value) for value in input_shape)
        if channels <= 0 or height <= 1 or width <= 1:
            raise ValueError("input_shape must be (channels, height, width) with positive sizes")
        if num_classes <= 0 or hidden_channels <= 0:
            raise ValueError("num_classes and hidden_channels must be positive")
        pooled_h = (height - 2) // 2 + 1
        pooled_w = (width - 2) // 2 + 1
        if pooled_h <= 0 or pooled_w <= 0:
            raise ValueError("input spatial dimensions are too small for the default pooling layer")
        self.input_shape = (channels, height, width)
        self.num_classes = num_classes
        self.hidden_channels = hidden_channels
        self.model = Sequential(
            Conv2d(channels, hidden_channels, kernel_size=3, padding=1),
            ReLU(),
            MaxPool2d(kernel_size=2),
            Flatten(),
            Linear(hidden_channels * pooled_h * pooled_w, num_classes),
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.model(input)

    def extra_repr(self) -> str:
        return (
            f"input_shape={self.input_shape}, num_classes={self.num_classes}, "
            f"hidden_channels={self.hidden_channels}"
        )


def make_cnn_classifier(
    input_shape: tuple[int, int, int],
    num_classes: int,
    hidden_channels: int = 8,
) -> Sequential:
    """Build a small Sequential CNN classifier."""

    return TinyConvClassifier(input_shape, num_classes, hidden_channels).model


def make_image_classifier(
    input_shape: tuple[int, int, int],
    num_classes: int,
    channels: tuple[int, ...] | list[int] = (8, 16),
) -> Sequential:
    """Build a configurable Sequential CNN classifier."""

    return ImageClassifier(input_shape, num_classes, channels).model


def make_unet(in_channels: int, num_classes: int, base_channels: int = 8) -> CompactUNet:
    """Build a compact UNet segmentation model."""

    return CompactUNet(in_channels, num_classes, base_channels)


__all__ = [
    "CompactUNet",
    "ConvBlock",
    "DepthwiseSeparableBlock",
    "ImageClassifier",
    "ResidualBlock",
    "TinyConvClassifier",
    "make_cnn_classifier",
    "make_image_classifier",
    "make_unet",
]
