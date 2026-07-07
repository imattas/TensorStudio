"""Small vision model builders."""

from __future__ import annotations

from tensorstudio.nn import Conv2d, Flatten, Linear, MaxPool2d, Module, ReLU, Sequential
from tensorstudio.tensor import Tensor


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


__all__ = ["TinyConvClassifier", "make_cnn_classifier"]
