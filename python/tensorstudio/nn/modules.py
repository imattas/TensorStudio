"""Small neural network module system built on TensorStudio tensors."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

from tensorstudio.tensor import Tensor, ones, rand, randn, tensor, zeros

PairLike = int | tuple[int, int] | list[int]


def _pair(value: PairLike, name: str) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair of ints")
    return (int(value[0]), int(value[1]))


class Parameter:
    """Factory class that creates tensors with ``requires_grad=True``."""

    def __new__(cls, data: Any, dtype: str | None = None) -> Tensor:  # type: ignore[misc]
        if isinstance(data, Tensor):
            data.requires_grad = True
            return data
        return tensor(data, dtype=dtype, requires_grad=True)


class Module:
    """Base class for Python-level neural network modules."""

    training: bool

    def __init__(self) -> None:
        self.training = True
        self._buffers: dict[str, Tensor] = {}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        return [parameter for _, parameter in self.named_parameters()]

    def trainable_parameters(self) -> list[Tensor]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        seen: set[int] = set()
        return list(self._iter_named_parameters(prefix, seen))

    def register_buffer(self, name: str, tensor: Tensor) -> None:
        if not name or "." in name:
            raise ValueError("buffer name must be non-empty and cannot contain '.'")
        tensor.requires_grad = False
        self._buffers[name] = tensor

    def buffers(self) -> list[Tensor]:
        return [buffer for _, buffer in self.named_buffers()]

    def named_buffers(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        seen: set[int] = set()
        return list(self._iter_named_buffers(prefix, seen))

    def children(self) -> list[Module]:
        return list(self._children())

    def named_children(self, prefix: str = "") -> list[tuple[str, Module]]:
        return list(self._named_children(prefix))

    def modules(self) -> list[Module]:
        result = [self]
        for child in self._children():
            result.extend(child.modules())
        return result

    def named_modules(self, prefix: str = "") -> list[tuple[str, Module]]:
        result = [(prefix, self)]
        for name, child in self._named_children(prefix):
            result.extend(child.named_modules(name))
        return result

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def requires_grad_(self, requires_grad: bool = True) -> Module:
        for parameter in self.parameters():
            parameter.requires_grad = requires_grad
        return self

    def freeze(self) -> Module:
        return self.requires_grad_(False)

    def unfreeze(self) -> Module:
        return self.requires_grad_(True)

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.trainable_parameters() if trainable_only else self.parameters()
        return sum(parameter.size for parameter in parameters)

    def summary(self, input_shape: tuple[int, ...] | list[int] | None = None) -> dict[str, Any]:
        from .summary import model_summary

        return model_summary(self, input_shape=input_shape)

    def apply(self, fn: Any) -> Module:
        fn(self)
        for module in self._children():
            module.apply(fn)
        return self

    def train(self) -> Module:
        self.training = True
        for module in self._children():
            module.train()
        return self

    def eval(self) -> Module:
        self.training = False
        for module in self._children():
            module.eval()
        return self

    def state_dict(self) -> dict[str, Tensor]:
        state = {name: parameter.detach().clone() for name, parameter in self.named_parameters()}
        state.update({name: buffer.detach().clone() for name, buffer in self.named_buffers()})
        return state

    def load_state_dict(self, state: dict[str, Tensor], strict: bool = True) -> None:
        current = dict(self.named_parameters())
        current.update(dict(self.named_buffers()))
        missing = [name for name in current if name not in state]
        unexpected = [name for name in state if name not in current]
        if strict and (missing or unexpected):
            raise ValueError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
        for name, value in state.items():
            if name in current:
                current[name]._assign(value)

    def extra_repr(self) -> str:
        return ""

    def __repr__(self) -> str:
        children = list(self._named_children())
        extra = self.extra_repr()
        if not children:
            if extra:
                return f"{self.__class__.__name__}({extra})"
            return f"{self.__class__.__name__}()"
        body = ", ".join(f"{name}={module!r}" for name, module in children)
        if extra:
            body = f"{extra}, {body}"
        return f"{self.__class__.__name__}({body})"

    def _children(self) -> Iterator[Module]:
        for _, module in self._named_children():
            yield module

    def _named_children(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        for name, value in self.__dict__.items():
            if name in {"training", "_buffers"}:
                continue
            yield from self._modules_from_value(name, value, prefix)

    def _iter_parameters(self, seen: set[int]) -> Iterator[Tensor]:
        for _, parameter in self._iter_named_parameters("", seen):
            yield parameter

    def _iter_named_parameters(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
        for name, value in self.__dict__.items():
            if name == "_buffers":
                continue
            yield from self._named_parameters_from_value(name, value, prefix, seen)

    def _iter_named_buffers(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
        for name, buffer in self._buffers.items():
            marker = id(buffer)
            if marker in seen:
                continue
            seen.add(marker)
            yield self._join_name(prefix, name), buffer
        for child_name, child in self._named_children(prefix):
            yield from child._iter_named_buffers(child_name, seen)

    @classmethod
    def _join_name(cls, prefix: str, name: str) -> str:
        if not prefix:
            return name
        if not name:
            return prefix
        return f"{prefix}.{name}"

    @classmethod
    def _named_parameters_from_value(
        cls,
        name: str,
        value: Any,
        prefix: str,
        seen: set[int],
    ) -> Iterator[tuple[str, Tensor]]:
        full_name = cls._join_name(prefix, name)
        if isinstance(value, Tensor):
            marker = id(value)
            if marker not in seen:
                seen.add(marker)
                yield full_name, value
        elif isinstance(value, Module):
            yield from value._iter_named_parameters(full_name, seen)
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from cls._named_parameters_from_value(str(key), item, full_name, seen)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from cls._named_parameters_from_value(str(index), item, full_name, seen)

    @classmethod
    def _modules_from_value(
        cls,
        name: str,
        value: Any,
        prefix: str,
    ) -> Iterator[tuple[str, Module]]:
        full_name = cls._join_name(prefix, name)
        if isinstance(value, Module):
            yield full_name, value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from cls._modules_from_value(str(key), item, full_name)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from cls._modules_from_value(str(index), item, full_name)


class Linear(Module):
    """Fully connected layer with shape ``input @ weight.T + bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        scale = 1.0 / math.sqrt(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = cast(Tensor, Parameter(randn((out_features, in_features)) * scale))
        self.bias = cast(Tensor, Parameter(zeros((out_features,)))) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        from .functional import linear

        return linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class Conv2d(Module):
    """2D NCHW convolution layer backed by TensorStudio's native CPU kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: PairLike,
        stride: PairLike = 1,
        padding: PairLike = 0,
        dilation: PairLike = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("groups must be positive and divide in_channels and out_channels")
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive")
        stride_h, stride_w = _pair(stride, "stride")
        padding_h, padding_w = _pair(padding, "padding")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("stride values must be positive")
        if padding_h < 0 or padding_w < 0:
            raise ValueError("padding values must be non-negative")
        if dilation_h <= 0 or dilation_w <= 0:
            raise ValueError("dilation values must be positive")

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups

        scale = 1.0 / math.sqrt((in_channels // groups) * kernel_h * kernel_w)
        self.weight = cast(
            Tensor,
            Parameter(randn((out_channels, in_channels // groups, kernel_h, kernel_w)) * scale),
        )
        self.bias = cast(Tensor, Parameter(zeros((out_channels,)))) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        from .functional import conv2d

        return conv2d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}, groups={self.groups}, "
            f"bias={self.bias is not None}"
        )


class Conv1d(Module):
    """1D NCL convolution layer backed by TensorStudio's native 2D CPU kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0 or kernel_size <= 0:
            raise ValueError("in_channels, out_channels, and kernel_size must be positive")
        if stride <= 0 or dilation <= 0:
            raise ValueError("stride and dilation must be positive")
        if padding < 0:
            raise ValueError("padding must be non-negative")
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("groups must be positive and divide in_channels and out_channels")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        scale = 1.0 / math.sqrt((in_channels // groups) * kernel_size)
        self.weight = cast(
            Tensor,
            Parameter(randn((out_channels, in_channels // groups, kernel_size)) * scale),
        )
        self.bias = cast(Tensor, Parameter(zeros((out_channels,)))) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        from .functional import conv1d

        return conv1d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"dilation={self.dilation}, groups={self.groups}, bias={self.bias is not None}"
        )


class DepthwiseConv2d(Conv2d):
    """Depthwise 2D convolution with ``groups=in_channels``."""

    def __init__(
        self,
        in_channels: int,
        kernel_size: PairLike,
        stride: PairLike = 1,
        padding: PairLike = 0,
        dilation: PairLike = 1,
        depth_multiplier: int = 1,
        bias: bool = True,
    ) -> None:
        if depth_multiplier <= 0:
            raise ValueError("depth_multiplier must be positive")
        self.depth_multiplier = depth_multiplier
        super().__init__(
            in_channels,
            in_channels * depth_multiplier,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=in_channels,
            bias=bias,
        )

    def extra_repr(self) -> str:
        return f"{super().extra_repr()}, depth_multiplier={self.depth_multiplier}"


class ConvTranspose2d(Module):
    """2D NCHW transposed convolution backed by TensorStudio's native CPU kernel."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: PairLike,
        stride: PairLike = 1,
        padding: PairLike = 0,
        output_padding: PairLike = 0,
        dilation: PairLike = 1,
        groups: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("in_channels and out_channels must be positive")
        if groups <= 0 or in_channels % groups != 0 or out_channels % groups != 0:
            raise ValueError("groups must be positive and divide in_channels and out_channels")
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        stride_h, stride_w = _pair(stride, "stride")
        padding_h, padding_w = _pair(padding, "padding")
        output_padding_h, output_padding_w = _pair(output_padding, "output_padding")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive")
        if stride_h <= 0 or stride_w <= 0 or dilation_h <= 0 or dilation_w <= 0:
            raise ValueError("stride and dilation values must be positive")
        if padding_h < 0 or padding_w < 0 or output_padding_h < 0 or output_padding_w < 0:
            raise ValueError("padding and output_padding values must be non-negative")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.output_padding = (output_padding_h, output_padding_w)
        self.dilation = (dilation_h, dilation_w)
        self.groups = groups
        scale = 1.0 / math.sqrt((out_channels // groups) * kernel_h * kernel_w)
        self.weight = cast(
            Tensor,
            Parameter(randn((in_channels, out_channels // groups, kernel_h, kernel_w)) * scale),
        )
        self.bias = cast(Tensor, Parameter(zeros((out_channels,)))) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        from .functional import conv_transpose2d

        return conv_transpose2d(
            input,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=self.output_padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"output_padding={self.output_padding}, dilation={self.dilation}, "
            f"groups={self.groups}, bias={self.bias is not None}"
        )


class MaxPool2d(Module):
    """2D NCHW max pooling layer backed by TensorStudio's native CPU kernel."""

    def __init__(
        self,
        kernel_size: PairLike,
        stride: PairLike | None = None,
        padding: PairLike = 0,
        dilation: PairLike = 1,
    ) -> None:
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive")
        stride_h, stride_w = _pair(kernel_size if stride is None else stride, "stride")
        padding_h, padding_w = _pair(padding, "padding")
        dilation_h, dilation_w = _pair(dilation, "dilation")
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("stride values must be positive")
        if padding_h < 0 or padding_w < 0:
            raise ValueError("padding values must be non-negative")
        if dilation_h <= 0 or dilation_w <= 0:
            raise ValueError("dilation values must be positive")
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.dilation = (dilation_h, dilation_w)

    def forward(self, input: Tensor) -> Tensor:
        from .functional import max_pool2d

        return max_pool2d(
            input,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, dilation={self.dilation}"
        )


class AvgPool2d(Module):
    """2D NCHW average pooling layer backed by TensorStudio's native CPU kernel."""

    def __init__(
        self,
        kernel_size: PairLike,
        stride: PairLike | None = None,
        padding: PairLike = 0,
        count_include_pad: bool = False,
    ) -> None:
        super().__init__()
        kernel_h, kernel_w = _pair(kernel_size, "kernel_size")
        if kernel_h <= 0 or kernel_w <= 0:
            raise ValueError("kernel_size values must be positive")
        stride_h, stride_w = _pair(kernel_size if stride is None else stride, "stride")
        padding_h, padding_w = _pair(padding, "padding")
        if stride_h <= 0 or stride_w <= 0:
            raise ValueError("stride values must be positive")
        if padding_h < 0 or padding_w < 0:
            raise ValueError("padding values must be non-negative")
        self.kernel_size = (kernel_h, kernel_w)
        self.stride = (stride_h, stride_w)
        self.padding = (padding_h, padding_w)
        self.count_include_pad = count_include_pad

    def forward(self, input: Tensor) -> Tensor:
        from .functional import avg_pool2d

        return avg_pool2d(
            input,
            self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            count_include_pad=self.count_include_pad,
        )

    def extra_repr(self) -> str:
        return (
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, count_include_pad={self.count_include_pad}"
        )


class AdaptiveAvgPool2d(Module):
    """Adaptive average pooling to a target ``(height, width)`` output size."""

    def __init__(self, output_size: PairLike) -> None:
        super().__init__()
        out_h, out_w = _pair(output_size, "output_size")
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_size values must be positive")
        self.output_size = (out_h, out_w)

    def forward(self, input: Tensor) -> Tensor:
        from .functional import adaptive_avg_pool2d

        return adaptive_avg_pool2d(input, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class AdaptiveMaxPool2d(Module):
    """Adaptive max pooling to a target ``(height, width)`` output size."""

    def __init__(self, output_size: PairLike) -> None:
        super().__init__()
        out_h, out_w = _pair(output_size, "output_size")
        if out_h <= 0 or out_w <= 0:
            raise ValueError("output_size values must be positive")
        self.output_size = (out_h, out_w)

    def forward(self, input: Tensor) -> Tensor:
        from .functional import adaptive_max_pool2d

        return adaptive_max_pool2d(input, self.output_size)

    def extra_repr(self) -> str:
        return f"output_size={self.output_size}"


class GlobalAvgPool2d(Module):
    """Global average pooling over NCHW spatial dimensions."""

    def __init__(self, keepdims: bool = False) -> None:
        super().__init__()
        self.keepdims = keepdims

    def forward(self, input: Tensor) -> Tensor:
        from .functional import global_avg_pool2d

        return global_avg_pool2d(input, keepdims=self.keepdims)

    def extra_repr(self) -> str:
        return f"keepdims={self.keepdims}"


class GlobalMaxPool2d(Module):
    """Global max pooling over NCHW spatial dimensions."""

    def __init__(self, keepdims: bool = False) -> None:
        super().__init__()
        self.keepdims = keepdims

    def forward(self, input: Tensor) -> Tensor:
        from .functional import global_max_pool2d

        return global_max_pool2d(input, keepdims=self.keepdims)

    def extra_repr(self) -> str:
        return f"keepdims={self.keepdims}"


class BatchNorm1d(Module):
    """Batch normalization over ``(N, C)`` or ``(N, C, L)`` inputs."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.weight = cast(Tensor, Parameter(ones((num_features,)))) if affine else None
        self.bias = cast(Tensor, Parameter(zeros((num_features,)))) if affine else None
        if track_running_stats:
            self.register_buffer("running_mean", zeros((num_features,)))
            self.register_buffer("running_var", ones((num_features,)))

    def forward(self, input: Tensor) -> Tensor:
        from .functional import batch_norm

        if input.ndim not in {2, 3}:
            raise ValueError("BatchNorm1d expects input with shape (N, C) or (N, C, L)")
        return batch_norm(
            input,
            self.weight,
            self.bias,
            self._buffers.get("running_mean"),
            self._buffers.get("running_var"),
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class BatchNorm2d(Module):
    """Batch normalization over NCHW inputs."""

    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> None:
        super().__init__()
        if num_features <= 0:
            raise ValueError("num_features must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if not 0.0 <= momentum <= 1.0:
            raise ValueError("momentum must be in [0, 1]")
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.weight = cast(Tensor, Parameter(ones((num_features,)))) if affine else None
        self.bias = cast(Tensor, Parameter(zeros((num_features,)))) if affine else None
        if track_running_stats:
            self.register_buffer("running_mean", zeros((num_features,)))
            self.register_buffer("running_var", ones((num_features,)))

    def forward(self, input: Tensor) -> Tensor:
        from .functional import batch_norm

        if input.ndim != 4:
            raise ValueError("BatchNorm2d expects input with shape (N, C, H, W)")
        return batch_norm(
            input,
            self.weight,
            self.bias,
            self._buffers.get("running_mean"),
            self._buffers.get("running_var"),
            training=self.training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def extra_repr(self) -> str:
        return (
            f"num_features={self.num_features}, eps={self.eps}, momentum={self.momentum}, "
            f"affine={self.affine}, track_running_stats={self.track_running_stats}"
        )


class LayerNorm(Module):
    """Layer normalization over the last ``normalized_shape`` dimensions."""

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...] | list[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        shape = (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )
        if not shape or any(dim <= 0 for dim in shape):
            raise ValueError("normalized_shape must contain positive dimensions")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.normalized_shape = shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.weight = cast(Tensor, Parameter(ones(shape))) if elementwise_affine else None
        self.bias = cast(Tensor, Parameter(zeros(shape))) if elementwise_affine else None

    def forward(self, input: Tensor) -> Tensor:
        from .functional import layer_norm

        return layer_norm(input, self.normalized_shape, self.weight, self.bias, eps=self.eps)

    def extra_repr(self) -> str:
        return (
            f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class Embedding(Module):
    """Lookup table for integer token or category ids."""

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        scale = 1.0 / math.sqrt(embedding_dim)
        self.weight = cast(Tensor, Parameter(randn((num_embeddings, embedding_dim)) * scale))

    def forward(self, input: Tensor) -> Tensor:
        from .functional import embedding

        return embedding(input, self.weight)

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class Sequential(Module):
    """A simple ordered container of modules."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.layers = list(modules)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for module in self.layers:
            output = module(output)
        return output

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Module:
        return self.layers[index]

    def _named_children(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        for index, module in enumerate(self.layers):
            yield self._join_name(prefix, str(index)), module

    def _iter_named_parameters(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
        for index, module in enumerate(self.layers):
            yield from module._iter_named_parameters(self._join_name(prefix, str(index)), seen)


class Identity(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, input: Tensor) -> Tensor:
        from .functional import leaky_relu

        return leaky_relu(input, negative_slope=self.negative_slope)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()


class Softplus(Module):
    def forward(self, input: Tensor) -> Tensor:
        from .functional import softplus

        return softplus(input)


class GELU(Module):
    def forward(self, input: Tensor) -> Tensor:
        from .functional import gelu

        return gelu(input)


class ELU(Module):
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, input: Tensor) -> Tensor:
        from .functional import elu

        return elu(input, alpha=self.alpha)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class SELU(Module):
    def forward(self, input: Tensor) -> Tensor:
        from .functional import selu

        return selu(input)


class SiLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        from .functional import silu

        return silu(input)


class Mish(Module):
    def forward(self, input: Tensor) -> Tensor:
        from .functional import mish

        return mish(input)


class Dropout(Module):
    """Inverted dropout using TensorStudio random tensors."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must satisfy 0 <= p < 1")
        self.p = p
        self.seed = seed

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        mask = rand(input.shape, dtype=input.dtype, seed=self.seed) > self.p
        return input * mask / (1.0 - self.p)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class Flatten(Module):
    """Flatten a tensor from ``start_dim`` through the last dimension."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, input: Tensor) -> Tensor:
        ndim = input.ndim
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        if start < 0 or start > ndim:
            raise ValueError("start_dim is out of range")
        prefix = input.shape[:start]
        flattened = math.prod(input.shape[start:]) if start < ndim else 1
        return input.reshape((*prefix, flattened))

    def extra_repr(self) -> str:
        return f"start_dim={self.start_dim}"


__all__ = [
    "AdaptiveAvgPool2d",
    "AdaptiveMaxPool2d",
    "AvgPool2d",
    "BatchNorm1d",
    "BatchNorm2d",
    "Conv1d",
    "Conv2d",
    "ConvTranspose2d",
    "DepthwiseConv2d",
    "Dropout",
    "ELU",
    "Embedding",
    "Flatten",
    "GELU",
    "GlobalAvgPool2d",
    "GlobalMaxPool2d",
    "Identity",
    "LayerNorm",
    "LeakyReLU",
    "Linear",
    "MaxPool2d",
    "Mish",
    "Module",
    "Parameter",
    "ReLU",
    "SELU",
    "Sequential",
    "SiLU",
    "Sigmoid",
    "Softplus",
    "Tanh",
]
