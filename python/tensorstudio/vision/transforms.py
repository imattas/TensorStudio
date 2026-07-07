"""Image preprocessing helpers for TensorStudio vision workflows."""

from __future__ import annotations

from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor, from_numpy, tensor

_VALID_DTYPES = {"float32", "float64", "int32", "int64", "bool"}


def _as_array(image: Any) -> np.ndarray:
    if isinstance(image, Tensor):
        return image.numpy()
    array = np.asarray(image)
    if array.size == 0:
        raise ValueError("image must not be empty")
    return array


def _is_channel_count(value: int) -> bool:
    return value in {1, 3, 4}


def _to_channel_first(array: np.ndarray, channels_first: bool | None) -> np.ndarray:
    if array.ndim == 2:
        return array[np.newaxis, :, :]
    if array.ndim == 3:
        if channels_first is None:
            channels_first = _is_channel_count(array.shape[0]) and not _is_channel_count(
                array.shape[-1]
            )
        return array if channels_first else np.transpose(array, (2, 0, 1))
    if array.ndim == 4:
        if channels_first is None:
            channels_first = _is_channel_count(array.shape[1]) and not _is_channel_count(
                array.shape[-1]
            )
        return array if channels_first else np.transpose(array, (0, 3, 1, 2))
    raise ValueError("image must have shape HxW, HxWxC, CxHxW, NHWC, or NCHW")


def to_tensor(
    image: Any,
    dtype: str = "float32",
    scale: bool = True,
    channels_first: bool | None = None,
) -> Tensor:
    """Convert a NumPy/Pillow/Tensor image to a channel-first TensorStudio tensor.

    ``uint8`` inputs are scaled to ``[0, 1]`` by default. Floating-point inputs
    are copied as-is unless an explicit dtype conversion is requested.
    """

    if dtype not in _VALID_DTYPES:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    array = _to_channel_first(_as_array(image), channels_first)
    if scale and np.issubdtype(array.dtype, np.integer):
        array = array.astype(np.float32) / 255.0
    else:
        array = array.astype(np.float32 if dtype == "float32" else array.dtype, copy=False)
    if dtype != "float32":
        array = array.astype(_numpy_dtype(dtype), copy=False)
    return from_numpy(np.ascontiguousarray(array))


def normalize(
    input: Tensor,
    mean: float | list[float] | tuple[float, ...],
    std: float | list[float] | tuple[float, ...],
) -> Tensor:
    """Normalize a CHW or NCHW tensor with per-channel mean and std."""

    channels = input.shape[0] if input.ndim == 3 else input.shape[1] if input.ndim == 4 else None
    if channels is None:
        raise ValueError("normalize expects a CHW or NCHW tensor")
    mean_values = _channel_values(mean, channels, "mean")
    std_values = _channel_values(std, channels, "std")
    if any(value == 0.0 for value in std_values):
        raise ValueError("std values must be non-zero")

    shape = (channels, 1, 1) if input.ndim == 3 else (1, channels, 1, 1)
    mean_tensor = tensor(mean_values, dtype=input.dtype).reshape(shape)
    std_tensor = tensor(std_values, dtype=input.dtype).reshape(shape)
    return (input - mean_tensor) / std_tensor


def center_crop(image: Any, size: int | tuple[int, int]) -> np.ndarray | Tensor:
    """Return a centered crop from an image-like object.

    Tensor inputs return Tensor outputs; other image-like inputs return NumPy
    arrays. Both HWC and CHW layouts are supported.
    """

    crop_h, crop_w = _pair(size, "size")
    array = _as_array(image)
    height_axis, width_axis = _spatial_axes(array)
    height = array.shape[height_axis]
    width = array.shape[width_axis]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size cannot exceed image spatial dimensions")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    slices = [slice(None)] * array.ndim
    slices[height_axis] = slice(top, top + crop_h)
    slices[width_axis] = slice(left, left + crop_w)
    cropped = np.ascontiguousarray(array[tuple(slices)])
    if isinstance(image, Tensor):
        return from_numpy(cropped)
    return cropped


def resize_nearest(image: Any, size: int | tuple[int, int]) -> np.ndarray | Tensor:
    """Resize an image-like object with deterministic nearest-neighbor sampling."""

    out_h, out_w = _pair(size, "size")
    array = _as_array(image)
    height_axis, width_axis = _spatial_axes(array)
    height = array.shape[height_axis]
    width = array.shape[width_axis]
    row_indices = np.minimum((np.arange(out_h) * height // out_h), height - 1)
    col_indices = np.minimum((np.arange(out_w) * width // out_w), width - 1)
    resized = np.take(array, row_indices, axis=height_axis)
    resized = np.take(resized, col_indices, axis=width_axis)
    resized = np.ascontiguousarray(resized)
    if isinstance(image, Tensor):
        return from_numpy(resized)
    return resized


def _channel_values(
    values: float | list[float] | tuple[float, ...],
    channels: int,
    name: str,
) -> list[float]:
    if isinstance(values, (int, float)):
        return [float(values)] * channels
    result = [float(value) for value in values]
    if len(result) != channels:
        raise ValueError(f"{name} must be a scalar or contain one value per channel")
    return result


def _pair(value: int | tuple[int, int] | list[int], name: str) -> tuple[int, int]:
    if isinstance(value, int):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"{name} must be an int or a pair of ints")
    first, second = int(value[0]), int(value[1])
    if first <= 0 or second <= 0:
        raise ValueError(f"{name} values must be positive")
    return (first, second)


def _spatial_axes(array: np.ndarray) -> tuple[int, int]:
    if array.ndim == 2:
        return (0, 1)
    if array.ndim == 3:
        if _is_channel_count(array.shape[0]) and not _is_channel_count(array.shape[-1]):
            return (1, 2)
        return (0, 1)
    if array.ndim == 4:
        if _is_channel_count(array.shape[1]) and not _is_channel_count(array.shape[-1]):
            return (2, 3)
        return (1, 2)
    raise ValueError("image must have shape HxW, HxWxC, CxHxW, NHWC, or NCHW")


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    mapping = {
        "float32": np.dtype(np.float32),
        "float64": np.dtype(np.float64),
        "int32": np.dtype(np.int32),
        "int64": np.dtype(np.int64),
        "bool": np.dtype(np.bool_),
    }
    return mapping[dtype]


__all__ = ["center_crop", "normalize", "resize_nearest", "to_tensor"]
