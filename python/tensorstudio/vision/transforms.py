"""Image preprocessing and augmentation helpers for TensorStudio vision workflows."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

import numpy as np

from tensorstudio.tensor import Tensor, from_numpy, tensor

_VALID_DTYPES = {"float32", "float64", "int32", "int64", "bool"}
_ImageLike = Any


class Compose:
    """Compose several image transforms."""

    def __init__(self, transforms: list[Callable[[_ImageLike], _ImageLike]]) -> None:
        self.transforms = list(transforms)

    def __call__(self, image: _ImageLike) -> _ImageLike:
        output = image
        for transform in self.transforms:
            output = transform(output)
        return output


class ToTensor:
    """Callable wrapper around :func:`to_tensor`."""

    def __init__(
        self,
        dtype: str = "float32",
        scale: bool = True,
        channels_first: bool | None = None,
    ) -> None:
        self.dtype = dtype
        self.scale = scale
        self.channels_first = channels_first

    def __call__(self, image: _ImageLike) -> Tensor:
        return to_tensor(
            image,
            dtype=self.dtype,
            scale=self.scale,
            channels_first=self.channels_first,
        )


class Normalize:
    """Callable wrapper around :func:`normalize`."""

    def __init__(
        self,
        mean: float | list[float] | tuple[float, ...],
        std: float | list[float] | tuple[float, ...],
    ) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, image: Tensor) -> Tensor:
        return normalize(image, self.mean, self.std)


class Resize:
    """Resize image-like inputs using nearest or bilinear interpolation."""

    def __init__(self, size: int | tuple[int, int], interpolation: str = "bilinear") -> None:
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        self.size = size
        self.interpolation = interpolation

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        if self.interpolation == "nearest":
            return resize_nearest(image, self.size)
        return resize_bilinear(image, self.size)


class CenterCrop:
    """Callable wrapper around :func:`center_crop`."""

    def __init__(self, size: int | tuple[int, int]) -> None:
        self.size = size

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        return center_crop(image, self.size)


class RandomCrop:
    """Deterministic optional random crop transform."""

    def __init__(
        self,
        size: int | tuple[int, int],
        padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.padding = padding
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        return random_crop(image, self.size, padding=self.padding, seed=self.seed)


class RandomHorizontalFlip:
    """Flip an image horizontally with probability ``p``."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError("p must satisfy 0 <= p <= 1")
        self.p = p
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        return random_horizontal_flip(image, p=self.p, seed=self.seed)


class ColorJitter:
    """Randomly perturb brightness, contrast, saturation, and hue."""

    def __init__(
        self,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        hue: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.brightness = _jitter_bound(brightness, "brightness")
        self.contrast = _jitter_bound(contrast, "contrast")
        self.saturation = _jitter_bound(saturation, "saturation")
        if not 0.0 <= hue <= 0.5:
            raise ValueError("hue must satisfy 0 <= hue <= 0.5")
        self.hue = hue
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        return color_jitter(
            image,
            brightness=self.brightness,
            contrast=self.contrast,
            saturation=self.saturation,
            hue=self.hue,
            seed=self.seed,
        )


class RandomResizedCrop:
    """Crop a random area and resize to ``size``."""

    def __init__(
        self,
        size: int | tuple[int, int],
        scale: tuple[float, float] = (0.08, 1.0),
        ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation: str = "bilinear",
        seed: int | None = None,
    ) -> None:
        if interpolation not in {"nearest", "bilinear"}:
            raise ValueError("interpolation must be 'nearest' or 'bilinear'")
        if not 0.0 < scale[0] <= scale[1]:
            raise ValueError("scale must satisfy 0 < min <= max")
        if not 0.0 < ratio[0] <= ratio[1]:
            raise ValueError("ratio must satisfy 0 < min <= max")
        self.size = size
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        return random_resized_crop(
            image,
            self.size,
            scale=self.scale,
            ratio=self.ratio,
            interpolation=self.interpolation,
            seed=self.seed,
        )


class RandomRotation:
    """Rotate an image by a uniformly sampled angle in degrees."""

    def __init__(
        self,
        degrees: float | tuple[float, float],
        fill: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.degrees = _degree_range(degrees)
        self.fill = fill
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        rng = random.Random(self.seed)
        angle = rng.uniform(self.degrees[0], self.degrees[1])
        return rotate(image, angle, fill=self.fill)


class RandomAffine:
    """Apply a deterministic random affine transform."""

    def __init__(
        self,
        degrees: float | tuple[float, float],
        translate: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        shear: float | tuple[float, float] = 0.0,
        fill: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.degrees = _degree_range(degrees)
        if translate is not None and (
            len(translate) != 2 or any(value < 0.0 or value > 1.0 for value in translate)
        ):
            raise ValueError("translate must contain fractions in [0, 1]")
        if scale is not None and (scale[0] <= 0.0 or scale[1] <= 0.0 or scale[0] > scale[1]):
            raise ValueError("scale must satisfy 0 < min <= max")
        self.translate = translate
        self.scale = scale
        self.shear = _degree_range(shear)
        self.fill = fill
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        rng = random.Random(self.seed)
        angle = rng.uniform(self.degrees[0], self.degrees[1])
        scale = 1.0 if self.scale is None else rng.uniform(self.scale[0], self.scale[1])
        shear = rng.uniform(self.shear[0], self.shear[1])
        array = _as_array(image)
        height_axis, width_axis = _spatial_axes(array)
        max_dx = 0.0 if self.translate is None else self.translate[0] * array.shape[width_axis]
        max_dy = 0.0 if self.translate is None else self.translate[1] * array.shape[height_axis]
        translate = (rng.uniform(-max_dx, max_dx), rng.uniform(-max_dy, max_dy))
        return affine(
            image,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
            fill=self.fill,
        )


class Cutout:
    """Mask a random rectangle in an image or image batch."""

    def __init__(
        self,
        size: int | tuple[int, int],
        value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        self.size = size
        self.value = value
        self.seed = seed

    def __call__(self, image: _ImageLike) -> np.ndarray | Tensor:
        return cutout(image, self.size, value=self.value, seed=self.seed)


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


def to_numpy_image(
    image: Any,
    channels_last: bool = True,
    dtype: str | None = None,
    unscale: bool = False,
) -> np.ndarray:
    """Convert a TensorStudio/Pillow/NumPy image to a NumPy image array."""

    array = np.array(_as_array(image), copy=True)
    if channels_last:
        if isinstance(image, Tensor) and array.ndim == 3:
            array = np.moveaxis(array, 0, -1)
        elif isinstance(image, Tensor) and array.ndim == 4:
            array = np.moveaxis(array, 1, -1)
        else:
            array = _to_channel_last(array)
    if unscale:
        array = np.clip(array, 0.0, 1.0) * 255.0
    if dtype is not None:
        array = array.astype(_numpy_dtype(dtype), copy=False)
    return np.ascontiguousarray(array)


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
    """Return a centered crop from an image-like object."""

    crop_h, crop_w = _pair(size, "size")
    array = _as_array(image)
    height_axis, width_axis = _spatial_axes(array)
    height = array.shape[height_axis]
    width = array.shape[width_axis]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size cannot exceed image spatial dimensions")
    top = (height - crop_h) // 2
    left = (width - crop_w) // 2
    return _same_kind(image, _crop_array(array, top, left, crop_h, crop_w))


def random_crop(
    image: Any,
    size: int | tuple[int, int],
    padding: int | tuple[int, int] | tuple[int, int, int, int] = 0,
    seed: int | None = None,
) -> np.ndarray | Tensor:
    """Randomly crop an image-like object with optional zero padding."""

    crop_h, crop_w = _pair(size, "size")
    array = pad(image, padding) if padding != 0 else image
    padded = _as_array(array)
    height_axis, width_axis = _spatial_axes(padded)
    height = padded.shape[height_axis]
    width = padded.shape[width_axis]
    if crop_h > height or crop_w > width:
        raise ValueError("crop size cannot exceed image spatial dimensions")
    rng = random.Random(seed)
    top = rng.randint(0, height - crop_h)
    left = rng.randint(0, width - crop_w)
    return _same_kind(image, _crop_array(padded, top, left, crop_h, crop_w))


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
    return _same_kind(image, resized)


def resize_bilinear(image: Any, size: int | tuple[int, int]) -> np.ndarray | Tensor:
    """Resize an image-like object with bilinear interpolation."""

    out_h, out_w = _pair(size, "size")
    array = _as_array(image)
    height_axis, width_axis = _spatial_axes(array)
    height = array.shape[height_axis]
    width = array.shape[width_axis]
    working: np.ndarray = np.moveaxis(array, (height_axis, width_axis), (0, 1)).astype(np.float64)

    y_coords = np.zeros(out_h) if out_h == 1 else np.linspace(0, height - 1, out_h)
    x_coords = np.zeros(out_w) if out_w == 1 else np.linspace(0, width - 1, out_w)
    y0 = np.floor(y_coords).astype(np.int64)
    x0 = np.floor(x_coords).astype(np.int64)
    y1 = np.minimum(y0 + 1, height - 1)
    x1 = np.minimum(x0 + 1, width - 1)
    wy = (y_coords - y0).reshape(out_h, 1, *([1] * (working.ndim - 2)))
    wx = (x_coords - x0).reshape(1, out_w, *([1] * (working.ndim - 2)))

    top = working[y0][:, x0] * (1.0 - wx) + working[y0][:, x1] * wx
    bottom = working[y1][:, x0] * (1.0 - wx) + working[y1][:, x1] * wx
    resized = top * (1.0 - wy) + bottom * wy
    resized = np.moveaxis(resized, (0, 1), (height_axis, width_axis))
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        resized = np.rint(resized).clip(info.min, info.max).astype(array.dtype)
    else:
        resized = resized.astype(array.dtype, copy=False)
    return _same_kind(image, resized)


def horizontal_flip(image: Any) -> np.ndarray | Tensor:
    """Flip an image horizontally."""

    array = _as_array(image)
    _, width_axis = _spatial_axes(array)
    return _same_kind(image, np.flip(array, axis=width_axis))


def vertical_flip(image: Any) -> np.ndarray | Tensor:
    """Flip an image vertically."""

    array = _as_array(image)
    height_axis, _ = _spatial_axes(array)
    return _same_kind(image, np.flip(array, axis=height_axis))


def random_horizontal_flip(
    image: Any,
    p: float = 0.5,
    seed: int | None = None,
) -> np.ndarray | Tensor:
    """Flip an image horizontally with probability ``p``."""

    if not 0.0 <= p <= 1.0:
        raise ValueError("p must satisfy 0 <= p <= 1")
    if random.Random(seed).random() < p:
        return horizontal_flip(image)
    return _same_kind(image, _as_array(image))


def batch_resize(
    images: Any,
    size: int | tuple[int, int],
    interpolation: str = "bilinear",
) -> np.ndarray | Tensor:
    """Resize an NCHW or NHWC batch."""

    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    resize = resize_nearest if interpolation == "nearest" else resize_bilinear
    return _same_kind(images, _apply_batch(images, lambda item: _as_array(resize(item, size))))


def batch_center_crop(images: Any, size: int | tuple[int, int]) -> np.ndarray | Tensor:
    """Center-crop an NCHW or NHWC batch."""

    return _same_kind(images, _apply_batch(images, lambda item: _as_array(center_crop(item, size))))


def batch_normalize(
    images: Tensor,
    mean: float | list[float] | tuple[float, ...],
    std: float | list[float] | tuple[float, ...],
) -> Tensor:
    """Normalize a CHW or NCHW TensorStudio tensor."""

    return normalize(images, mean, std)


def color_jitter(
    image: Any,
    brightness: float = 0.0,
    contrast: float = 0.0,
    saturation: float = 0.0,
    hue: float = 0.0,
    seed: int | None = None,
) -> np.ndarray | Tensor:
    """Randomly perturb color properties with deterministic seeding."""

    brightness = _jitter_bound(brightness, "brightness")
    contrast = _jitter_bound(contrast, "contrast")
    saturation = _jitter_bound(saturation, "saturation")
    if not 0.0 <= hue <= 0.5:
        raise ValueError("hue must satisfy 0 <= hue <= 0.5")
    rng = random.Random(seed)
    output = _as_float_image(_as_array(image))
    channel_axis = _channel_axis(output)
    if brightness:
        output *= rng.uniform(max(0.0, 1.0 - brightness), 1.0 + brightness)
    if contrast:
        axes = tuple(axis for axis in range(output.ndim) if axis != channel_axis)
        mean = output.mean(axis=axes, keepdims=True)
        output = (output - mean) * rng.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast) + mean
    if saturation and channel_axis is not None and output.shape[channel_axis] >= 3:
        gray = _as_array(rgb_to_grayscale(output))
        output = gray + (output - gray) * rng.uniform(max(0.0, 1.0 - saturation), 1.0 + saturation)
    if hue and channel_axis is not None and output.shape[channel_axis] >= 3:
        output = _shift_hue_rgb(output, channel_axis, rng.uniform(-hue, hue))
    output = _restore_image_dtype(output, _as_array(image).dtype)
    return _same_kind(image, output)


def random_resized_crop(
    image: Any,
    size: int | tuple[int, int],
    scale: tuple[float, float] = (0.08, 1.0),
    ratio: tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
    interpolation: str = "bilinear",
    seed: int | None = None,
) -> np.ndarray | Tensor:
    """Crop a random area of an image and resize it."""

    if interpolation not in {"nearest", "bilinear"}:
        raise ValueError("interpolation must be 'nearest' or 'bilinear'")
    if not 0.0 < scale[0] <= scale[1]:
        raise ValueError("scale must satisfy 0 < min <= max")
    if not 0.0 < ratio[0] <= ratio[1]:
        raise ValueError("ratio must satisfy 0 < min <= max")
    rng = random.Random(seed)
    array = _as_array(image)
    height_axis, width_axis = _spatial_axes(array)
    height = array.shape[height_axis]
    width = array.shape[width_axis]
    area = height * width
    for _ in range(10):
        target_area = area * rng.uniform(scale[0], scale[1])
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if 0 < crop_w <= width and 0 < crop_h <= height:
            top = rng.randint(0, height - crop_h)
            left = rng.randint(0, width - crop_w)
            crop = _crop_array(array, top, left, crop_h, crop_w)
            resize = resize_nearest if interpolation == "nearest" else resize_bilinear
            return resize(_same_kind(image, crop), size)
    cropped = center_crop(image, min(height, width))
    resize = resize_nearest if interpolation == "nearest" else resize_bilinear
    return resize(cropped, size)


def rotate(image: Any, angle_degrees: float, fill: float = 0.0) -> np.ndarray | Tensor:
    """Rotate an image or image batch around its center using nearest sampling."""

    return affine(image, angle=angle_degrees, fill=fill)


def affine(
    image: Any,
    angle: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    shear: float = 0.0,
    fill: float = 0.0,
) -> np.ndarray | Tensor:
    """Apply an affine transform using inverse nearest-neighbor sampling."""

    if scale <= 0.0:
        raise ValueError("scale must be positive")
    return _same_kind(
        image,
        _apply_batch(
            image,
            lambda item: _affine_single(_as_array(item), angle, translate, scale, shear, fill),
        ),
    )


def cutout(
    image: Any,
    size: int | tuple[int, int],
    value: float = 0.0,
    seed: int | None = None,
) -> np.ndarray | Tensor:
    """Mask a random spatial rectangle in a single image or batch."""

    cut_h, cut_w = _pair(size, "size")
    rng = random.Random(seed)

    def apply(item: Any) -> np.ndarray:
        array = np.array(_as_array(item), copy=True)
        height_axis, width_axis = _spatial_axes(array)
        height = array.shape[height_axis]
        width = array.shape[width_axis]
        erase_h = min(cut_h, height)
        erase_w = min(cut_w, width)
        top = rng.randint(0, height - erase_h)
        left = rng.randint(0, width - erase_w)
        slices = [slice(None)] * array.ndim
        slices[height_axis] = slice(top, top + erase_h)
        slices[width_axis] = slice(left, left + erase_w)
        array[tuple(slices)] = value
        return array

    return _same_kind(image, _apply_batch(image, apply))


def mixup(
    images: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    alpha: float = 0.2,
    seed: int | None = None,
) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray, Tensor | np.ndarray, float]:
    """Create a mixup batch and return ``mixed, target_a, target_b, lambda``."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    lam = float(rng.beta(alpha, alpha))
    image_array = _as_array(images)
    target_array = _as_array(targets)
    if image_array.ndim < 1 or image_array.shape[0] != target_array.shape[0]:
        raise ValueError("images and targets must have the same batch size")
    indices = rng.permutation(image_array.shape[0])
    mixed = lam * image_array + (1.0 - lam) * image_array[indices]
    return (
        _same_kind(images, np.ascontiguousarray(mixed)),
        _same_kind(targets, np.ascontiguousarray(target_array)),
        _same_kind(targets, np.ascontiguousarray(target_array[indices])),
        lam,
    )


def cutmix(
    images: Tensor | np.ndarray,
    targets: Tensor | np.ndarray,
    alpha: float = 1.0,
    seed: int | None = None,
) -> tuple[Tensor | np.ndarray, Tensor | np.ndarray, Tensor | np.ndarray, float]:
    """Create a CutMix batch and return ``mixed, target_a, target_b, lambda``."""

    if alpha <= 0.0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    image_array = np.array(_as_array(images), copy=True)
    target_array = _as_array(targets)
    if image_array.ndim != 4 or image_array.shape[0] != target_array.shape[0]:
        raise ValueError("CutMix expects image batch shape NCHW or NHWC and matching targets")
    _, height_axis, width_axis = _batch_spatial_axes(image_array)
    height = image_array.shape[height_axis]
    width = image_array.shape[width_axis]
    lam = float(rng.beta(alpha, alpha))
    cut_ratio = math.sqrt(1.0 - lam)
    cut_h = max(1, int(height * cut_ratio))
    cut_w = max(1, int(width * cut_ratio))
    center_y = int(rng.integers(0, height))
    center_x = int(rng.integers(0, width))
    top = max(0, center_y - cut_h // 2)
    bottom = min(height, center_y + cut_h // 2)
    left = max(0, center_x - cut_w // 2)
    right = min(width, center_x + cut_w // 2)
    indices = rng.permutation(image_array.shape[0])
    slices = [slice(None)] * image_array.ndim
    slices[0] = slice(None)
    slices[height_axis] = slice(top, bottom)
    slices[width_axis] = slice(left, right)
    image_array[tuple(slices)] = image_array[indices][tuple(slices)]
    lam = 1.0 - ((bottom - top) * (right - left) / float(height * width))
    return (
        _same_kind(images, np.ascontiguousarray(image_array)),
        _same_kind(targets, np.ascontiguousarray(target_array)),
        _same_kind(targets, np.ascontiguousarray(target_array[indices])),
        lam,
    )


def pad(
    image: Any,
    padding: int | tuple[int, int] | tuple[int, int, int, int],
    value: float = 0.0,
) -> np.ndarray | Tensor:
    """Pad image spatial dimensions.

    ``padding`` accepts an int, ``(pad_h, pad_w)``, or
    ``(top, bottom, left, right)``.
    """

    top, bottom, left, right = _padding4(padding)
    array = _as_array(image)
    height_axis, width_axis = _spatial_axes(array)
    pad_width = [(0, 0)] * array.ndim
    pad_width[height_axis] = (top, bottom)
    pad_width[width_axis] = (left, right)
    padded = np.pad(array, pad_width, mode="constant", constant_values=value)
    return _same_kind(image, padded)


def rgb_to_grayscale(image: Any) -> np.ndarray | Tensor:
    """Convert RGB/RGBA image-like inputs to single-channel grayscale."""

    array = _as_array(image)
    if array.ndim == 2:
        return _same_kind(image, array)
    channel_axis = _channel_axis(array)
    if channel_axis is None:
        return _same_kind(image, array)
    if array.shape[channel_axis] == 1:
        return _same_kind(image, array)
    if array.shape[channel_axis] < 3:
        raise ValueError("RGB grayscale conversion requires at least three channels")
    moved: np.ndarray = np.moveaxis(array, channel_axis, -1).astype(np.float64)
    gray = (
        moved[..., 0] * 0.299
        + moved[..., 1] * 0.587
        + moved[..., 2] * 0.114
    )
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        gray = np.rint(gray).clip(info.min, info.max).astype(array.dtype)
    else:
        gray = gray.astype(array.dtype, copy=False)
    gray = np.expand_dims(gray, axis=-1)
    gray = np.moveaxis(gray, -1, channel_axis)
    return _same_kind(image, gray)


def grayscale_to_rgb(image: Any) -> np.ndarray | Tensor:
    """Repeat a grayscale channel into RGB."""

    array = _as_array(image)
    if array.ndim == 2:
        rgb = np.repeat(array[..., np.newaxis], 3, axis=-1)
        return _same_kind(image, rgb)
    channel_axis = _channel_axis(array)
    if channel_axis is None:
        raise ValueError("grayscale_to_rgb expects a grayscale image")
    if array.shape[channel_axis] == 3:
        return _same_kind(image, array)
    if array.shape[channel_axis] != 1:
        raise ValueError("grayscale_to_rgb expects a single-channel image")
    rgb = np.repeat(array, 3, axis=channel_axis)
    return _same_kind(image, rgb)


def _apply_batch(image: Any, fn: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    array = _as_array(image)
    if array.ndim != 4:
        return fn(array)
    items = [fn(array[index]) for index in range(array.shape[0])]
    return np.ascontiguousarray(np.stack(items, axis=0))


def _batch_spatial_axes(array: np.ndarray) -> tuple[int, int, int]:
    if array.ndim != 4:
        raise ValueError("expected a 4D image batch")
    if _is_channel_count(array.shape[1]) and not _is_channel_count(array.shape[-1]):
        return (1, 2, 3)
    return (3, 1, 2)


def _affine_single(
    array: np.ndarray,
    angle: float,
    translate: tuple[float, float],
    scale: float,
    shear: float,
    fill: float,
) -> np.ndarray:
    height_axis, width_axis = _spatial_axes(array)
    height = array.shape[height_axis]
    width = array.shape[width_axis]
    working = np.moveaxis(array, (height_axis, width_axis), (0, 1))
    output = np.full_like(working, fill)
    radians = math.radians(angle)
    shear_radians = math.radians(shear)
    cos_a = math.cos(radians)
    sin_a = math.sin(radians)
    shear_tan = math.tan(shear_radians)
    matrix = np.array(
        [
            [cos_a / scale, (-sin_a + shear_tan * cos_a) / scale],
            [sin_a / scale, (cos_a + shear_tan * sin_a) / scale],
        ],
        dtype=np.float64,
    )
    center = np.array([(height - 1) / 2.0, (width - 1) / 2.0], dtype=np.float64)
    offset = np.array([translate[1], translate[0]], dtype=np.float64)
    for y in range(height):
        for x in range(width):
            src_y, src_x = matrix @ (np.array([y, x], dtype=np.float64) - center - offset) + center
            nearest_y = int(round(float(src_y)))
            nearest_x = int(round(float(src_x)))
            if 0 <= nearest_y < height and 0 <= nearest_x < width:
                output[(y, x, *([slice(None)] * (working.ndim - 2)))] = working[
                    (nearest_y, nearest_x, *([slice(None)] * (working.ndim - 2)))
                ]
    output = np.moveaxis(output, (0, 1), (height_axis, width_axis))
    return np.ascontiguousarray(output)


def _as_float_image(array: np.ndarray) -> np.ndarray:
    result = array.astype(np.float32, copy=True)
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        result = result / float(info.max)
    return result


def _restore_image_dtype(array: np.ndarray, dtype: np.dtype[Any]) -> np.ndarray:
    clipped = np.clip(array, 0.0, 1.0)
    if np.issubdtype(dtype, np.integer):
        info: Any = np.iinfo(dtype)
        return np.rint(clipped * float(info.max)).clip(info.min, info.max).astype(dtype)
    return clipped.astype(dtype, copy=False)


def _shift_hue_rgb(array: np.ndarray, channel_axis: int, delta: float) -> np.ndarray:
    moved = np.moveaxis(array, channel_axis, -1)
    rgb = moved[..., :3]
    rotated = np.roll(rgb, int(round(delta * 6.0)), axis=-1)
    output = moved.copy()
    output[..., :3] = rotated
    return np.moveaxis(output, -1, channel_axis)


def _jitter_bound(value: float, name: str) -> float:
    if value < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return float(value)


def _degree_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        amount = float(value)
        if amount < 0.0:
            raise ValueError("degree values must be non-negative")
        return (-amount, amount)
    if len(value) != 2:
        raise ValueError("degrees must be a scalar or pair")
    first, second = float(value[0]), float(value[1])
    if first > second:
        raise ValueError("degree range must satisfy min <= max")
    return (first, second)


def _same_kind(original: Any, array: np.ndarray) -> np.ndarray | Tensor:
    contiguous = np.ascontiguousarray(array)
    if isinstance(original, Tensor):
        return from_numpy(contiguous)
    return contiguous


def _crop_array(array: np.ndarray, top: int, left: int, height: int, width: int) -> np.ndarray:
    height_axis, width_axis = _spatial_axes(array)
    slices = [slice(None)] * array.ndim
    slices[height_axis] = slice(top, top + height)
    slices[width_axis] = slice(left, left + width)
    return np.ascontiguousarray(array[tuple(slices)])


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


def _padding4(
    padding: int | tuple[int, int] | tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    if isinstance(padding, int):
        values = (padding, padding, padding, padding)
    elif len(padding) == 2:
        values = (int(padding[0]), int(padding[0]), int(padding[1]), int(padding[1]))
    elif len(padding) == 4:
        values = (int(padding[0]), int(padding[1]), int(padding[2]), int(padding[3]))
    else:
        raise ValueError("padding must be an int, pair, or four-tuple")
    if any(value < 0 for value in values):
        raise ValueError("padding values must be non-negative")
    return values


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


def _channel_axis(array: np.ndarray) -> int | None:
    if array.ndim == 2:
        return None
    if array.ndim == 3:
        if _is_channel_count(array.shape[0]) and not _is_channel_count(array.shape[-1]):
            return 0
        return 2
    if array.ndim == 4:
        if _is_channel_count(array.shape[1]) and not _is_channel_count(array.shape[-1]):
            return 1
        return 3
    raise ValueError("image must have shape HxW, HxWxC, CxHxW, NHWC, or NCHW")


def _to_channel_last(array: np.ndarray) -> np.ndarray:
    if array.ndim == 2:
        return array
    if array.ndim == 3:
        return np.moveaxis(array, 0, -1) if _channel_axis(array) == 0 else array
    if array.ndim == 4:
        return np.moveaxis(array, 1, -1) if _channel_axis(array) == 1 else array
    raise ValueError("image must have shape HxW, HxWxC, CxHxW, NHWC, or NCHW")


def _numpy_dtype(dtype: str) -> np.dtype[Any]:
    mapping = {
        "float32": np.dtype(np.float32),
        "float64": np.dtype(np.float64),
        "int32": np.dtype(np.int32),
        "int64": np.dtype(np.int64),
        "bool": np.dtype(np.bool_),
        "uint8": np.dtype(np.uint8),
    }
    if dtype not in mapping:
        raise ValueError(f"unsupported dtype: {dtype!r}")
    return mapping[dtype]


__all__ = [
    "CenterCrop",
    "Compose",
    "ColorJitter",
    "Cutout",
    "Normalize",
    "RandomCrop",
    "RandomAffine",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "RandomRotation",
    "Resize",
    "ToTensor",
    "affine",
    "batch_center_crop",
    "batch_normalize",
    "batch_resize",
    "center_crop",
    "color_jitter",
    "cutmix",
    "cutout",
    "grayscale_to_rgb",
    "horizontal_flip",
    "mixup",
    "normalize",
    "pad",
    "random_crop",
    "random_horizontal_flip",
    "random_resized_crop",
    "resize_bilinear",
    "resize_nearest",
    "rotate",
    "rgb_to_grayscale",
    "to_numpy_image",
    "to_tensor",
    "vertical_flip",
]
