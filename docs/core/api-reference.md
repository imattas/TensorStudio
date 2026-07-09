# API Overview

The public package is imported as:

```python
import tensorstudio as ts
```

The native extension module is `tensorstudio._C`. Most users should use the
Python API, which is a thin layer over the C++ tensor core plus Python-level
modules, optimizers, data utilities, and serialization helpers.

## Top-Level Exports

- `Tensor`
- `tensor`
- `from_numpy`
- `to_device`
- `zeros`
- `zeros_like`
- `ones`
- `ones_like`
- `empty`
- `empty_like`
- `full`
- `full_like`
- `rand`
- `rand_like`
- `randn`
- `randn_like`
- `arange`
- `eye`
- `linspace`
- `normalize_dtype`
- `dtype_of`
- `can_cast`
- `cast`
- `promote_types`
- `result_type`
- `sin`
- `cos`
- `tan`
- `asin`
- `acos`
- `atan`
- `log1p`
- `rsqrt`
- `equal`
- `not_equal`
- `less`
- `less_equal`
- `greater`
- `greater_equal`
- `maximum`
- `minimum`
- `where`
- `conv2d`
- `max_pool2d`
- `avg_pool2d`
- `astype`
- `concat`
- `stack`
- `manual_seed`
- `save`
- `load`
- `save_npz`
- `load_npz`
- `inspect_npz`
- `check_npz_compatibility`
- `export_onnx`
- `inspect_onnx`
- `onnx_runtime_info`
- `check_onnx_runtime_compatibility`
- `run_onnx_inference`
- `inspect_model_format`
- `inspect_keras`
- `inspect_saved_model`
- `inspect_hdf5`
- `inspect_tflite`
- `no_grad`
- `is_grad_enabled`
- `set_grad_enabled`
- `checkpoint`
- `current_device`
- `set_default_device`
- `reset_default_device`
- `device_scope`
- `storage_telemetry`
- `reset_storage_telemetry`
- `interchange`
- `vision`
- `dtypes`
- `math`
- `project`
- `data`
- `nn`
- `optim`
- `__version__`

## Tensor Creation

```python
ts.tensor(data, dtype=None, requires_grad=False, device=None)
ts.from_numpy(array, requires_grad=False, device=None)
ts.zeros(shape, dtype="float32", requires_grad=False, device=None)
ts.zeros_like(input, dtype=None, requires_grad=None, device=None)
ts.ones(shape, dtype="float32", requires_grad=False, device=None)
ts.ones_like(input, dtype=None, requires_grad=None, device=None)
ts.empty(shape, dtype="float32", requires_grad=False, device=None)
ts.empty_like(input, dtype=None, requires_grad=None, device=None)
ts.full(shape, fill_value, dtype="float32", requires_grad=False, device=None)
ts.full_like(input, fill_value, dtype=None, requires_grad=None, device=None)
ts.rand(shape, dtype="float32", seed=None, requires_grad=False, device=None)
ts.rand_like(input, dtype=None, seed=None, requires_grad=None, device=None)
ts.randn(shape, dtype="float32", seed=None, requires_grad=False, device=None)
ts.randn_like(input, dtype=None, seed=None, requires_grad=None, device=None)
ts.arange(start, stop=None, step=1, dtype="float32", requires_grad=False, device=None)
ts.eye(n, m=None, dtype="float32", requires_grad=False, device=None)
ts.linspace(start, stop, steps, dtype="float32", requires_grad=False, device=None)
```

Nested Python data must be rectangular. Ragged lists raise `ShapeError`.
`requires_grad=True` is accepted for floating point tensors and rejected for
integer and boolean tensors.
When `device=None`, creation uses `current_device()`. `*_like` helpers default
to the input tensor's device unless `device=` is supplied.

## DType Helpers

```python
ts.normalize_dtype("double")
ts.dtype_of(ts.tensor([1, 2, 3]))
ts.can_cast("int32", "float64", casting="safe")
ts.cast(ts.tensor([1, 2, 3], dtype="int32"), "int64", casting="safe")
ts.promote_types("int32", "float32")
ts.result_type("int64", "int32", op="div")
```

`promote_types` returns the arithmetic result dtype for `add`, `sub`, `mul`,
and `matmul`. `result_type` also accounts for operation-specific rules:
division returns `float32` or `float64`, and comparisons return `bool`.
`can_cast` and `cast` support `no`, `equiv`, `safe`, `same_kind`, and
`unsafe` casting policies.

## Tensor Properties

- `shape`
- `strides`
- `dtype`
- `device`
- `ndim`
- `size`
- `requires_grad`
- `grad`
- `is_contiguous`
- `T`

## Tensor Methods

Conversion:

- `numpy()`
- `tolist()`
- `item()`
- `to_device(device="cpu", copy=False)`

Autograd and copying:

- `backward(gradient=None)`
- `zero_grad()`
- `clone()`
- `detach()`

Top-level autograd helpers:

- `checkpoint(function, *tensor_args)`

`checkpoint` supports eager Tensor-only subgraphs. It runs the function without
recording intermediates in forward, then recomputes during backward and returns
gradients for the original Tensor arguments.

Views:

- `reshape(*shape)` or `reshape(shape)`
- `flatten()`
- `transpose()`
- `__getitem__(key)` through `x[key]`

Indexing keys may contain integers, slices, tuples, ellipsis, and `None` for
new axes. Integer indexing drops the indexed dimension. Slice indexing returns
strided views, including negative-step views. A single Python integer-list axis
is supported as materialized gather indexing, including negative, repeated, and
empty lists. A single Python boolean-list axis is supported as materialized
mask indexing when the mask length matches the consumed axis. A single
TensorStudio `int32`/`int64` tensor index is supported with rank one or higher
and materialized as a copy whose advanced dimensions match the selector shape.
A single 1D TensorStudio `bool` tensor mask is supported as an axis mask, and a
full-shape TensorStudio `bool` mask is supported as flattened boolean
selection. Prefix-shape TensorStudio `bool` masks are supported when the mask
shape matches leading tensor dimensions, preserving trailing dimensions in the
output. Multi-dimensional TensorStudio `bool` masks are also supported inside a
tuple, such as `x[:, mask]`, when the mask shape matches the consecutive tensor
axes at that tuple position. Adjacent multi-axis integer-list or integer-tensor
selectors are supported as paired vectorized indexing, including broadcasted
higher-rank integer tensor selector shapes. Adjacent and non-adjacent
one-dimensional boolean masks can also participate in multi-axis vectorized
indexing, including mixed boolean/integer selectors. Non-adjacent advanced
axes place their broadcasted selector shape before remaining basic output
dimensions. Tuple-position partial boolean masks expand to coordinate gathers
and can mix with integer and one-dimensional boolean advanced selectors,
including gradient scatter-back for differentiable source tensors.

Math:

- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `max(axis=None, keepdims=False)`
- `min(axis=None, keepdims=False)`
- `argmax(axis=None, keepdims=False)`
- `argmin(axis=None, keepdims=False)`

For reductions, `axis` may be `None`, an int, or a tuple/list of ints.
Negative axes are normalized against the input rank, duplicate axes are
rejected, and `axis=()` is a no-op identity.

For arg reductions, `axis` may be `None` or an int. All-element arg reductions
return the flat index of the first selected value. Axis arg reductions return
`int64` indices along that axis and are not differentiable.

Comparisons and selection:

- `equal(other)`
- `not_equal(other)`
- `less(other)`
- `less_equal(other)`
- `greater(other)`
- `greater_equal(other)`
- `maximum(other)`
- `minimum(other)`
- `where(true_value, false_value)`

`maximum` and `minimum` broadcast like other binary elementwise operations.
`where` treats the tensor it is called on as a boolean condition tensor and
routes values from the two branch tensors. Gradients flow through floating point
branches, not through the boolean condition.

- `relu()`
- `sigmoid()`
- `tanh()`
- `exp()`
- `log()`
- `log1p()`
- `sqrt()`
- `rsqrt()`
- `sin()`
- `cos()`
- `tan()`
- `asin()`
- `acos()`
- `atan()`
- `abs()`
- `clamp(min_value, max_value)`
- `clip(min_value, max_value)`
- `astype(dtype)`
- `to(dtype)`

## Operators

- `x + y`
- `x - y`
- `x * y`
- `x / y`
- `-x`
- `x ** exponent`
- `x @ y`
- `x == y`
- `x != y`
- `x < y`
- `x <= y`
- `x > y`
- `x >= y`
- `x[key]`

`+=` is intentionally unsupported and raises a clear error.

Device movement:

```python
ts.to_device(input, device="cpu", copy=False)
input.to_device("cpu", copy=True)
```

The current executable backend is CPU-only. CPU-to-CPU transfer can either
return a storage-sharing alias (`copy=False`) or a clone (`copy=True`).
Non-CPU targets raise `DeviceError` until backend storage and copy kernels are
implemented.

## Functional Ops

```python
ts.sin(input)
ts.cos(input)
ts.tan(input)
ts.asin(input)
ts.acos(input)
ts.atan(input)
ts.log1p(input)
ts.rsqrt(input)
ts.equal(left, right)
ts.not_equal(left, right)
ts.less(left, right)
ts.less_equal(left, right)
ts.greater(left, right)
ts.greater_equal(left, right)
ts.maximum(left, right)
ts.minimum(left, right)
ts.where(condition, true_value, false_value)
ts.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1)
ts.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1)
ts.avg_pool2d(input, kernel_size, stride=None, padding=0, count_include_pad=False)
```

Comparison functions always return `bool` tensors. `where` requires a `bool`
condition and broadcasts condition, true branch, and false branch together.
The branch dtype is promoted with the same compact promotion order used by
arithmetic ops.

`conv2d` expects NCHW input with shape `(batch, in_channels, height, width)`
and weight shape `(out_channels, in_channels, kernel_h, kernel_w)`. Stride,
padding, and dilation accept either an integer or a pair of integers. The
current implementation is CPU-only and supports autograd for input, weight, and
bias.

`max_pool2d` and `avg_pool2d` also expect NCHW input. `kernel_size`, `stride`,
`padding`, and `dilation` accept an integer or a pair where supported. If
`stride=None`, pooling defaults to non-overlapping windows with
`stride=kernel_size`. `max_pool2d` supports dilation; `avg_pool2d` supports
`count_include_pad`.

```python
ts.astype(input, dtype, casting="unsafe")
ts.concat(tensors, axis=0)
ts.stack(tensors, axis=0)
```

`concat` joins same-dtype tensors along an existing axis. `stack` inserts a new
axis and requires identical input shapes.

## `tensorstudio.nn`

- `Module`
- `Parameter`
- `Linear`
- `Conv2d`
- `MaxPool2d`
- `AvgPool2d`
- `Sequential`
- `ReLU`
- `Sigmoid`
- `Tanh`
- `Dropout`
- `Flatten`
- `Identity`
- `LeakyReLU`
- `Softplus`
- `MSELoss`
- `L1Loss`
- `BCELoss`
- `BCEWithLogitsLoss`
- `CrossEntropyLoss`
- `HuberLoss`
- Functional equivalents for linear layers, convolution, pooling, activations,
  softmax/log-softmax, and losses.

Module methods:

- `parameters()`
- `named_parameters()`
- `trainable_parameters()`
- `children()`
- `named_children()`
- `modules()`
- `named_modules()`
- `train()`
- `eval()`
- `zero_grad()`
- `requires_grad_()`
- `freeze()`
- `unfreeze()`
- `parameter_count()`
- `apply()`
- `state_dict()`
- `load_state_dict()`

## `tensorstudio.optim`

- `SGD(params, lr=0.01, momentum=0.0, weight_decay=0.0)`
- `Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`
- `AdamW(params, lr=0.001, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)`
- `clip_grad_norm_(params, max_norm)`
- `clip_grad_value_(params, clip_value)`
- `StepLR(optimizer, step_size, gamma=0.1)`
- `ExponentialLR(optimizer, gamma)`

Optimizers implement `zero_grad`, `step`, `state_dict`, and `load_state_dict`.

## `tensorstudio.data`

- `Dataset`
- `TensorDataset`
- `DataLoader`

`DataLoader` supports batching, shuffle, `drop_last`, and deterministic seed.

## `tensorstudio.math`

- `square(input)`
- `reciprocal(input)`
- `variance(input, axis=None, keepdims=False, correction=0)`
- `std(input, axis=None, keepdims=False, correction=0)`
- `norm(input, ord=2, axis=None, keepdims=False)`

These helpers accept the same reduction axis forms as native reductions:
`None`, an int, or a tuple/list of ints.

These helpers compose TensorStudio tensors and keep autograd support.

## `tensorstudio.project`

- `ProjectConfig(name, version="0.1.0", seed=None, metadata=None)`
- `Project(root, config=None, create=True)`
- `Trainer(model, optimizer, loss_fn, metric_fn=None)`
- `History`
- `save_state_dict(model, path)`
- `load_state_dict(model, path, strict=True)`
- `save_checkpoint(model, path, optimizer=None, metadata=None)`
- `load_checkpoint(model, path, optimizer=None, strict=True)`

State-dict helpers use non-pickle NPZ files. Full checkpoints use trusted
pickle for optimizer state and metadata.

## Error Types

Import native exception aliases from `tensorstudio.errors`:

- `TensorStudioError`
- `ShapeError`
- `DTypeError`
- `DeviceError`
- `AutogradError`

## Serialization

```python
ts.save(obj, "object.tsmodel")
loaded = ts.load("object.tsmodel")
ts.save_npz(model.state_dict(), "weights.tsnpz")
info = ts.inspect_npz("weights.tsnpz")
compat = ts.check_npz_compatibility("weights.tsnpz")
state = ts.load_npz("weights.tsnpz")
```

`save`/`load` use pickle. Only load pickle files from trusted sources.
`save_npz`/`load_npz` store tensors and flat state dictionaries without pickle.
`inspect_npz` and `check_npz_compatibility` read archive metadata with
`allow_pickle=False` before constructing tensors.

## `tensorstudio.interchange`

- `export_onnx(model, path, input_shape, input_name="input", output_name="output")`
- `inspect_onnx(path, check=True)`
- `onnx_runtime_info(providers=None)`
- `check_onnx_runtime_compatibility(path, providers=None)`
- `run_onnx_inference(path, inputs, providers=None, output_names=None, as_tensor=True)`
- `inspect_model_format(path)`
- `inspect_keras(path)`
- `inspect_saved_model(path)`
- `inspect_hdf5(path)`
- `inspect_tflite(path)`

ONNX export supports `Linear`, `Conv2d`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`,
`MaxPool2d`, and `AvgPool2d` module stacks.
ONNX metadata inspection reports graph IO, opsets, operators, initializers, and
checker status without executing the model. ONNX Runtime diagnostics are
optional and report provider availability/session compatibility without running
inference. `run_onnx_inference` delegates compatible static graphs to ONNX
Runtime when the optional extra is installed.
Model-format inspection safely reports metadata for ONNX files, Keras `.keras`
archives, TensorFlow SavedModel directories, HDF5/Keras weight files, and
TFLite flatbuffer identifiers without loading model code.

## `tensorstudio.hardware` And `tensorstudio.kernels`

- `device("cpu")`
- `current_device()`
- `set_default_device(device="cpu")`
- `reset_default_device()`
- `device_scope(device)`
- `is_available(device="cpu")`
- `available_devices()`
- `backend_info()`
- `backend_allocator_info(device=None)`
- `backend_runtime_info(device=None)`
- `backend_device_properties(device=None)`
- `logical_device_info(device=None)`
- `backend_op_info(op=None)`
- `backend_kernel_info(device=None)`
- `backend_supports_kernel(op, device="cpu", dtype="float32")`
- `kernel_placement_info(op, device="cpu", dtype="float32")`
- `backend_execution_plan(op, device="cpu", dtype="float32", input_devices=None)`
- `device_transfer_info(source, target)`
- `can_transfer(source, target)`
- `storage_telemetry()`
- `reset_storage_telemetry()`
- `register_backend(name, compiled=True, available=False, device_count=0, reason="")`
- `register_backend_kernel(backend, op, dtypes, forward=True, backward=False, available=True, reason="", memory_space="device", execution_mode="sync", deterministic=True, priority=50)`
- `unregister_backend(name)`
- `load_kernel_manifest(path)`
- `validate_kernel_manifest(path)`
- `discover_kernel_manifests(root)`
- `register_kernel_manifest(manifest)`

External backend and kernel registration is metadata-only. TensorStudio does
not load native plugins or allocate non-CPU tensors in this release.

The hardware registry separates operation interfaces from device kernels:
`backend_op_info()` reports op arity, category, differentiability, and
shape-inference metadata, while `backend_kernel_info()` reports per-backend
dtype, memory-space, execution-mode, determinism, and priority metadata.
`backend_runtime_info()` reports whether a device is executable, whether it
supports eager execution, graph/compiler lowering, streams/events, peer access,
and host fallback. `backend_device_properties()` reports physical device
metadata and `logical_device_info()` reports currently exposed logical devices.
`kernel_placement_info()` reports whether a requested device can run an op
directly and whether a CPU fallback path exists. `backend_execution_plan()`
combines placement, runtime, kernel, and transfer diagnostics for an eager op.
`storage_telemetry()` reports native CPU storage allocation counters for local
profiling.

## `tensorstudio.vision`

- `load_image(path, mode="RGB")`
- `save_image(image, path)`
- `make_grid(images, nrow=8, padding=2, value=0.0)`
- `to_tensor(image, dtype="float32", scale=True, channels_first=None)`
- `to_numpy_image(image, channels_last=True, dtype=None, unscale=False)`
- `normalize(input, mean, std)`
- `center_crop(image, size)`
- `resize_nearest(image, size)`
- `resize_bilinear(image, size)`
- `random_crop(image, size, padding=0, seed=None)`
- `horizontal_flip(image)`
- `vertical_flip(image)`
- `random_horizontal_flip(image, p=0.5, seed=None)`
- `pad(image, padding, value=0.0)`
- `rgb_to_grayscale(image)`
- `grayscale_to_rgb(image)`
- `Compose`, `ToTensor`, `Normalize`, `Resize`, `CenterCrop`,
  `RandomCrop`, `RandomHorizontalFlip`
- `ImageFolder(root, transform=None, target_transform=None)`
- `ImageList(samples, transform=None, target_transform=None)`
- `ImageManifestDataset(manifest, root=None, transform=None, target_transform=None)`
- `build_image_manifest(root, output_path=None, checksum=True)`
- `load_image_manifest(path)`
- `save_image_manifest(manifest, path)`
- `validate_image_manifest(manifest, root=None, checksum=None)`
- `accuracy(logits, target)`
- `top_k_accuracy(logits, target, k=5)`
- `confusion_matrix(prediction, target, num_classes=None, normalize=False)`
- `box_iou(boxes1, boxes2)`
- `draw_bounding_boxes(image, boxes, labels=None, color=(255, 0, 0), width=1)`
- `ConvBlock(in_channels, out_channels, kernel_size=3, padding=1, pool=True)`
- `TinyConvClassifier(input_shape, num_classes, hidden_channels=8)`
- `ImageClassifier(input_shape, num_classes, channels=(8, 16))`
- `make_cnn_classifier(input_shape, num_classes, hidden_channels=8)`
- `make_image_classifier(input_shape, num_classes, channels=(8, 16))`
