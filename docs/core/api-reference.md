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
- `from_dlpack`
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
- `uniform`
- `uniform_like`
- `normal`
- `normal_like`
- `randint`
- `randint_like`
- `bernoulli`
- `bernoulli_like`
- `arange`
- `eye`
- `linspace`
- `normalize_dtype`
- `dtype_of`
- `promote_types`
- `result_type`
- `sin`
- `cos`
- `tan`
- `asin`
- `acos`
- `atan`
- `logsumexp`
- `softmax`
- `log_softmax`
- `log1p`
- `rsqrt`
- `bmm`
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
- `conv_transpose2d`
- `embedding`
- `max_pool2d`
- `avg_pool2d`
- `all`
- `any`
- `var`
- `variance`
- `std`
- `norm`
- `einsum`
- `astype`
- `concat`
- `stack`
- `manual_seed`
- `TensorSpec`
- `trace`
- `compile_graph`
- `save_graph`
- `load_graph`
- `graph`
- `SparseCOOTensor`
- `SparseCSRTensor`
- `csr_from_dense`
- `csr_from_coo`
- `sparse_coo_tensor`
- `sparse_csr_tensor`
- `sparse_csr_mm`
- `sparse_from_dense`
- `sparse_mm`
- `create_model`
- `list_models`
- `model_info`
- `register_model`
- `register_kernel`
- `unregister_kernel`
- `call_kernel`
- `list_kernels`
- `get_kernel`
- `kernel_info`
- `run_onnx`
- `onnxruntime_is_available`
- `onnxruntime_available_providers`
- `check_onnxruntime_compatibility`
- `save`
- `load`
- `save_npz`
- `load_npz`
- `export_onnx`
- `no_grad`
- `is_grad_enabled`
- `set_grad_enabled`
- `interchange`
- `vision`
- `dtypes`
- `math`
- `project`
- `distributed`
- `kernels`
- `model_zoo`
- `quantization`
- `sparse`
- `data`
- `nn`
- `optim`
- `__version__`

## Tensor Creation

```python
ts.tensor(data, dtype=None, requires_grad=False)
ts.from_numpy(array, requires_grad=False)
ts.zeros(shape, dtype="float32", requires_grad=False)
ts.zeros_like(input, dtype=None, requires_grad=None)
ts.ones(shape, dtype="float32", requires_grad=False)
ts.ones_like(input, dtype=None, requires_grad=None)
ts.empty(shape, dtype="float32", requires_grad=False)
ts.empty_like(input, dtype=None, requires_grad=None)
ts.full(shape, fill_value, dtype="float32", requires_grad=False)
ts.full_like(input, fill_value, dtype=None, requires_grad=None)
ts.rand(shape, dtype="float32", seed=None, requires_grad=False)
ts.rand_like(input, dtype=None, seed=None, requires_grad=None)
ts.randn(shape, dtype="float32", seed=None, requires_grad=False)
ts.randn_like(input, dtype=None, seed=None, requires_grad=None)
ts.uniform(shape, low=0.0, high=1.0, dtype="float32", seed=None, requires_grad=False)
ts.uniform_like(input, low=0.0, high=1.0, dtype=None, seed=None, requires_grad=None)
ts.normal(shape, mean=0.0, stddev=1.0, dtype="float32", seed=None, requires_grad=False)
ts.normal_like(input, mean=0.0, stddev=1.0, dtype=None, seed=None, requires_grad=None)
ts.randint(shape, low, high, dtype="int64", seed=None)
ts.randint_like(input, low, high, dtype=None, seed=None)
ts.bernoulli(shape, probability=0.5, dtype="bool", seed=None)
ts.bernoulli_like(input, probability=0.5, dtype="bool", seed=None)
ts.arange(start, stop=None, step=1, dtype="float32", requires_grad=False)
ts.eye(n, m=None, dtype="float32", requires_grad=False)
ts.linspace(start, stop, steps, dtype="float32", requires_grad=False)
```

Nested Python data must be rectangular. Ragged lists raise `ShapeError`.
`requires_grad=True` is accepted for floating point tensors and rejected for
integer and boolean tensors.

## DType Helpers

```python
ts.normalize_dtype("double")
ts.dtype_of(ts.tensor([1, 2, 3]))
ts.promote_types("int32", "float32")
ts.result_type("int64", "int32", op="div")
```

`promote_types` returns the arithmetic result dtype for `add`, `sub`, `mul`,
and `matmul`. `result_type` also accounts for operation-specific rules:
division returns `float32` or `float64`, and comparisons return `bool`.

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
- `is_leaf`
- `T`

## Tensor Methods

Conversion:

- `numpy()`
- `tolist()`
- `item()`
- `to("float64")` for dtype casts
- `to("cpu")`, `to_device("cpu")`, and `cpu()` for device transfers

Autograd and copying:

- `backward(gradient=None, retain_graph=False)`
- `zero_grad()`
- `clone()`
- `detach()`
- `detach_()`
- `clear_history()`
- `zero_()`
- `fill_(value)`
- `add_(other, alpha=1.0)`

Views:

- `reshape(*shape)` or `reshape(shape)`
- `flatten()`
- `transpose()` to reverse axes
- `transpose(axis0, axis1)` to swap two axes
- `permute(*axes)` or `permute(axes)`
- `squeeze(axis=None)`
- `unsqueeze(axis)`
- `__getitem__(key)` through `x[key]`

Indexing keys may contain integers, slices, tuples, ellipsis, and `None` for
new axes. Integer indexing drops the indexed dimension. Slice indexing returns
strided views, including negative-step views. Advanced list, tensor, and
boolean-mask indexing are rejected.

Math:

- `sum(axis=None, keepdims=False)`
- `mean(axis=None, keepdims=False)`
- `var(axis=None, keepdims=False, correction=0)`
- `variance(axis=None, keepdims=False, correction=0)`
- `std(axis=None, keepdims=False, correction=0)`
- `norm(ord=2, axis=None, keepdims=False)`
- `max(axis=None, keepdims=False)`
- `min(axis=None, keepdims=False)`
- `argmax(axis=None, keepdims=False)`
- `argmin(axis=None, keepdims=False)`
- `all(axis=None, keepdims=False)`
- `any(axis=None, keepdims=False)`

For reductions, `axis` may be `None`, an int, or a tuple/list of ints.
Negative axes are normalized against the input rank, duplicate axes are
rejected, and `axis=()` is a no-op identity.

For arg reductions, `axis` may be `None` or an int. All-element arg reductions
return the flat index of the first selected value. Axis arg reductions return
`int64` indices along that axis and are not differentiable.

Tensor `var` and `std` support all-elements and single-axis reductions.
Tuple-axis variance and standard deviation are available through
`tensorstudio.math.variance` and `tensorstudio.math.std`.

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
- `logsumexp(axis=None, keepdims=False)`
- `softmax(axis=-1)`
- `log_softmax(axis=-1)`
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

## Functional Ops

```python
ts.sin(input)
ts.cos(input)
ts.tan(input)
ts.asin(input)
ts.acos(input)
ts.atan(input)
ts.log1p(input)
ts.logsumexp(input, axis=None, keepdims=False)
ts.softmax(input, axis=-1)
ts.log_softmax(input, axis=-1)
ts.rsqrt(input)
ts.bmm(left, right)
ts.equal(left, right)
ts.not_equal(left, right)
ts.less(left, right)
ts.less_equal(left, right)
ts.greater(left, right)
ts.greater_equal(left, right)
ts.maximum(left, right)
ts.minimum(left, right)
ts.where(condition, true_value, false_value)
ts.conv2d(input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
ts.conv_transpose2d(
    input,
    weight,
    bias=None,
    stride=1,
    padding=0,
    output_padding=0,
    dilation=1,
    groups=1,
)
ts.embedding(indices, weight)
ts.max_pool2d(input, kernel_size, stride=None, padding=0, dilation=1)
ts.avg_pool2d(input, kernel_size, stride=None, padding=0, count_include_pad=False)
ts.all(input, axis=None, keepdims=False)
ts.any(input, axis=None, keepdims=False)
ts.var(input, axis=None, keepdims=False, correction=0)
ts.variance(input, axis=None, keepdims=False, correction=0)
ts.std(input, axis=None, keepdims=False, correction=0)
ts.norm(input, ord=2, axis=None, keepdims=False)
ts.einsum(pattern, *operands)
```

Comparison functions always return `bool` tensors. `where` requires a `bool`
condition and broadcasts condition, true branch, and false branch together.
The branch dtype is promoted with the same compact promotion order used by
arithmetic ops.

`conv2d` expects NCHW input with shape `(batch, in_channels, height, width)`
and weight shape `(out_channels, in_channels / groups, kernel_h, kernel_w)`.
Stride, padding, and dilation accept either an integer or a pair of integers.
The current implementation is CPU-only and supports autograd for input, weight,
and bias.

`conv_transpose2d` expects NCHW input and weight shape
`(in_channels, out_channels / groups, kernel_h, kernel_w)`. The optional bias
shape is `(out_channels,)`.

`embedding` expects integer indices and a 2D weight table shaped
`(num_embeddings, embedding_dim)`. Gradients accumulate into repeated weight
rows; indices are not differentiable.

`max_pool2d` and `avg_pool2d` also expect NCHW input. `kernel_size`, `stride`,
`padding`, and `dilation` accept an integer or a pair where supported. If
`stride=None`, pooling defaults to non-overlapping windows with
`stride=kernel_size`. `max_pool2d` supports dilation; `avg_pool2d` supports
`count_include_pad`.

`bmm` expects two 3D tensors with shapes `(batch, rows, inner)` and
`(batch, inner, columns)`. The `@` operator dispatches to the same batched
kernel when both operands are 3D.

`softmax`, `log_softmax`, and `logsumexp` use max-shifted stable numerics.
`einsum` supports a documented practical subset of common matrix, batched
matrix, dot-product, transpose, and sum patterns.

```python
ts.astype(input, dtype)
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
- `from_csv(path, target_column=None, feature_columns=None, dtype="float32", target_dtype=None)`
- `from_jsonl(path, feature_key="features", target_key="label", dtype="float32", target_dtype=None)`
- `from_text_lines(path, strip=True, skip_empty=True)`
- `from_libsvm(path, num_features=None, dtype="float32", target_dtype="float32")`

`DataLoader` supports batching, shuffle, `drop_last`, and deterministic seed.
The public-format readers create TensorStudio datasets from common local files
without running user code from those files.

## `tensorstudio.math`

- `square(input)`
- `reciprocal(input)`
- `variance(input, axis=None, keepdims=False, correction=0)`
- `std(input, axis=None, keepdims=False, correction=0)`
- `norm(input, ord=2, axis=None, keepdims=False)`
- `logsumexp(input, axis=None, keepdims=False)`
- `softmax(input, axis=-1)`
- `log_softmax(input, axis=-1)`
- `all(input, axis=None, keepdims=False)`
- `any(input, axis=None, keepdims=False)`
- `einsum(pattern, *operands)`

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

## `tensorstudio.hardware`

- `Device("cpu")`
- `device("cpu")`
- `available_devices()`
- `backend_info()`
- `is_available(device)`
- `device_count("cpu")`
- `cuda_is_available()`
- `metal_is_available()`

The default wheels report CPU available and CUDA/Metal unavailable. Tensor
factories accept `device="cpu"` and reject unavailable accelerator targets with
`DeviceError`.

## `tensorstudio.graph`

- `TensorSpec(shape, dtype="float32", name=None)`
- `trace(function, input_specs)`
- `compile_graph(function_or_graph, input_specs=None, optimize=True)`
- `save_graph(graph, path)`
- `load_graph(path)`
- `Graph`
- `GraphNode`
- `GraphTensor`
- `ExecutableGraph`

`trace()` executes a Python callable once with symbolic `GraphTensor`
placeholders. The function must return a `GraphTensor` or a sequence of
`GraphTensor` outputs. Supported symbolic operations include binary arithmetic,
rank-2 matrix multiplication, unary activations, `sum`, `mean`, `reshape`,
`flatten`, and rank-2 transpose.

`Graph.save()` and `save_graph()` write an inspectable JSON graph payload.
`Graph.optimize()` currently performs constant folding and scalar
multiply-add fusion. `Graph.compile()` and `compile_graph()` return an
`ExecutableGraph` that runs the supported graph through TensorStudio eager
tensor operations.

```python
graph = ts.trace(lambda x: (x * 2.0 + 1.0).relu(), [ts.TensorSpec((4,))])
compiled = graph.compile()
print(compiled(ts.ones((4,))).tolist())
print(compiled.profile(ts.ones((4,))))
print(compiled.memory_plan())
```

The graph runtime is intentionally constrained in `1.15.0`: it does not capture
arbitrary Python control flow and does not generate machine code.

## `tensorstudio.sparse`

- `SparseCOOTensor(indices, values, shape, coalesced=False)`
- `SparseCSRTensor(crow_indices, col_indices, values, shape)`
- `sparse_coo_tensor(indices, values, shape, dtype=None, coalesced=False)`
- `sparse_csr_tensor(crow_indices, col_indices, values, shape, dtype=None)`
- `sparse_from_dense(input)`
- `csr_from_dense(input)`
- `csr_from_coo(input)`
- `sparse_mm(sparse, dense)`
- `sparse_csr_mm(sparse, dense)`

The sparse API is experimental and stores COO or CSR indices plus dense
TensorStudio values. It supports dense conversion, duplicate coalescing for
COO, COO-to-CSR conversion, rank-2 transpose for COO, and sparse-dense matrix
multiplication. Sparse layout mutation is not a full differentiable sparse
system.

## `tensorstudio.model_zoo`

- `ModelCard(name, task, description, default_kwargs, tags=())`
- `register_model(card, factory, overwrite=False)`
- `list_models(task=None, tag=None)`
- `model_info(name)`
- `create_model(name, **overrides)`

Built-in factories include `tiny_mlp`, `tiny_cnn`, and
`tiny_bigram_language_model`. They are deterministic small examples intended
for tests, examples, and docs rather than pretrained production models.

## `tensorstudio.nn` Language Helpers

- `Vocabulary`
- `TokenEmbedding`
- `PositionalEmbedding`
- `CausalLanguageModel`
- `MultiHeadSelfAttention`
- `TransformerEncoderBlock`
- `make_causal_lm_batch(tokens, sequence_length)`
- `causal_language_model_loss(logits, targets)`
- `scaled_dot_product_attention(query, key, value, mask=None, causal=False)`

These helpers cover tiny tokenized examples and causal next-token losses. They
include compact attention blocks, but do not implement pretrained tokenizer or
large language-model runtimes.

## `tensorstudio.quantization`

- `QuantizationConfig(num_bits=8, symmetric=False, dtype="int32")`
- `CalibrationStats`
- `QuantizedTensor`
- `calibrate_tensor(input)`
- `calibrate_state_dict(state)`
- `merge_calibration_stats(stats)`
- `quantization_error(input, config=None)`
- `quantize_tensor(input, config=None)`
- `dequantize_tensor(quantized)`
- `fake_quantize(input, config=None)`
- `quantize_state_dict(state, config=None)`
- `dequantize_state_dict(state)`
- `quantization_report(state)`

Quantization utilities are research helpers for affine per-tensor quantization
and reporting. They do not replace hardware-specific INT8 kernels.

## `tensorstudio.kernels`

- `register_kernel(name, callable, backend="python", description="", overwrite=False)`
- `unregister_kernel(name)`
- `call_kernel(name, *args, **kwargs)`
- `list_kernels()`
- `get_kernel(name)`
- `kernel_info(name)`

The registry is a controlled extension point for Python callables or native
extension callables. It does not load arbitrary shared libraries by itself.

## `tensorstudio.distributed`

- `DistributedConfig(world_size=1, rank=0, local_rank=0, backend="single")`
- `config_from_env()`
- `distributed_info(config=None)`
- `all_reduce_sum(tensor, config=None)`
- `average_gradients(parameters, config=None)`
- `data_parallel_plan(dataset_size, batch_size, config=None)`

Only explicit single-process collectives execute in-process. Multi-process
collectives raise `NotImplementedError` until TensorStudio has a tested
transport backend.

## Error Types

Import native exception aliases from `tensorstudio.errors`:

- `TensorStudioError`
- `ShapeError`
- `DTypeError`
- `DeviceError`
- `AutogradError`

Shape, dtype, and indexing errors are intended to be actionable. Common
messages include tensor shapes, ranks, mismatched broadcast axes, requested
reshape element counts, unsupported dtype names, and unsupported Python index
types where that context is available from the C++/binding layer.

## Serialization

```python
ts.save(obj, "object.tsmodel")
loaded = ts.load("object.tsmodel")
ts.save_npz(model.state_dict(), "weights.tsnpz")
state = ts.load_npz("weights.tsnpz")
metadata = ts.load_npz_metadata("weights.tsnpz")
```

`save`/`load` use pickle. Only load pickle files from trusted sources.
`save_npz`/`load_npz` store tensors and flat state dictionaries without pickle.
`inspect_model_metadata` reads metadata from TensorStudio NPZ files,
SafeTensors files, supported ONNX files, and trusted TensorStudio checkpoints.

## `tensorstudio.interchange`

- `export_onnx(model, path, input_shape, input_name="input", output_name="output")`
- `inspect_onnx(path)`
- `import_onnx(path)`
- `run_onnx(path, input, prefer_onnxruntime=True, providers=None)`
- `onnxruntime_is_available()`
- `onnxruntime_available_providers()`
- `check_onnxruntime_compatibility(path, providers=None)`
- `export_model_card_metadata(metadata, path)`
- `save_safetensors(tensors, path, metadata=None)`
- `load_safetensors(path)`
- `inspect_model_metadata(path, trusted_pickle=False)`

ONNX export supports `Linear`, grouped/depthwise `Conv2d`,
`ConvTranspose2d`, `Flatten`, `ReLU`, `Sigmoid`, `Tanh`, `MaxPool2d`, and
`AvgPool2d` module stacks. ONNX import supports a constrained static subset,
not arbitrary ONNX models. `run_onnx()` can delegate to the external
`onnxruntime` package when installed through `tensorstudio[onnxruntime]`; it is
not a native TensorStudio full-runtime implementation.

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
