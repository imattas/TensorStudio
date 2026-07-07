# Autograd

TensorStudio includes a small reverse-mode automatic differentiation engine.
It is designed to be understandable and useful for small eager computations.

## Basic Example

```python
import tensorstudio as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(x.grad.tolist())  # [2.0, 4.0, 6.0]
```

## What Gets Tracked

Autograd tracks operations when at least one parent tensor has
`requires_grad=True`. Only floating point tensors can require gradients.

Each differentiable output stores:

- parent tensors
- a backward function
- whether the output requires gradients
- optional accumulated gradient

## Backward Pass

Calling `backward()`:

1. Validates the output tensor.
2. Uses an all-ones gradient for scalar outputs when no gradient is supplied.
3. Builds a topological ordering of the computation graph.
4. Clears stale intermediate gradients from previous retained-graph passes.
5. Walks the graph in reverse.
6. Calls each operation's backward function.
7. Accumulates gradients on leaf tensors that require them.
8. Frees non-leaf graph history unless `retain_graph=True` is supplied.

Normal backward releases graph history so a second backward through the same
non-leaf output raises a clear error. Use `retain_graph=True` on the first
backward call only when the same graph must be reused.

## Supported Gradient Operations

`1.12.0` supports gradients for:

- `add`
- `sub`
- `mul`
- `div`
- `neg`
- scalar `pow`
- `matmul`
- `bmm`
- 3D batched `@`
- `sum`
- `mean`
- `var`
- `std`
- `norm` through differentiable reduction paths
- `max`
- `min`
- `relu`
- `sigmoid`
- `tanh`
- `exp`
- `log`
- `logsumexp`
- `softmax`
- `log_softmax`
- `log1p`
- `sqrt`
- `rsqrt`
- `sin`
- `cos`
- `tan`
- `asin`
- `acos`
- `atan`
- `abs`
- `reshape`
- `flatten`
- `transpose`
- `permute`
- `squeeze`
- `unsqueeze`
- basic integer and slice indexing views

## Coverage Matrix

| Operation family | Forward support | Backward support | Notes |
|---|---|---|---|
| Binary arithmetic | yes | yes | Broadcasting gradients are reduced back to input shapes. |
| Comparisons | yes | no | Return `bool` tensors. |
| Selection | yes | partial | `where`, `maximum`, and `minimum` differentiate selected floating branches. |
| Matrix multiply | yes | yes | Includes 2D `matmul` and 3D batched `bmm`. |
| Reductions | yes | partial | `sum`, `mean`, `var`, `std`, `max`, and `min` differentiate; arg/boolean reductions do not. |
| Stable probability ops | yes | yes | `logsumexp`, `softmax`, and `log_softmax` are finite-difference tested. |
| Views/layout | yes | yes | Metadata views propagate gradients back to the original layout. |
| Indexing views | basic | basic | Integer and slice views scatter gradients back; advanced indexing is not implemented. |
| Convolution/pooling | yes | yes | Native CPU NCHW kernels with finite-difference coverage. |
| Random ops | yes | no | Random sampling is intentionally non-differentiable. |

## Broadcasting Gradients

When an operation broadcasts an input, the backward pass reduces the gradient
back to the original input shape.

```python
x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = ts.tensor([10.0, 20.0], requires_grad=True)

loss = (x + b).sum()
loss.backward()

print(b.grad.tolist())  # [2.0, 2.0]
```

## Non-Scalar Outputs

`backward()` without a gradient is only valid for scalar outputs. For non-scalar
outputs, pass an explicit gradient if supported by the operation chain.

```python
y = x * x
y.backward(ts.ones(y.shape))
```

## Graph Lifecycle

```python
x = ts.tensor([1.0, 2.0], requires_grad=True)
loss = (x * x).sum()

loss.backward(retain_graph=True)
loss.backward()
```

The first call keeps the graph alive. The second call frees it. A third call on
the same `loss` raises an `AutogradError`.

Use `clear_history()` when you intentionally want to turn a non-leaf tensor
into a leaf-like tensor without changing its values:

```python
y = (x * 2.0)
y.clear_history()
```

`detach_()` clears history and disables gradient tracking in place.

## Zeroing Gradients

Gradients accumulate until cleared.

```python
x.zero_grad()
```

Optimizers also expose `zero_grad()`.

## Limitations

- No graph compiler.
- Public `zero_`, `fill_`, and `add_` are guarded and should be used inside
  `no_grad()` when mutating tensors that require gradients.
- No higher-order gradients.
- No gradient checkpointing.
- Basic integer and slice indexing gradients scatter back into the source
  tensor. Advanced list, tensor, and boolean-mask indexing gradients are not
  implemented.
- CPU tensors only.
