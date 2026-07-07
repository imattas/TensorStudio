# Autograd

TensorStudio includes a compact reverse-mode autograd engine for eager tensor
programs.

## Basic Use

```python
import tensorstudio as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()

print(x.grad.tolist())
```

Only floating point tensors can require gradients.

## Graph Recording

When graph recording is enabled and at least one parent has
`requires_grad=True`, differentiable operations attach autograd metadata to the
output tensor. That metadata stores parent tensors and a backward function.

```python
with ts.no_grad():
    y = x * 2.0

print(y.requires_grad)  # False
```

`no_grad()` is thread-local through the native extension state.

## Backward

`Tensor.backward()`:

1. Validates the output tensor.
2. Creates an all-ones gradient for scalar outputs when no gradient is supplied.
3. Builds a deterministic topological order.
4. Clears stale intermediate gradients from previous retained-graph passes.
5. Walks the graph in reverse.
6. Calls operation-specific backward functions.
7. Accumulates gradients into leaf tensors that require gradients.
8. Releases non-leaf graph history unless `retain_graph=True` is supplied.

For non-scalar outputs, pass an explicit gradient:

```python
y = x * 2.0
y.backward(ts.ones(y.shape))
```

Reuse a graph only when needed:

```python
loss = (x * x).sum()
loss.backward(retain_graph=True)
loss.backward()
```

The final call frees the graph. Calling `backward()` again on the same non-leaf
output raises an error.

Gradients accumulate until cleared:

```python
x.zero_grad()
optimizer.zero_grad()
```

## Supported Gradients

The v1 release supports gradients for:

- `add`, `sub`, `mul`, `div`
- `neg`
- scalar `pow`
- `matmul`, `bmm`
- `sum`, `mean`, `var`, `std`, `max`, `min`
- `logsumexp`, `softmax`, `log_softmax`
- `relu`, `sigmoid`, `tanh`
- `exp`, `log`, `log1p`, `sqrt`, `rsqrt`, `abs`
- trigonometric and inverse trigonometric functions
- `reshape`, `flatten`, `transpose`, `permute`, `squeeze`, `unsqueeze`
- basic integer and slice indexing views
- broadcasted binary operations

`max` and `min` split the incoming gradient evenly across tied extrema.

## Broadcasting Gradients

Broadcasted inputs receive gradients reduced back to the original input shape.

```python
x = ts.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
b = ts.tensor([10.0, 20.0], requires_grad=True)

loss = (x + b).sum()
loss.backward()

print(x.grad.tolist())
print(b.grad.tolist())  # [2.0, 2.0]
```

## Detach And Clone

`detach()` returns a tensor view that shares storage but does not carry autograd
history. `detach_()` clears history and disables gradient tracking on the
tensor in place.

`clone()` copies tensor values into new storage.
`clear_history()` removes parent/backward metadata while preserving
`requires_grad`.

```python
detached = x.detach()
copied = x.clone()
y = (x * 2.0).clear_history()
```

## Safe In-Place Mutation

Public in-place mutation is intentionally small:

- `zero_()`
- `fill_(value)`
- `add_(other, alpha=1.0)`

These methods reject mutation of tensors that require gradients while grad mode
is enabled. Use them inside `no_grad()` for parameter initialization or manual
state updates:

```python
with ts.no_grad():
    parameter.zero_()
```

## Limitations

- No higher-order gradients.
- No graph compiler or tracing runtime.
- No checkpointing.
- Basic indexing and slicing views scatter gradients back into the source.
  Advanced list, tensor, and boolean-mask indexing gradients are not
  implemented.
- No general in-place autograd tracking.
- CPU tensors only.
