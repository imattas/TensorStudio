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
4. Walks the graph in reverse.
5. Calls operation-specific backward functions.
6. Accumulates gradients into tensors that require gradients.

For non-scalar outputs, pass an explicit gradient:

```python
y = x * 2.0
y.backward(ts.ones(y.shape))
```

Gradients accumulate until cleared:

```python
x.zero_grad()
optimizer.zero_grad()
```

## Supported Gradients

The release candidate supports gradients for:

- `add`, `sub`, `mul`, `div`
- `neg`
- scalar `pow`
- `matmul`
- `sum`, `mean`, `max`, `min`
- `relu`, `sigmoid`, `tanh`
- `exp`, `log`, `sqrt`, `abs`
- `reshape`, `flatten`, `transpose`
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
history.

`clone()` copies tensor values into new storage.

```python
detached = x.detach()
copied = x.clone()
```

## Limitations

- No higher-order gradients.
- No graph compiler or tracing runtime.
- No checkpointing.
- No advanced indexing gradients.
- No in-place operation tracking beyond optimizer internals.
- CPU tensors only.
