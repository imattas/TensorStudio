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
4. Walks the graph in reverse.
5. Calls each operation's backward function.
6. Accumulates gradients on leaf and intermediate tensors that require them.

## Supported Gradient Operations

v0.1.x supports gradients for:

- `add`
- `sub`
- `mul`
- `div`
- `neg`
- scalar `pow`
- `matmul`
- `sum`
- `mean`
- `relu`
- `sigmoid`
- `tanh`
- `exp`
- `log`
- `reshape`
- `transpose`

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

## Zeroing Gradients

Gradients accumulate until cleared.

```python
x.zero_grad()
```

Optimizers also expose `zero_grad()`.

## Limitations

- No graph compiler.
- No in-place operation tracking beyond optimizer assignment helpers.
- No higher-order gradients.
- No gradient checkpointing.
- No advanced indexing gradients.
- CPU tensors only.
