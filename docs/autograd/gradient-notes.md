# Gradient Notes

TensorStudio uses eager reverse-mode autodiff. Each differentiable native op can
attach a small backward function to its output tensor.

## Scalar Backward

Calling `backward()` without an explicit gradient is valid for scalar outputs:

```python
import tensorstudio as ts

x = ts.tensor([1.0, 2.0, 3.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad.tolist())
```

For non-scalar outputs, pass an explicit gradient tensor.

## Broadcasting Gradients

When a forward operation broadcasts an input, the backward pass sums gradients
back to the original input shape.

```python
x = ts.ones((2, 3), requires_grad=True)
b = ts.ones((3,), requires_grad=True)
(x + b).sum().backward()
print(b.grad.tolist())  # [2.0, 2.0, 2.0]
```

## Selection Gradients

`where(condition, a, b)` does not differentiate through the boolean condition.
Gradients flow to `a` where the condition is true and to `b` where it is false.

`maximum(a, b)` and `minimum(a, b)` split gradients evenly when values are tied.
This matches the tie-handling used by TensorStudio's extrema reductions and
avoids unstable left-or-right-only behavior.

## Non-Differentiable Outputs

Integer and boolean tensors do not carry autograd history. Arg reductions return
`int64` indices and are intentionally non-differentiable.

