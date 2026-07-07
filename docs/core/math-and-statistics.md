# Math And Statistics

TensorStudio `1.7.0` expands the C++ math core beyond basic arithmetic and
activations. The goal is not full NumPy coverage yet; the goal is a compact,
well-tested set of operations that model code needs immediately.

## Stable Probability Ops

`softmax`, `log_softmax`, and `logsumexp` use max-shifted numerics so large
logits remain finite:

```python
import tensorstudio as ts

logits = ts.tensor([[1000.0, 1001.0, 999.0]])

prob = logits.softmax(axis=1)
log_prob = logits.log_softmax(axis=1)
normalizer = logits.logsumexp(axis=1)

print(prob.tolist())
print(log_prob.tolist())
print(normalizer.tolist())
```

The same operations are also exposed as functional helpers:

```python
ts.softmax(logits, axis=1)
ts.log_softmax(logits, axis=1)
ts.logsumexp(logits, axis=1, keepdims=True)
```

`tensorstudio.nn.functional.cross_entropy` uses `log_softmax`, so classifier
training gets the same stable numerics.

## Batched Matrix Multiplication

`matmul` still handles 2D matrix multiplication. In `1.7.0`, two 3D tensors
dispatch to batched matrix multiplication:

```python
left = ts.randn((4, 8, 16), seed=1)
right = ts.randn((4, 16, 32), seed=2)

out = left @ right
also_out = ts.bmm(left, right)

print(out.shape)       # (4, 8, 32)
print(also_out.shape)  # (4, 8, 32)
```

The required shape contract is:

- left: `(batch, rows, inner)`
- right: `(batch, inner, columns)`
- output: `(batch, rows, columns)`

Both operands receive gradients when they are floating point tensors with
`requires_grad=True`.

## Statistical Reductions

Tensor methods:

```python
x = ts.tensor([[1.0, 2.0, 5.0], [2.0, 4.0, 8.0]])

x.var()
x.variance(axis=1)
x.std(axis=0, correction=1)
x.norm(ord=2)
x.norm(ord=1, axis=0)
```

Functional helpers:

```python
ts.var(x)
ts.variance(x, axis=1)
ts.std(x, axis=0)
ts.norm(x)
```

The native Tensor `var`/`std` methods support all-elements and single-axis
reductions. `tensorstudio.math.variance` and `tensorstudio.math.std` also
support tuple/list axes by composing native C++ reductions.

`correction=1` gives the usual sample-variance denominator. TensorStudio raises
a shape error when the corrected denominator would be zero or negative.

## Boolean Reductions

`all` and `any` treat non-zero values as true and return `bool` tensors:

```python
mask = ts.tensor([[True, False, True], [True, True, True]])

mask.all()
mask.any(axis=1)
ts.all(mask, axis=(0, 1))
ts.any(mask, keepdims=True)
```

These reductions are not differentiable.

## Random Distributions

Seeded native random helpers:

```python
ts.rand((2, 3), seed=1)
ts.randn((2, 3), seed=2)
ts.uniform((2, 3), low=-1.0, high=1.0, seed=3)
ts.normal((2, 3), mean=5.0, stddev=0.25, seed=4)
ts.randint((8,), low=0, high=10, seed=5)
ts.bernoulli((8,), probability=0.25, seed=6)
```

`rand` is an alias for `uniform(..., low=0.0, high=1.0)`. `randn` is an alias
for `normal(..., mean=0.0, stddev=1.0)`.

Calling `manual_seed(seed)` seeds TensorStudio's process-local generator.
Passing `seed=` to an individual random call uses a temporary generator and
does not advance the process-local generator.

## Practical `einsum` Subset

TensorStudio supports a small explicit subset of `einsum` patterns:

| Pattern | Meaning |
|---|---|
| `ij,jk->ik` | matrix multiplication |
| `bi,ij->bj` | batched row matrix multiplication |
| `bij,bjk->bik` | batched matrix multiplication |
| `i,i->` | vector dot product |
| `ij,ij->` | Frobenius inner product |
| `ij->ji` | matrix transpose |
| `bij->bji` | transpose the last two axes of a batch |
| `i->` | vector sum |
| `ij->` | matrix sum |
| `ij->i` | row sums |
| `ij->j` | column sums |

Unsupported patterns raise a clear `ValueError`. That is intentional: this is
a reliable subset, not a compatibility claim for the full NumPy `einsum`
language.

```python
a = ts.randn((2, 3), seed=1)
b = ts.randn((3, 4), seed=2)

ts.einsum("ij,jk->ik", a, b)
ts.einsum("ij,ij->", a, a)
ts.einsum("ij->j", a)
```

## Autograd Notes

Gradient support is available for:

- `bmm`
- 3D `@`
- `logsumexp`
- `softmax`
- `log_softmax`
- `var`
- `std`
- `norm` for differentiable paths built from differentiable reductions

`all`, `any`, `randint`, and `bernoulli` are intentionally non-differentiable.
