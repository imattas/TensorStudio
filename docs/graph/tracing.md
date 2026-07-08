# Graph Tracing

TensorStudio `1.15.0` adds a constrained symbolic graph system. It is not a
full Python tracer and does not capture arbitrary control flow. Instead, it
traces functions written against `GraphTensor` placeholders and then executes
the resulting graph through TensorStudio eager tensor operations.

## Trace A Function

```python
import tensorstudio as ts

def model(x):
    return ((x * 2.0) + 1.0).relu().mean()

graph = ts.trace(model, [ts.TensorSpec((4,), dtype="float32", name="x")])
compiled = graph.compile()
out = compiled(ts.tensor([-2.0, -0.25, 1.0, 3.0]))
```

Supported traced operations include:

- `+`, `-`, `*`, `/`
- unary `-`
- `@` for rank-2 matrix multiplication
- `relu`, `sigmoid`, `tanh`, `exp`, `log`
- `sum`, `mean`
- `reshape`, `flatten`, rank-2 `transpose`, and `T`

Unsupported Python control flow should stay in eager code until the graph
runtime grows stronger.

## Compile A Callable

```python
compiled = ts.compile_graph(
    lambda x, y: (x @ y).tanh(),
    [ts.TensorSpec((2, 2), name="x"), ts.TensorSpec((2, 2), name="y")],
)
```

`compile_graph()` returns an `ExecutableGraph`. It is a small ahead-of-time plan
for the supported graph, not a machine-code JIT.
