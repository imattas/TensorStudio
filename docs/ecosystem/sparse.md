# Sparse Tensors

TensorStudio `1.16.0` includes an experimental COO sparse tensor container for
small sparse workflows.

```python
import tensorstudio as ts

sparse = ts.sparse_coo_tensor(
    [[0, 1], [1, 0], [1, 0]],
    [2.0, 3.0, 4.0],
    (2, 2),
)

coalesced = sparse.coalesce()
print(coalesced.to_dense().tolist())
print((coalesced @ ts.ones((2, 1))).tolist())
```

Supported operations:

- COO construction from indices and values
- duplicate-index coalescing
- dense conversion
- rank-2 transpose
- sparse-dense vector or matrix multiplication

The sparse API is not yet a full sparse autograd backend. Convert to dense when
you need regular TensorStudio tensor operations.
