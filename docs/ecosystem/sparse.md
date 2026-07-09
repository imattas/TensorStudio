# Sparse Tensors

TensorStudio includes experimental COO and CSR sparse tensor containers for
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
- CSR construction from row pointers, columns, and values
- duplicate-index coalescing
- dense conversion
- rank-2 transpose
- sparse-dense vector or matrix multiplication

```python
dense = ts.tensor([[0.0, 2.0], [3.0, 0.0]])
csr = ts.csr_from_dense(dense)
print(csr.to_dense().tolist())
print((csr @ ts.ones((2, 1))).tolist())
```

The sparse API is not yet a full sparse autograd backend. Convert to dense when
you need regular TensorStudio tensor operations.
