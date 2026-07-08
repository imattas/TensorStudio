# Public Dataset Formats

TensorStudio can load common small public data formats into map-style datasets.

```python
import tensorstudio as ts

csv_data = ts.data.from_csv(
    "train.csv",
    target_column="label",
    target_dtype="int64",
)

jsonl_data = ts.data.from_jsonl("records.jsonl", target_dtype="int64")
text_data = ts.data.from_text_lines("corpus.txt")
libsvm_data = ts.data.from_libsvm("train.libsvm", num_features=128)
```

Supported loaders:

- `from_csv`
- `from_jsonl`
- `from_text_lines`
- `from_libsvm`

These helpers are intended for local experiments and reproducible examples.
Large-scale streaming and multiprocessing ingestion remain outside the current
scope.
