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

## Manifests And Caching

TensorStudio can build deterministic file manifests with SHA-256 checksums for
local datasets:

```python
manifest = ts.data.build_dataset_manifest("data/images")
manifest.save("data/images.manifest.json")
print(ts.data.validate_dataset_manifest("data/images.manifest.json"))
```

Wrap deterministic map-style datasets in a small memory cache when repeated
sample access is expected:

```python
cached = ts.data.cache_dataset(csv_data, max_items=256)
features, target = cached[0]
```

These helpers do not download data or execute code from manifests.
