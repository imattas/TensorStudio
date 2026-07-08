# Distributed Research And ONNX Runtime

TensorStudio does not ship production distributed training. The distributed
namespace exposes explicit research helpers so small projects can plan for
multi-process training without pretending that collectives are implemented.

```python
import tensorstudio as ts

config = ts.distributed.DistributedConfig()
print(ts.distributed.distributed_info(config))
print(ts.distributed.data_parallel_plan(dataset_size=100, batch_size=8, config=config))
```

Single-process `all_reduce_sum` returns a clone. Multi-process collectives raise
`NotImplementedError` until a real backend is implemented and tested.

## ONNX Runtime Adapter

TensorStudio has two ONNX execution paths:

- `import_onnx(path)` runs a supported static subset through TensorStudio ops.
- `run_onnx(path, input)` uses optional ONNX Runtime when installed, otherwise
  falls back to TensorStudio import.

```bash
python -m pip install "tensorstudio[onnxruntime]"
```

```python
import tensorstudio as ts

x = ts.ones((1, 2))
print(ts.onnxruntime_is_available())
print(ts.run_onnx("model.onnx", x).shape)
```

ONNX Runtime can execute a broader ONNX surface than TensorStudio's importer,
but it is an optional dependency and separate runtime.
