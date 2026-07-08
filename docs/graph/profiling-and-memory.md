# Profiling And Memory Planning

Compiled graphs expose runtime profiling hooks and a simple static memory plan.

## Profiling

```python
compiled = ts.compile_graph(
    lambda x, y: (x @ y).tanh(),
    [ts.TensorSpec((2, 2)), ts.TensorSpec((2, 2))],
)

profile = compiled.profile(ts.ones((2, 2)), ts.ones((2, 2)))
print(profile["total_ms"])
print(profile["nodes"])
```

Profiles report per-node elapsed wall time in milliseconds. They are useful for
local investigation, not cross-machine benchmark claims.

## Memory Plan

```python
plan = compiled.memory_plan()
for item in plan:
    print(item["id"], item["estimated_bytes"], item["last_use_index"])
```

The memory plan records shape, dtype, estimated output bytes, and last-use index
for each node. It does not yet reuse buffers, but it gives the runtime enough
metadata for future memory planning work.
