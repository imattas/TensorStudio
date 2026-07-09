# Serialization And Optimization

Graph serialization uses a JSON file with TensorStudio node metadata. It is
intended for supported traced graphs, not arbitrary Python modules.

## Save And Load

```python
graph = ts.trace(lambda x: x * 2.0 + 1.0, [ts.TensorSpec((2,))])
ts.save_graph(graph, "linear.tsgraph.json")
loaded = ts.load_graph("linear.tsgraph.json")
```

The JSON payload records:

- format and version
- nodes
- graph input ids
- graph output ids
- node op names, inputs, attributes, shape, dtype, and optional names

## Optimization Passes

```python
optimized = graph.optimize()
compiled = optimized.compile(optimize=False)
```

`1.15.0` includes:

- constant folding for supported all-constant subgraphs
- a simple multiply-add fusion for scalar `x * scale + bias` patterns

These passes keep the graph interpretable. They do not claim the performance of
industrial graph compilers.
