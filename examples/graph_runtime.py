from __future__ import annotations

from pathlib import Path

import tensorstudio as ts


def model(x: ts.graph.GraphTensor) -> ts.graph.GraphTensor:
    return ((x * 2.0) + 1.0).relu().mean()


def main() -> None:
    graph = ts.trace(model, [ts.TensorSpec((4,), dtype="float32", name="x")])
    compiled = graph.compile()
    x = ts.tensor([-2.0, -0.25, 1.0, 3.0])

    output = compiled(x)
    print("output:", output.item())
    print("optimized ops:", [node.op for node in graph.optimize().nodes])
    print("profile nodes:", len(compiled.profile(x)["nodes"]))
    print("memory plan entries:", len(compiled.memory_plan()))

    path = Path("graph_runtime_example.tsgraph.json")
    ts.save_graph(graph, path)
    loaded = ts.load_graph(path)
    print("loaded output:", loaded.run(x).item())
    path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
