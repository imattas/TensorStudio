from __future__ import annotations

import numpy as np
import tensorstudio as ts


def test_trace_compile_and_run_matches_eager() -> None:
    spec = ts.TensorSpec((4,), name="x")

    def model(x: ts.graph.GraphTensor) -> ts.graph.GraphTensor:
        return ((x * 2.0) + 1.0).relu().mean()

    graph = ts.trace(model, [spec])
    executable = graph.compile()
    x = ts.tensor([-2.0, -0.25, 1.0, 3.0])

    expected = ((x * 2.0) + 1.0).relu().mean()
    actual = executable(x)

    np.testing.assert_allclose(actual.numpy(), expected.numpy())


def test_graph_serialization_roundtrip(tmp_path) -> None:
    graph = ts.trace(lambda x: (x + 1.0).sigmoid(), [ts.TensorSpec((3,), dtype="float32")])
    path = tmp_path / "graph.tsgraph.json"

    ts.save_graph(graph, path)
    loaded = ts.load_graph(path)

    x = ts.tensor([0.0, 1.0, 2.0])
    np.testing.assert_allclose(loaded.run(x).numpy(), graph.run(x).numpy())


def test_graph_optimization_fuses_linear_pattern() -> None:
    graph = ts.trace(lambda x: x * 2.0 + 1.0, [ts.TensorSpec((2,))])
    optimized = graph.optimize()

    assert any(node.op == "fused_mul_add" for node in optimized.nodes)
    np.testing.assert_allclose(
        optimized.run(ts.tensor([1.0, 2.0])).numpy(),
        np.array([3.0, 5.0], dtype=np.float32),
    )


def test_graph_profile_and_memory_plan() -> None:
    executable = ts.compile_graph(
        lambda x, y: (x @ y).tanh(),
        [ts.TensorSpec((2, 2), name="x"), ts.TensorSpec((2, 2), name="y")],
    )

    x = ts.ones((2, 2))
    y = ts.ones((2, 2))
    profile = executable.profile(x, y)
    plan = executable.memory_plan()

    assert profile["total_ms"] >= 0.0
    assert any(node["op"] == "matmul" for node in profile["nodes"])
    assert any(item["estimated_bytes"] >= 16 for item in plan)
