"""Constrained symbolic graph tracing and execution."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..tensor import Tensor, tensor

ShapeLike = tuple[int, ...]
JsonDict = dict[str, Any]


@dataclass(frozen=True)
class TensorSpec:
    """Input or output tensor metadata for graph tracing."""

    shape: ShapeLike
    dtype: str = "float32"
    name: str | None = None

    def to_dict(self) -> JsonDict:
        return {"shape": list(self.shape), "dtype": self.dtype, "name": self.name}

    @classmethod
    def from_dict(cls, data: JsonDict) -> TensorSpec:
        return cls(tuple(int(dim) for dim in data["shape"]), str(data["dtype"]), data.get("name"))


@dataclass
class GraphNode:
    """Single operation in a TensorStudio graph."""

    id: str
    op: str
    inputs: list[str] = field(default_factory=list)
    attrs: JsonDict = field(default_factory=dict)
    shape: ShapeLike = field(default_factory=tuple)
    dtype: str = "float32"
    name: str | None = None

    def to_dict(self) -> JsonDict:
        return {
            "id": self.id,
            "op": self.op,
            "inputs": list(self.inputs),
            "attrs": dict(self.attrs),
            "shape": list(self.shape),
            "dtype": self.dtype,
            "name": self.name,
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> GraphNode:
        return cls(
            id=str(data["id"]),
            op=str(data["op"]),
            inputs=[str(value) for value in data.get("inputs", [])],
            attrs=dict(data.get("attrs", {})),
            shape=tuple(int(dim) for dim in data.get("shape", [])),
            dtype=str(data.get("dtype", "float32")),
            name=data.get("name"),
        )


class GraphTensor:
    """Symbolic tensor used while tracing a constrained graph."""

    def __init__(self, graph: GraphBuilder, node_id: str) -> None:
        self._graph = graph
        self.node_id = node_id

    @property
    def shape(self) -> ShapeLike:
        return self._graph.node(self.node_id).shape

    @property
    def dtype(self) -> str:
        return self._graph.node(self.node_id).dtype

    @property
    def T(self) -> GraphTensor:
        return self.transpose()

    def __add__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("add", self, other)

    def __radd__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("add", other, self)

    def __sub__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("sub", self, other)

    def __rsub__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("sub", other, self)

    def __mul__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("mul", self, other)

    def __rmul__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("mul", other, self)

    def __truediv__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("div", self, other)

    def __rtruediv__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("div", other, self)

    def __neg__(self) -> GraphTensor:
        return self._graph.unary("neg", self)

    def __matmul__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("matmul", self, other)

    def __rmatmul__(self, other: GraphValue) -> GraphTensor:
        return self._graph.binary("matmul", other, self)

    def relu(self) -> GraphTensor:
        return self._graph.unary("relu", self)

    def sigmoid(self) -> GraphTensor:
        return self._graph.unary("sigmoid", self)

    def tanh(self) -> GraphTensor:
        return self._graph.unary("tanh", self)

    def exp(self) -> GraphTensor:
        return self._graph.unary("exp", self)

    def log(self) -> GraphTensor:
        return self._graph.unary("log", self)

    def sum(self) -> GraphTensor:
        return self._graph.reduction("sum", self)

    def mean(self) -> GraphTensor:
        return self._graph.reduction("mean", self)

    def reshape(self, *shape: int | Sequence[int]) -> GraphTensor:
        normalized = _normalize_shape_args(shape)
        return self._graph.view("reshape", self, normalized)

    def flatten(self) -> GraphTensor:
        return self._graph.view("flatten", self, (_numel(self.shape),))

    def transpose(self) -> GraphTensor:
        if len(self.shape) != 2:
            raise ValueError("graph transpose currently supports rank-2 tensors")
        return self._graph.view("transpose", self, (self.shape[1], self.shape[0]))


GraphValue = GraphTensor | Tensor | float | int | bool


class GraphBuilder:
    """Mutable graph builder used during tracing."""

    def __init__(self) -> None:
        self.nodes: list[GraphNode] = []
        self.inputs: list[str] = []
        self.outputs: list[str] = []
        self._next_id = 0

    def node(self, node_id: str) -> GraphNode:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise KeyError(node_id)

    def input(self, spec: TensorSpec, index: int) -> GraphTensor:
        node_id = self._new_id("input")
        name = spec.name or f"input_{index}"
        self.nodes.append(
            GraphNode(node_id, "input", shape=spec.shape, dtype=spec.dtype, name=name)
        )
        self.inputs.append(node_id)
        return GraphTensor(self, node_id)

    def constant(self, value: Tensor | float | int | bool) -> GraphTensor:
        if isinstance(value, Tensor):
            node_id = self._new_id("const")
            self.nodes.append(
                GraphNode(
                    node_id,
                    "const",
                    attrs={"value": value.tolist()},
                    shape=tuple(int(dim) for dim in value.shape),
                    dtype=value.dtype,
                )
            )
            return GraphTensor(self, node_id)
        dtype = "bool" if isinstance(value, bool) else "float32"
        node_id = self._new_id("const")
        self.nodes.append(GraphNode(node_id, "const", attrs={"value": value}, dtype=dtype))
        return GraphTensor(self, node_id)

    def binary(self, op: str, left: GraphValue, right: GraphValue) -> GraphTensor:
        left_tensor = self.ensure_graph_tensor(left)
        right_tensor = self.ensure_graph_tensor(right)
        left_node = self.node(left_tensor.node_id)
        right_node = self.node(right_tensor.node_id)
        shape = _infer_binary_shape(left_node.shape, right_node.shape, op)
        dtype = _infer_binary_dtype(left_node.dtype, right_node.dtype, op)
        return self._add_node(op, [left_tensor.node_id, right_tensor.node_id], shape, dtype)

    def unary(self, op: str, input: GraphTensor) -> GraphTensor:
        node = self.node(input.node_id)
        return self._add_node(op, [input.node_id], node.shape, node.dtype)

    def reduction(self, op: str, input: GraphTensor) -> GraphTensor:
        node = self.node(input.node_id)
        return self._add_node(op, [input.node_id], (), node.dtype)

    def view(self, op: str, input: GraphTensor, shape: ShapeLike) -> GraphTensor:
        node = self.node(input.node_id)
        return self._add_node(op, [input.node_id], shape, node.dtype, {"shape": list(shape)})

    def ensure_graph_tensor(self, value: GraphValue) -> GraphTensor:
        if isinstance(value, GraphTensor):
            if value._graph is not self:
                raise ValueError("cannot mix GraphTensor values from different traces")
            return value
        return self.constant(value)

    def finalize(self, outputs: GraphTensor | Sequence[GraphTensor]) -> Graph:
        if isinstance(outputs, GraphTensor):
            output_ids = [outputs.node_id]
        else:
            output_ids = [output.node_id for output in outputs]
        self.outputs = output_ids
        return Graph(list(self.nodes), list(self.inputs), output_ids)

    def _new_id(self, prefix: str) -> str:
        node_id = f"{prefix}_{self._next_id}"
        self._next_id += 1
        return node_id

    def _add_node(
        self,
        op: str,
        inputs: list[str],
        shape: ShapeLike,
        dtype: str,
        attrs: JsonDict | None = None,
    ) -> GraphTensor:
        node_id = self._new_id(op)
        self.nodes.append(GraphNode(node_id, op, inputs, dict(attrs or {}), shape, dtype))
        return GraphTensor(self, node_id)


class Graph:
    """Serializable TensorStudio graph."""

    def __init__(self, nodes: list[GraphNode], inputs: list[str], outputs: list[str]) -> None:
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs

    def to_dict(self) -> JsonDict:
        return {
            "format": "tensorstudio.graph",
            "version": 1,
            "nodes": [node.to_dict() for node in self.nodes],
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
        }

    @classmethod
    def from_dict(cls, data: JsonDict) -> Graph:
        if data.get("format") != "tensorstudio.graph":
            raise ValueError("not a TensorStudio graph payload")
        if data.get("version") != 1:
            raise ValueError(f"unsupported TensorStudio graph version: {data.get('version')}")
        return cls(
            [GraphNode.from_dict(node) for node in data["nodes"]],
            [str(value) for value in data["inputs"]],
            [str(value) for value in data["outputs"]],
        )

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
        return output

    def optimize(self, *, constant_folding: bool = True, fuse_linear: bool = True) -> Graph:
        graph = self
        if constant_folding:
            graph = _constant_fold(graph)
        if fuse_linear:
            graph = _fuse_linear(graph)
        return graph

    def compile(self, *, optimize: bool = True) -> ExecutableGraph:
        graph = self.optimize() if optimize else self
        return ExecutableGraph(graph)

    def run(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        return self.compile(optimize=False).run(*inputs)

    def memory_plan(self) -> list[JsonDict]:
        return _memory_plan(self)


class ExecutableGraph:
    """Executable graph plan backed by TensorStudio eager tensor operations."""

    def __init__(self, graph: Graph) -> None:
        self.graph = graph

    def __call__(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        return self.run(*inputs)

    def run(self, *inputs: Tensor) -> Tensor | tuple[Tensor, ...]:
        outputs, _ = self._execute(inputs, profile=False)
        return _pack_outputs(outputs)

    def profile(self, *inputs: Tensor) -> JsonDict:
        outputs, profiles = self._execute(inputs, profile=True)
        return {
            "total_ms": sum(float(item["elapsed_ms"]) for item in profiles),
            "nodes": profiles,
            "output_count": len(outputs),
        }

    def memory_plan(self) -> list[JsonDict]:
        return self.graph.memory_plan()

    def _execute(
        self,
        inputs: Sequence[Tensor],
        *,
        profile: bool,
    ) -> tuple[list[Tensor], list[JsonDict]]:
        if len(inputs) != len(self.graph.inputs):
            raise ValueError(f"expected {len(self.graph.inputs)} graph inputs, got {len(inputs)}")
        env: dict[str, Tensor] = dict(zip(self.graph.inputs, inputs, strict=True))
        profiles: list[JsonDict] = []
        for node in self.graph.nodes:
            if node.op == "input":
                continue
            start = time.perf_counter()
            env[node.id] = _execute_node(node, env)
            elapsed_ms = (time.perf_counter() - start) * 1_000.0
            if profile:
                profiles.append(
                    {
                        "id": node.id,
                        "op": node.op,
                        "shape": list(node.shape),
                        "dtype": node.dtype,
                        "elapsed_ms": elapsed_ms,
                    }
                )
        return [env[node_id] for node_id in self.graph.outputs], profiles


def trace(function: Callable[..., GraphTensor], input_specs: Sequence[TensorSpec]) -> Graph:
    """Trace a constrained symbolic TensorStudio function."""

    builder = GraphBuilder()
    symbolic_inputs = [builder.input(spec, index) for index, spec in enumerate(input_specs)]
    outputs = function(*symbolic_inputs)
    if not isinstance(outputs, (GraphTensor, tuple, list)):
        raise TypeError(
            "trace function must return a GraphTensor or a sequence of GraphTensor values"
        )
    return builder.finalize(outputs)


def compile_graph(
    function_or_graph: Callable[..., GraphTensor] | Graph,
    input_specs: Sequence[TensorSpec] | None = None,
    *,
    optimize: bool = True,
) -> ExecutableGraph:
    """Trace or compile a supported graph into an executable plan."""

    graph = function_or_graph
    if not isinstance(graph, Graph):
        if input_specs is None:
            raise ValueError("input_specs are required when compiling a callable")
        graph = trace(graph, input_specs)
    return graph.compile(optimize=optimize)


def load_graph(path: str | Path) -> Graph:
    """Load a TensorStudio graph JSON file."""

    return Graph.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_graph(graph: Graph, path: str | Path) -> Path:
    """Save a TensorStudio graph JSON file."""

    return graph.save(path)


def _execute_node(node: GraphNode, env: dict[str, Tensor]) -> Tensor:
    values = [env[node_id] for node_id in node.inputs]
    if node.op == "const":
        return tensor(node.attrs["value"], dtype=node.dtype)
    if node.op == "add":
        return values[0] + values[1]
    if node.op == "sub":
        return values[0] - values[1]
    if node.op == "mul":
        return values[0] * values[1]
    if node.op == "div":
        return values[0] / values[1]
    if node.op == "matmul":
        return values[0] @ values[1]
    if node.op == "neg":
        return -values[0]
    if node.op in {"relu", "sigmoid", "tanh", "exp", "log"}:
        return getattr(values[0], node.op)()
    if node.op in {"sum", "mean"}:
        return getattr(values[0], node.op)()
    if node.op == "reshape":
        return values[0].reshape(tuple(int(dim) for dim in node.attrs["shape"]))
    if node.op == "flatten":
        return values[0].flatten()
    if node.op == "transpose":
        return values[0].transpose()
    if node.op == "fused_mul_add":
        return values[0] * float(node.attrs["scale"]) + float(node.attrs["bias"])
    raise ValueError(f"unsupported graph op: {node.op}")


def _constant_fold(graph: Graph) -> Graph:
    constants: dict[str, Tensor] = {}
    new_nodes: list[GraphNode] = []
    for node in graph.nodes:
        if node.op == "const":
            constants[node.id] = tensor(node.attrs["value"], dtype=node.dtype)
            new_nodes.append(node)
            continue
        if node.op == "input":
            new_nodes.append(node)
            continue
        if all(input_id in constants for input_id in node.inputs):
            folded = _execute_node(node, constants)
            folded_node = GraphNode(
                node.id,
                "const",
                attrs={"value": folded.tolist()},
                shape=tuple(int(dim) for dim in folded.shape),
                dtype=folded.dtype,
                name=node.name,
            )
            constants[node.id] = folded
            new_nodes.append(folded_node)
            continue
        new_nodes.append(node)
    return Graph(new_nodes, list(graph.inputs), list(graph.outputs))


def _fuse_linear(graph: Graph) -> Graph:
    by_id = {node.id: node for node in graph.nodes}
    consumed: set[str] = set()
    fused_nodes: list[GraphNode] = []
    for node in graph.nodes:
        if node.id in consumed:
            continue
        fused = _try_fuse_mul_add(node, by_id)
        if fused is not None:
            fused_nodes.append(fused)
            consumed.update(node.inputs)
            continue
        fused_nodes.append(node)
    return _prune_dead_nodes(Graph(fused_nodes, list(graph.inputs), list(graph.outputs)))


def _prune_dead_nodes(graph: Graph) -> Graph:
    by_id = {node.id: node for node in graph.nodes}
    reachable: set[str] = set(graph.inputs)
    stack = list(graph.outputs)
    while stack:
        node_id = stack.pop()
        if node_id in reachable:
            continue
        reachable.add(node_id)
        node = by_id.get(node_id)
        if node is not None:
            stack.extend(node.inputs)
    nodes = [node for node in graph.nodes if node.id in reachable]
    return Graph(nodes, list(graph.inputs), list(graph.outputs))


def _try_fuse_mul_add(node: GraphNode, by_id: dict[str, GraphNode]) -> GraphNode | None:
    if node.op != "add" or len(node.inputs) != 2:
        return None
    left = by_id[node.inputs[0]]
    right = by_id[node.inputs[1]]
    mul_node = left if left.op == "mul" else right if right.op == "mul" else None
    bias_node = right if mul_node is left else left
    if mul_node is None or bias_node.op != "const":
        return None
    mul_left = by_id[mul_node.inputs[0]]
    mul_right = by_id[mul_node.inputs[1]]
    value_node = (
        mul_left if mul_right.op == "const" else mul_right if mul_left.op == "const" else None
    )
    scale_node = (
        mul_right if value_node is mul_left else mul_left if value_node is mul_right else None
    )
    if value_node is None or scale_node is None:
        return None
    scale = _scalar_constant(scale_node)
    bias = _scalar_constant(bias_node)
    if scale is None or bias is None:
        return None
    return GraphNode(
        node.id,
        "fused_mul_add",
        inputs=[value_node.id],
        attrs={"scale": scale, "bias": bias},
        shape=node.shape,
        dtype=node.dtype,
        name=node.name,
    )


def _scalar_constant(node: GraphNode) -> float | None:
    value = node.attrs.get("value")
    if isinstance(value, (bool, int, float)):
        return float(value)
    return None


def _memory_plan(graph: Graph) -> list[JsonDict]:
    last_use: dict[str, int] = {}
    for index, node in enumerate(graph.nodes):
        for input_id in node.inputs:
            last_use[input_id] = index
    return [
        {
            "id": node.id,
            "op": node.op,
            "shape": list(node.shape),
            "dtype": node.dtype,
            "estimated_bytes": _estimated_bytes(node.shape, node.dtype),
            "last_use_index": last_use.get(node.id),
        }
        for node in graph.nodes
    ]


def _estimated_bytes(shape: ShapeLike, dtype: str) -> int:
    sizes = {"float32": 4, "float64": 8, "int32": 4, "int64": 8, "bool": 1}
    return _numel(shape) * sizes.get(dtype, 4)


def _pack_outputs(outputs: list[Tensor]) -> Tensor | tuple[Tensor, ...]:
    if len(outputs) == 1:
        return outputs[0]
    return tuple(outputs)


def _normalize_shape_args(shape: Sequence[int | Sequence[int]]) -> ShapeLike:
    if len(shape) == 1 and not isinstance(shape[0], int):
        return tuple(int(dim) for dim in shape[0])
    dims: list[int] = []
    for dim in shape:
        if not isinstance(dim, int):
            raise TypeError("graph shape dimensions must be integers")
        dims.append(int(dim))
    return tuple(dims)


def _infer_binary_shape(left: ShapeLike, right: ShapeLike, op: str) -> ShapeLike:
    if op == "matmul":
        if len(left) != 2 or len(right) != 2:
            raise ValueError("graph matmul currently supports rank-2 tensors")
        return (left[0], right[1])
    return _broadcast_shape(left, right)


def _infer_binary_dtype(left: str, right: str, op: str) -> str:
    if left == right:
        return left
    if op == "div":
        return "float64" if "float64" in {left, right} else "float32"
    if "float64" in {left, right}:
        return "float64"
    if "float32" in {left, right}:
        return "float32"
    if "int64" in {left, right}:
        return "int64"
    return left


def _broadcast_shape(left: ShapeLike, right: ShapeLike) -> ShapeLike:
    result: list[int] = []
    for left_dim, right_dim in zip(reversed(left), reversed(right), strict=False):
        if left_dim == 1:
            result.append(right_dim)
        elif right_dim == 1 or left_dim == right_dim:
            result.append(left_dim)
        else:
            raise ValueError(f"cannot broadcast graph shapes {left} and {right}")
    longer = left if len(left) > len(right) else right
    prefix = list(longer[: abs(len(left) - len(right))])
    return tuple(prefix + list(reversed(result)))


def _numel(shape: ShapeLike) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


__all__ = [
    "ExecutableGraph",
    "Graph",
    "GraphNode",
    "GraphTensor",
    "TensorSpec",
    "compile_graph",
    "load_graph",
    "save_graph",
    "trace",
]
