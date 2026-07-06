from __future__ import annotations

import argparse
import importlib
import os
import platform
import statistics
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorstudio as ts

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")


ArrayFactory = Callable[[np.ndarray], Any]
Operation = Callable[[Any, Any | None], Any]

VECTOR_SHAPES = [(1,), (8,), (32,), (128,), (1024,), (4096,), (16384,)]
MATRIX_SHAPES = [(16, 16), (64, 64), (128, 128), (256, 256)]
DTYPE = np.float32


@dataclass(frozen=True)
class Library:
    name: str
    factory: ArrayFactory
    available: bool = True
    note: str = ""


@dataclass(frozen=True)
class BenchmarkCase:
    category: str
    operation: str
    shape: tuple[int, ...]
    build: Callable[[Library, np.ndarray, np.ndarray], Callable[[], Any]] | None
    iterations: int
    warmups: int
    repeats: int


@dataclass(frozen=True)
class Stats:
    median_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float
    std_ms: float


def _optional_import(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except Exception:
        return None


def _shape_label(shape: tuple[int, ...]) -> str:
    return "(" + ", ".join(str(dim) for dim in shape) + ("," if len(shape) == 1 else "") + ")"


def _numel(shape: tuple[int, ...]) -> int:
    result = 1
    for dim in shape:
        result *= dim
    return result


def _iterations(shape: tuple[int, ...], category: str) -> int:
    n = _numel(shape)
    if category == "matmul":
        if n <= 256:
            return 100
        if n <= 4096:
            return 20
        if n <= 16384:
            return 5
        return 2
    if category in {"autograd", "training_loop"}:
        if n <= 128:
            return 100
        if n <= 4096:
            return 30
        return 10
    if n <= 128:
        return 2_000
    if n <= 4096:
        return 500
    return 100


def _time_function(fn: Callable[[], Any], iterations: int, warmups: int, repeats: int) -> Stats:
    for _ in range(warmups):
        fn()

    samples: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            fn()
        elapsed = time.perf_counter() - start
        samples.append((elapsed / iterations) * 1_000.0)

    return Stats(
        median_ms=statistics.median(samples),
        mean_ms=statistics.fmean(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        std_ms=statistics.pstdev(samples) if len(samples) > 1 else 0.0,
    )


def _tensorstudio_factory(array: np.ndarray) -> ts.Tensor:
    return ts.from_numpy(array.astype(DTYPE, copy=False))


def _numpy_factory(array: np.ndarray) -> np.ndarray:
    return array.astype(DTYPE, copy=True)


def _load_libraries() -> list[Library]:
    libraries = [
        Library("TensorStudio", _tensorstudio_factory),
        Library("NumPy", _numpy_factory),
    ]

    tf = _optional_import("tensorflow")
    if tf is not None:
        with suppress(Exception):
            tf.config.set_visible_devices([], "GPU")
        libraries.append(Library("TensorFlow CPU eager", lambda array: tf.constant(array)))
    else:
        libraries.append(Library("TensorFlow CPU eager", _numpy_factory, False, "not installed"))

    torch = _optional_import("torch")
    if torch is not None:
        libraries.append(Library("PyTorch CPU", lambda array: torch.from_numpy(array.copy())))
    else:
        libraries.append(Library("PyTorch CPU", _numpy_factory, False, "not installed"))

    jnp = _optional_import("jax.numpy")
    if jnp is not None:
        libraries.append(Library("JAX CPU dispatch", lambda array: jnp.array(array)))
    else:
        libraries.append(Library("JAX CPU dispatch", _numpy_factory, False, "not installed"))

    return libraries


def _finalize(value: Any) -> Any:
    if hasattr(value, "block_until_ready"):
        return value.block_until_ready()
    if hasattr(value, "numpy") and value.__class__.__module__.startswith("tensorflow"):
        return value.numpy()
    return value


def _binary_op(name: str) -> Callable[[Library, np.ndarray, np.ndarray], Callable[[], Any]]:
    def build(library: Library, left: np.ndarray, right: np.ndarray) -> Callable[[], Any]:
        x = library.factory(left)
        y = library.factory(right)

        def run() -> Any:
            if name == "add":
                return _finalize(x + y)
            if name == "sub":
                return _finalize(x - y)
            if name == "mul":
                return _finalize(x * y)
            if name == "div":
                return _finalize(x / y)
            raise ValueError(name)

        return run

    return build


def _unary_op(name: str) -> Callable[[Library, np.ndarray, np.ndarray], Callable[[], Any]]:
    def build(library: Library, left: np.ndarray, _: np.ndarray) -> Callable[[], Any]:
        x = library.factory(left)

        def run() -> Any:
            module = x.__class__.__module__
            if library.name == "TensorStudio":
                if name == "relu":
                    return x.relu()
                if name == "sigmoid":
                    return x.sigmoid()
                if name == "tanh":
                    return x.tanh()
                if name == "exp":
                    return x.exp()
                if name == "log":
                    return x.log()
            if name == "relu":
                return _finalize(np.maximum(x, 0) if library.name == "NumPy" else _relu_external(x))
            if name == "sigmoid":
                return _finalize(1.0 / (1.0 + _exp_external(-x, module)))
            if name == "tanh":
                return _finalize(_tanh_external(x, module))
            if name == "exp":
                return _finalize(_exp_external(x, module))
            if name == "log":
                return _finalize(_log_external(x, module))
            raise ValueError(name)

        return run

    return build


def _relu_external(value: Any) -> Any:
    module = value.__class__.__module__
    if module.startswith("torch"):
        torch = importlib.import_module("torch")
        return torch.relu(value)
    if module.startswith("tensorflow"):
        tf = importlib.import_module("tensorflow")
        return tf.nn.relu(value)
    jnp = _optional_import("jax.numpy")
    if jnp is not None and module.startswith("jax"):
        return jnp.maximum(value, 0)
    return np.maximum(value, 0)


def _exp_external(value: Any, module: str) -> Any:
    if module.startswith("torch"):
        torch = importlib.import_module("torch")
        return torch.exp(value)
    if module.startswith("tensorflow"):
        tf = importlib.import_module("tensorflow")
        return tf.exp(value)
    if module.startswith("jax"):
        jnp = importlib.import_module("jax.numpy")
        return jnp.exp(value)
    return np.exp(value)


def _tanh_external(value: Any, module: str) -> Any:
    if module.startswith("torch"):
        torch = importlib.import_module("torch")
        return torch.tanh(value)
    if module.startswith("tensorflow"):
        tf = importlib.import_module("tensorflow")
        return tf.math.tanh(value)
    if module.startswith("jax"):
        jnp = importlib.import_module("jax.numpy")
        return jnp.tanh(value)
    return np.tanh(value)


def _log_external(value: Any, module: str) -> Any:
    if module.startswith("torch"):
        torch = importlib.import_module("torch")
        return torch.log(value)
    if module.startswith("tensorflow"):
        tf = importlib.import_module("tensorflow")
        return tf.math.log(value)
    if module.startswith("jax"):
        jnp = importlib.import_module("jax.numpy")
        return jnp.log(value)
    return np.log(value)


def _reduction_op(name: str) -> Callable[[Library, np.ndarray, np.ndarray], Callable[[], Any]]:
    def build(library: Library, left: np.ndarray, _: np.ndarray) -> Callable[[], Any]:
        x = library.factory(left)

        def run() -> Any:
            if library.name == "TensorStudio":
                return x.sum() if name == "sum" else x.mean()
            return _finalize(x.sum() if name == "sum" else x.mean())

        return run

    return build


def _matmul_build(library: Library, left: np.ndarray, right: np.ndarray) -> Callable[[], Any]:
    x = library.factory(left)
    y = library.factory(right)

    def run() -> Any:
        return _finalize(x @ y)

    return run


def _chain_build(library: Library, left: np.ndarray, _: np.ndarray) -> Callable[[], Any]:
    x = library.factory(left)

    def run() -> Any:
        value = (x * 2.0 + 1.0) / 3.0
        if library.name == "TensorStudio":
            return value.relu()
        return _finalize(np.maximum(value, 0) if library.name == "NumPy" else _relu_external(value))

    return run


def _autograd_scalar_build(library: Library, left: np.ndarray, _: np.ndarray) -> Callable[[], Any]:
    if library.name == "TensorStudio":
        base = left.astype(DTYPE, copy=True)

        def run_ts() -> Any:
            x = ts.tensor(base.tolist(), dtype="float32", requires_grad=True)
            loss = (x * x).sum()
            loss.backward()
            return x.grad

        return run_ts

    if library.name == "PyTorch CPU" and library.available:
        torch = importlib.import_module("torch")
        base_torch = torch.from_numpy(left.astype(DTYPE, copy=True))

        def run_torch() -> Any:
            x = base_torch.clone().requires_grad_(True)
            loss = (x * x).sum()
            loss.backward()
            return x.grad

        return run_torch

    if library.name == "TensorFlow CPU eager" and library.available:
        tf = importlib.import_module("tensorflow")
        base_tf = tf.constant(left.astype(DTYPE, copy=True))

        def run_tf() -> Any:
            with tf.GradientTape() as tape:
                tape.watch(base_tf)
                loss = tf.reduce_sum(base_tf * base_tf)
            return tape.gradient(loss, base_tf)

        return run_tf

    raise NotImplementedError


def _autograd_elementwise_build(
    library: Library,
    left: np.ndarray,
    _: np.ndarray,
) -> Callable[[], Any]:
    if library.name == "TensorStudio":
        base = left.astype(DTYPE, copy=True)

        def run_ts() -> Any:
            x = ts.tensor(base.tolist(), dtype="float32", requires_grad=True)
            loss = ((x * 2.0 + 1.0) / 3.0).relu().mean()
            loss.backward()
            return x.grad

        return run_ts

    if library.name == "PyTorch CPU" and library.available:
        torch = importlib.import_module("torch")
        base_torch = torch.from_numpy(left.astype(DTYPE, copy=True))

        def run_torch() -> Any:
            x = base_torch.clone().requires_grad_(True)
            loss = torch.relu((x * 2.0 + 1.0) / 3.0).mean()
            loss.backward()
            return x.grad

        return run_torch

    if library.name == "TensorFlow CPU eager" and library.available:
        tf = importlib.import_module("tensorflow")
        base_tf = tf.constant(left.astype(DTYPE, copy=True))

        def run_tf() -> Any:
            with tf.GradientTape() as tape:
                tape.watch(base_tf)
                loss = tf.reduce_mean(tf.nn.relu((base_tf * 2.0 + 1.0) / 3.0))
            return tape.gradient(loss, base_tf)

        return run_tf

    raise NotImplementedError


def _training_loop_build(
    library: Library,
    left: np.ndarray,
    right: np.ndarray,
) -> Callable[[], Any]:
    x_data = left.reshape((-1, 1)).astype(DTYPE, copy=True)
    y_data = right.reshape((-1, 1)).astype(DTYPE, copy=True)

    if library.name == "TensorStudio":
        x_ts = ts.from_numpy(x_data)
        y_ts = ts.from_numpy(y_data)

        def run_ts() -> Any:
            ts.manual_seed(0)
            model = ts.nn.Linear(1, 1)
            loss_fn = ts.nn.MSELoss()
            optimizer = ts.optim.SGD(model.parameters(), lr=0.01)
            for _ in range(10):
                optimizer.zero_grad()
                loss = loss_fn(model(x_ts), y_ts)
                loss.backward()
                optimizer.step()
            return loss

        return run_ts

    if library.name == "NumPy":
        x_np = x_data
        y_np = y_data

        def run_np() -> Any:
            weight = np.array([[0.0]], dtype=DTYPE)
            bias = np.array([0.0], dtype=DTYPE)
            for _ in range(10):
                pred = x_np @ weight + bias
                diff = pred - y_np
                grad_w = (2.0 / x_np.shape[0]) * (x_np.T @ diff)
                grad_b = (2.0 / x_np.shape[0]) * diff.sum(axis=0)
                weight -= 0.01 * grad_w
                bias -= 0.01 * grad_b
            return weight, bias

        return run_np

    raise NotImplementedError


def _build_cases(sections: set[str]) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []

    def add_case(
        category: str,
        operation: str,
        shape: tuple[int, ...],
        build: Callable[[Library, np.ndarray, np.ndarray], Callable[[], Any]],
    ) -> None:
        if category in sections:
            cases.append(
                BenchmarkCase(
                    category=category,
                    operation=operation,
                    shape=shape,
                    build=build,
                    iterations=_iterations(shape, category),
                    warmups=3,
                    repeats=5,
                )
            )

    for shape in VECTOR_SHAPES:
        for op in ["add", "sub", "mul", "div"]:
            add_case("elementwise", op, shape, _binary_op(op))
        add_case("elementwise", "chain_relu((x*2+1)/3)", shape, _chain_build)

    for shape in VECTOR_SHAPES:
        for op in ["relu", "sigmoid", "tanh", "exp", "log"]:
            add_case("activations", op, shape, _unary_op(op))

    for shape in VECTOR_SHAPES:
        for op in ["sum", "mean"]:
            add_case("reductions", op, shape, _reduction_op(op))

    for shape in MATRIX_SHAPES:
        add_case("matmul", "matmul", shape, _matmul_build)

    for shape in [(1,), (128,), (1024,)]:
        add_case("autograd", "scalar_backward", shape, _autograd_scalar_build)
        add_case("autograd", "elementwise_backward", shape, _autograd_elementwise_build)

    if "training_loop" in sections:
        cases.append(
            BenchmarkCase(
                category="training_loop",
                operation="tiny_linear_regression_10_steps",
                shape=(32,),
                build=_training_loop_build,
                iterations=10,
                warmups=1,
                repeats=5,
            )
        )

    return cases


def _case_data(shape: tuple[int, ...], operation: str) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(123)
    left = rng.standard_normal(shape).astype(DTYPE)
    right = rng.standard_normal(shape).astype(DTYPE)
    if operation in {"div", "log"}:
        left = np.abs(left) + 0.25
        right = np.abs(right) + 0.25
    if operation == "tiny_linear_regression_10_steps":
        left = np.linspace(0.0, 1.0, shape[0], dtype=DTYPE)
        right = 2.0 * left + 1.0
    return left, right


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def _speedup(ts_ms: float | None, competitor_ms: float | None) -> float | None:
    if ts_ms is None or competitor_ms is None or ts_ms <= 0:
        return None
    return competitor_ms / ts_ms


def _run_cases(cases: Iterable[BenchmarkCase], libraries: list[Library]) -> dict[str, Stats]:
    results: dict[str, Stats] = {}
    for case in cases:
        left, right = _case_data(case.shape, case.operation)
        for library in libraries:
            key = _result_key(case, library.name)
            if not library.available or case.build is None:
                continue
            try:
                fn = case.build(library, left, right)
            except NotImplementedError:
                continue
            results[key] = _time_function(fn, case.iterations, case.warmups, case.repeats)
    return results


def _result_key(case: BenchmarkCase, library_name: str) -> str:
    return f"{case.category}|{case.operation}|{_shape_label(case.shape)}|{library_name}"


def _render_report(
    cases: list[BenchmarkCase],
    libraries: list[Library],
    results: dict[str, Stats],
) -> str:
    lines: list[str] = []
    lines.append("# TensorStudio Benchmark Results")
    lines.append("")
    lines.append("These benchmarks are local references, not marketing claims.")
    lines.append("TensorStudio is compared on CPU only and uses the same shapes and dtype.")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- Platform: `{platform.platform()}`")
    lines.append(f"- Processor: `{platform.processor() or 'unknown'}`")
    lines.append(f"- Python: `{sys.version.split()[0]}`")
    lines.append(f"- TensorStudio: `{ts.__version__}`")
    lines.append(f"- NumPy: `{np.__version__}`")
    for library in libraries:
        if library.name in {"TensorStudio", "NumPy"}:
            continue
        status = "available" if library.available else f"unavailable ({library.note})"
        lines.append(f"- {library.name}: {status}")
    lines.append("")
    lines.append("## Summary")
    lines.append("")

    wins = 0
    losses = 0
    for case in cases:
        ts_stats = results.get(_result_key(case, "TensorStudio"))
        np_stats = results.get(_result_key(case, "NumPy"))
        ratio = _speedup(
            ts_stats.median_ms if ts_stats else None,
            np_stats.median_ms if np_stats else None,
        )
        if ratio is None:
            continue
        if ratio > 1.0:
            wins += 1
        else:
            losses += 1
    lines.append(f"- TensorStudio wins versus NumPy: `{wins}`")
    lines.append(f"- TensorStudio losses versus NumPy: `{losses}`")
    lines.append("")
    if wins == 0:
        lines.append(
            "TensorStudio did not beat NumPy on this machine for the available benchmark set. "
            "Performance remains a blocker for a final `1.0.0` performance claim."
        )
    else:
        lines.append(
            "TensorStudio beat NumPy on at least one local benchmark case. "
            "See the detailed table before making any narrow performance claim."
        )
    lines.append("")
    lines.append("## Detailed Results")
    lines.append("")
    lines.append(
        "| category | operation | shape | library | median ms | mean ms | min ms | "
        "max ms | std ms | TS vs NumPy | TS vs TensorFlow | TS vs PyTorch | TS vs JAX | result |"
    )
    lines.append(
        "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    )

    for case in cases:
        ts_stats = results.get(_result_key(case, "TensorStudio"))
        medians = {
            library.name: results.get(_result_key(case, library.name)).median_ms
            for library in libraries
            if results.get(_result_key(case, library.name)) is not None
        }
        speedups = {
            "NumPy": _speedup(ts_stats.median_ms if ts_stats else None, medians.get("NumPy")),
            "TensorFlow CPU eager": _speedup(
                ts_stats.median_ms if ts_stats else None,
                medians.get("TensorFlow CPU eager"),
            ),
            "PyTorch CPU": _speedup(
                ts_stats.median_ms if ts_stats else None,
                medians.get("PyTorch CPU"),
            ),
            "JAX CPU dispatch": _speedup(
                ts_stats.median_ms if ts_stats else None,
                medians.get("JAX CPU dispatch"),
            ),
        }
        for library in libraries:
            stats = results.get(_result_key(case, library.name))
            if stats is None:
                continue
            result = "reference"
            if library.name == "TensorStudio":
                ratio = speedups["NumPy"]
                if ratio is None:
                    result = "no NumPy reference"
                elif ratio > 1.0:
                    result = "win vs NumPy"
                else:
                    result = "loss vs NumPy"
            elif library.name == "NumPy":
                result = "NumPy baseline"
            lines.append(
                "| "
                f"{case.category} | {case.operation} | `{_shape_label(case.shape)}` | "
                f"{library.name} | {_format_float(stats.median_ms)} | "
                f"{_format_float(stats.mean_ms)} | {_format_float(stats.min_ms)} | "
                f"{_format_float(stats.max_ms)} | {_format_float(stats.std_ms)} | "
                f"{_format_float(speedups['NumPy'])} | "
                f"{_format_float(speedups['TensorFlow CPU eager'])} | "
                f"{_format_float(speedups['PyTorch CPU'])} | "
                f"{_format_float(speedups['JAX CPU dispatch'])} | {result} |"
            )
    lines.append("")
    lines.append(
        "Speedup columns are `competitor median / TensorStudio median`; values above 1.0 mean"
    )
    lines.append("TensorStudio was faster for that specific case.")
    lines.append("")
    return "\n".join(lines)


def run_benchmarks(sections: set[str], output: Path | None) -> str:
    libraries = _load_libraries()
    cases = _build_cases(sections)
    results = _run_cases(cases, libraries)
    report = _render_report(cases, libraries, results)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TensorStudio CPU benchmarks.")
    parser.add_argument(
        "--section",
        action="append",
        choices=["elementwise", "matmul", "reductions", "activations", "autograd", "training_loop"],
        help="Benchmark section to run. Repeat for multiple sections. Defaults to all sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("results.md"),
        help="Markdown report output path.",
    )
    args = parser.parse_args()
    sections = set(
        args.section
        or ["elementwise", "matmul", "reductions", "activations", "autograd", "training_loop"]
    )
    report = run_benchmarks(sections, args.output)
    print(report)


if __name__ == "__main__":
    main()
