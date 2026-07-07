from __future__ import annotations

from pathlib import Path

from benchmark_report import run_benchmarks


def main() -> None:
    sections = {"elementwise", "activations", "reductions"}
    print(run_benchmarks(sections, Path(__file__).with_name("results_tensor_ops.md")))


if __name__ == "__main__":
    main()
