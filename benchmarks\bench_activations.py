from __future__ import annotations

from pathlib import Path

from benchmark_report import run_benchmarks


def main() -> None:
    print(run_benchmarks({"activations"}, Path(__file__).with_name("results_activations.md")))


if __name__ == "__main__":
    main()
