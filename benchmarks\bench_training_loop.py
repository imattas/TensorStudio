from __future__ import annotations

from pathlib import Path

from benchmark_report import run_benchmarks


def main() -> None:
    print(run_benchmarks({"training_loop"}, Path(__file__).with_name("results_training_loop.md")))


if __name__ == "__main__":
    main()
