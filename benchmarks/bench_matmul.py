from __future__ import annotations

from pathlib import Path

from benchmark_report import run_benchmarks


def main() -> None:
    print(run_benchmarks({"matmul"}, Path(__file__).with_name("results_matmul.md")))


if __name__ == "__main__":
    main()
