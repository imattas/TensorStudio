from __future__ import annotations

from pathlib import Path

from benchmark_report import run_benchmarks

if __name__ == "__main__":
    print(run_benchmarks({"conv2d"}, Path(__file__).with_name("results_conv2d.md")))
