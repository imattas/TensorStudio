from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.benchmark_report import (
    evaluate_thresholds,
    load_thresholds,
    render_benchmark_report,
    run_benchmark_data,
    write_benchmark_report,
)

ALL_SECTIONS = {
    "elementwise",
    "matmul",
    "conv2d",
    "pooling",
    "reductions",
    "activations",
    "autograd",
    "training_loop",
    "backends",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run every TensorStudio benchmark section.")
    parser.add_argument(
        "--section",
        action="append",
        choices=sorted(ALL_SECTIONS),
        help="Run only the selected section. Repeat for multiple sections.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks") / "results.md",
        help="Markdown report path. Defaults to benchmarks/results.md.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print the full Markdown report after writing it.",
    )
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=Path("benchmarks") / "thresholds.json",
        help="Benchmark regression threshold file.",
    )
    parser.add_argument(
        "--check-thresholds",
        action="store_true",
        help="Fail if TensorStudio median benchmark times exceed thresholds.",
    )
    args = parser.parse_args()

    selected_sections = set(args.section or ALL_SECTIONS)
    cases, libraries, results = run_benchmark_data(selected_sections)
    report = render_benchmark_report(cases, libraries, results)
    write_benchmark_report(report, args.output)
    if args.print:
        print(report)
    else:
        print(f"Wrote benchmark report to {args.output}")
    if args.check_thresholds:
        thresholds = load_thresholds(args.thresholds)
        failures = evaluate_thresholds(thresholds, results)
        if failures:
            print("Benchmark threshold failures:")
            for failure in failures:
                print(f"- {failure}")
            raise SystemExit(1)
        print(f"Benchmark thresholds passed using {args.thresholds}")


if __name__ == "__main__":
    main()
