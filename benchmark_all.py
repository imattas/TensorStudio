from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.benchmark_report import run_benchmarks

ALL_SECTIONS = {
    "elementwise",
    "matmul",
    "reductions",
    "activations",
    "autograd",
    "training_loop",
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
    args = parser.parse_args()

    selected_sections = set(args.section or ALL_SECTIONS)
    report = run_benchmarks(selected_sections, args.output)
    if args.print:
        print(report)
    else:
        print(f"Wrote benchmark report to {args.output}")


if __name__ == "__main__":
    main()
