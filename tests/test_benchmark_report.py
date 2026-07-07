from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from benchmarks.benchmark_report import BenchmarkCase, Library, Stats, _render_report  # noqa: E402


def _factory(array: np.ndarray) -> np.ndarray:
    return array


def test_benchmark_report_includes_win_columns() -> None:
    case = BenchmarkCase(
        category="elementwise",
        operation="add",
        shape=(2,),
        build=None,
        iterations=1,
        warmups=0,
        repeats=1,
    )
    libraries = [
        Library("TensorStudio", _factory),
        Library("NumPy", _factory),
        Library("PyTorch CPU", _factory),
    ]
    results = {
        "elementwise|add|(2,)|TensorStudio": Stats(1.0, 1.0, 1.0, 1.0, 0.0),
        "elementwise|add|(2,)|NumPy": Stats(2.0, 2.0, 2.0, 2.0, 0.0),
        "elementwise|add|(2,)|PyTorch CPU": Stats(0.5, 0.5, 0.5, 0.5, 0.0),
    }

    report = _render_report([case], libraries, results)

    assert "win vs NumPy" in report
    assert "win vs PyTorch" in report
    assert "fastest library" in report
    assert "| yes | n/a | no | n/a | PyTorch CPU | win vs NumPy |" in report


def test_all_runner_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "test_all.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Run TensorStudio local release checks" in completed.stdout


def test_benchmark_all_runner_help_works() -> None:
    completed = subprocess.run(
        [sys.executable, "benchmark_all.py", "--help"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "Run every TensorStudio benchmark section" in completed.stdout
