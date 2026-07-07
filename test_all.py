from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def _run(command: Sequence[str], *, quiet: bool) -> int:
    display = " ".join(command)
    print(f"\n==> {display}", flush=True)
    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=quiet,
        text=quiet,
    )
    if completed.returncode != 0:
        if quiet:
            if completed.stdout:
                print(completed.stdout, end="")
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
        print(f"FAILED: {display}", file=sys.stderr)
    return completed.returncode


def _remove_inside_repo(path: Path) -> None:
    resolved = path.resolve()
    if resolved == ROOT or ROOT not in resolved.parents:
        raise RuntimeError(f"refusing to remove path outside repository: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)


def _run_build(out_dir: Path, *, quiet: bool) -> int:
    _remove_inside_repo(out_dir)
    code = _run([sys.executable, "-m", "build", "--outdir", str(out_dir)], quiet=quiet)
    if code != 0:
        return code

    artifacts = sorted(str(path) for path in out_dir.glob("*") if path.is_file())
    if not artifacts:
        print(f"FAILED: no build artifacts found in {out_dir}", file=sys.stderr)
        return 1
    return _run([sys.executable, "-m", "twine", "check", *artifacts], quiet=quiet)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TensorStudio local release checks.")
    parser.add_argument("--skip-lint", action="store_true", help="Skip Ruff checks.")
    parser.add_argument("--skip-types", action="store_true", help="Skip mypy checks.")
    parser.add_argument("--skip-tests", action="store_true", help="Skip pytest.")
    parser.add_argument("--skip-examples", action="store_true", help="Skip example scripts.")
    parser.add_argument("--skip-build", action="store_true", help="Skip sdist/wheel build checks.")
    parser.add_argument("--skip-docs", action="store_true", help="Skip MkDocs build.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress command output unless it fails.",
    )
    args = parser.parse_args()

    commands: list[list[str]] = []
    if not args.skip_lint:
        commands.append([sys.executable, "-m", "ruff", "check", "."])
    if not args.skip_types:
        commands.append([sys.executable, "-m", "mypy", "python/tensorstudio"])
    if not args.skip_tests:
        commands.append([sys.executable, "-m", "pytest", "-q"])
    if not args.skip_examples:
        commands.extend(
            [sys.executable, str(path.relative_to(ROOT))]
            for path in sorted((ROOT / "examples").glob("*.py"))
        )
    if not args.skip_docs:
        commands.append([sys.executable, "-m", "mkdocs", "build"])

    for command in commands:
        code = _run(command, quiet=args.quiet)
        if code != 0:
            raise SystemExit(code)

    if not args.skip_build:
        code = _run_build(ROOT / "build" / "test-all-dist", quiet=args.quiet)
        if code != 0:
            raise SystemExit(code)

    print("\nAll selected checks passed.")


if __name__ == "__main__":
    main()
