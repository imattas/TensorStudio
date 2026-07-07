from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parents[1]

SMOKE = (
    "import tensorstudio as ts; "
    "import tensorstudio._C; "
    "x = ts.tensor([1.0, 2.0], requires_grad=True); "
    "y = (x * x).sum(); "
    "y.backward(); "
    "assert ts.__version__ == {version!r}; "
    "assert x.grad.tolist() == [2.0, 4.0]; "
    "print(ts.__version__, x.grad.tolist())"
)


def _die(message: str) -> NoReturn:
    print(f"FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def _run(command: list[str], *, cwd: Path | None = None) -> None:
    print(f"==> {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd or ROOT, check=True)


def _venv_python(venv: Path) -> Path:
    if sys.platform == "win32":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def _project_version() -> str:
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, flags=re.MULTILINE)
    if match is None:
        _die("could not read project version from pyproject.toml")
    return match.group(1)


def _find_sdist(path: Path | None, directory: Path | None) -> Path | None:
    if path is not None:
        return path
    if directory is None:
        return None
    matches = sorted(directory.glob("tensorstudio-*.tar.gz"))
    if not matches:
        _die(f"no TensorStudio sdist found in {directory}")
    if len(matches) > 1:
        _die(f"multiple TensorStudio sdists found in {directory}; pass --sdist explicitly")
    return matches[0]


def _verify_wheel_install(wheel_dir: Path, version: str) -> None:
    if not wheel_dir.exists():
        _die(f"wheel directory does not exist: {wheel_dir}")
    if not any(wheel_dir.glob("tensorstudio-*.whl")):
        _die(f"no TensorStudio wheel found in {wheel_dir}")
    with tempfile.TemporaryDirectory(prefix="tensorstudio-wheel-smoke-") as tmp:
        venv = Path(tmp) / "venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        python = _venv_python(venv)
        _run([str(python), "-m", "pip", "install", "-U", "pip"])
        _run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--find-links",
                str(wheel_dir),
                f"tensorstudio=={version}",
            ]
        )
        _run([str(python), "-c", SMOKE.format(version=version)])


def _verify_sdist_install(sdist: Path, version: str) -> None:
    if not sdist.exists():
        _die(f"sdist does not exist: {sdist}")
    with tempfile.TemporaryDirectory(prefix="tensorstudio-sdist-smoke-") as tmp:
        venv = Path(tmp) / "venv"
        _run([sys.executable, "-m", "venv", str(venv)])
        python = _venv_python(venv)
        _run([str(python), "-m", "pip", "install", "-U", "pip"])
        _run([str(python), "-m", "pip", "install", str(sdist)])
        _run([str(python), "-c", SMOKE.format(version=version)])


def _clean_local_shadowing() -> None:
    for name in ("tensorstudio", "tensorstudio.egg-info"):
        candidate = ROOT / name
        if candidate.exists():
            _die(f"local path would shadow installed package: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify TensorStudio wheel/sdist artifacts through clean installs."
    )
    parser.add_argument("--version", default=_project_version(), help="Expected package version.")
    parser.add_argument("--wheel-dir", type=Path, help="Directory containing built wheels.")
    parser.add_argument("--sdist", type=Path, help="Path to a built sdist tarball.")
    parser.add_argument("--sdist-dir", type=Path, help="Directory containing one built sdist.")
    parser.add_argument("--skip-wheel", action="store_true", help="Skip wheel install smoke.")
    parser.add_argument("--skip-sdist", action="store_true", help="Skip sdist install smoke.")
    args = parser.parse_args()

    _clean_local_shadowing()

    if not args.skip_wheel:
        wheel_dir = args.wheel_dir or (ROOT / "dist")
        _verify_wheel_install(wheel_dir.resolve(), args.version)
    if not args.skip_sdist:
        sdist = _find_sdist(args.sdist, args.sdist_dir or (ROOT / "dist"))
        if sdist is None:
            _die("no sdist path provided")
        _verify_sdist_install(sdist.resolve(), args.version)

    print("Artifact verification passed.")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        if exc.cmd and shutil.which(str(exc.cmd[0])) is None:
            _die(f"command not found: {exc.cmd[0]}")
        raise
