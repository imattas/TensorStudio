# Publishing

TensorStudio release publishing should use GitHub Actions with PyPI Trusted
Publishing. The release workflow intentionally does not accept a PyPI API token
secret; configure PyPI's trusted publisher for the GitHub repository and
environment instead. Do not hardcode PyPI tokens in workflows, docs examples, or
source files.

## Release Candidate Flow

Use release-candidate versions while hardening:

```bash
git tag v1.0.0rc2
git push origin v1.0.0rc2
```

Stable tags should only be created after the release checklist passes on
Windows, Linux, and macOS.

## Stable Release Flow

After the release checklist passes, tag the stable release:

```bash
git tag v1.16.0
git push origin v1.16.0
```

Stable release notes must avoid unsupported performance claims. Publish measured
benchmark results, including known losses versus NumPy or PyTorch where they
exist.

## GitHub Actions

Workflows:

- `ci.yml`: lint, type check, editable install, import test, pytest, and an
  example on Windows, Linux, and macOS. It also builds wheel/sdist artifacts on
  each major platform and verifies clean installs with
  `tools/verify_artifacts.py`.
- `wheels.yml`: cibuildwheel artifacts for Windows first, Linux second, macOS
  third, plus sdist, clean wheel/sdist smoke checks, and benchmark report
  artifacts.
- `publish.yml`: artifact verification plus PyPI publishing through
  `pypa/gh-action-pypi-publish` using Trusted Publishing only.
- `publish-testpypi.yml`: the same artifact path aimed at TestPyPI for dry-run
  publishing.
- `docs.yml`: strict MkDocs build and GitHub Pages deployment from `main`.

The publish workflow must include:

```yaml
permissions:
  contents: read
  id-token: write
```

It should publish only from a GitHub Release or an explicit manual dispatch. It
must not print secrets. Tag pushes are handled by the wheel-building workflow.

## TestPyPI First

Configure a PyPI trusted publisher for TestPyPI before rehearsing a release.
Then publish the candidate or stable version to TestPyPI and install it in a
clean environment:

```bash
python -m venv .venv-testpypi
. .venv-testpypi/bin/activate
python -m pip install -U pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tensorstudio==1.16.0
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
deactivate
```

On Windows PowerShell:

```powershell
python -m venv .venv-testpypi
.\.venv-testpypi\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ tensorstudio==1.16.0
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
deactivate
```

Promote to real PyPI only after TestPyPI install and runtime checks pass.

## Artifact Verification

`tools/verify_artifacts.py` is the local and CI clean-install smoke verifier:

```bash
python tools/verify_artifacts.py --wheel-dir dist --sdist-dir dist
```

It creates temporary virtual environments, installs the built wheel or sdist,
imports `tensorstudio._C`, checks the package version, and runs a tiny autograd
smoke test. This catches missing package data, broken wheel tags, native import
errors, and source-build regressions before upload.

See [Platform Compatibility](platform-compatibility.md) for wheel tags, ABI
notes, and source-build expectations.

## Version Checklist

- `pyproject.toml`
- `python/tensorstudio/_version.py`
- `include/tensorstudio/version.hpp`
- `CHANGELOG.md`
- `README.md`
- Docs that mention version or status

`CMakeLists.txt` uses a numeric project version. Keep the public Python/C++
version strings in package metadata and the version header synchronized.

## Required Checks Before A Release

- `python test_all.py` passes locally.
- `python benchmark_all.py` regenerates current benchmark tables.
- `ruff check .` passes.
- `mypy python/tensorstudio` passes.
- `pytest -q` passes on Windows.
- `pytest -q` passes on Linux.
- `pytest -q` passes on macOS.
- `python -m build` passes.
- `python -m twine check dist/*` passes.
- Clean wheel install passes on Windows, Linux, and macOS.
- Clean sdist install passes on Windows, Linux, and macOS.
- Examples pass on all platforms.
- Docs are accurate.
- Changelog is updated.
- No TODO-only files.
- No hidden failing tests or ignored CI failures.
- No publishing tokens are committed.
- No publishing tokens are configured as workflow fallbacks.

## Post-Release Verification

```bash
python -m venv .release-check
. .release-check/bin/activate
python -m pip install -U pip
python -m pip install tensorstudio
python -c "import tensorstudio as ts; import tensorstudio._C; print(ts.__version__)"
python - <<'PY'
import tensorstudio as ts
x = ts.tensor([1.0, 2.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad.tolist())
PY
deactivate
```

Use the PowerShell activation path on Windows.
