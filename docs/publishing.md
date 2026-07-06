# Publishing

TensorStudio is configured for GitHub Actions, wheel builds, and PyPI trusted
publishing. The preferred release path is trusted publishing because it avoids
long-lived PyPI tokens in developer environments.

## Preferred Flow: Trusted Publishing

1. Configure the PyPI project to trust `imattas/TensorStudio` and the
   `.github/workflows/publish.yml` workflow.
2. Ensure the workflow has `id-token: write` permission.
3. Confirm CI is green.
4. Create and push a version tag:

   ```bash
   git tag v0.1.1
   git push origin v0.1.1
   ```

5. Publish a GitHub release, or manually run the publish workflow.
6. Verify installation in a clean virtual environment.

The workflow must not contain hardcoded PyPI tokens.

## Local Token Upload

Maintainers can upload locally with a PyPI token when necessary. Keep the token
out of source control. If using `.env`, it should contain:

```text
PYPI_TOKEN=pypi-...
```

Then:

```bash
python -m pip install -U build twine
python -m build
twine check dist/*
twine upload -u __token__ -p "$PYPI_TOKEN" dist/*
```

On PowerShell, load the token without printing it:

```powershell
$env:PYPI_TOKEN = (Get-Content .env | Where-Object { $_ -match '^PYPI_TOKEN=' }) -replace '^PYPI_TOKEN=', ''
python -m twine upload -u __token__ -p $env:PYPI_TOKEN dist/*
```

Only upload artifacts you intend to publish. PyPI files are immutable for a
given project version.

## TestPyPI Rehearsal

For rehearsal releases, configure a separate TestPyPI trusted publisher or use:

```bash
twine upload --repository testpypi -u __token__ -p "$TEST_PYPI_TOKEN" dist/*
```

Install from TestPyPI:

```bash
python -m pip install --index-url https://test.pypi.org/simple/ tensorstudio
```

## Version Bump Checklist

- Update `pyproject.toml`
- Update `python/tensorstudio/_version.py`
- Update `include/tensorstudio/version.hpp`
- Update `CHANGELOG.md`
- Update docs that mention the version
- Run lint, type checks, tests, and builds

## Artifact Checklist

- `dist/*.tar.gz` source distribution
- Platform wheels from cibuildwheel
- README renders correctly on PyPI
- License file included
- Native extension imports cleanly
- Examples run against the built wheel

## Post-Release Verification

```bash
python -m venv .release-check
. .release-check/bin/activate
python -m pip install tensorstudio
python -c "import tensorstudio as ts; print(ts.__version__)"
python - <<'PY'
import tensorstudio as ts
x = ts.tensor([1.0, 2.0], requires_grad=True)
loss = (x * x).sum()
loss.backward()
print(x.grad.tolist())
PY
```

Remove the temporary environment after verification.
