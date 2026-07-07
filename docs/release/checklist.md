# Release Checklist

Use this checklist before publishing a TensorStudio release.

## Local Verification

```bash
python -m pip install -e ".[dev]"
python test_all.py --quiet
python benchmark_all.py
python -m build
```

Confirm:

- Tests pass.
- Examples run.
- Docs build.
- Wheel and sdist build.
- `twine check` passes.
- Benchmark report is refreshed when performance code changed.

## Repository Checks

- `CHANGELOG.md` has a dated release entry.
- Version numbers match.
- No `.env` or local secrets are tracked.
- Generated `build/`, `dist/`, `site/`, and cache folders are not committed.
- Docs navigation includes new pages.
- GitHub Actions workflows are expected to run for the release trigger.

## Publish Flow

1. Commit the release.
2. Push `main`.
3. Push the tag.
4. Create the GitHub release.
5. Watch CI, wheel, and publish workflows.
6. Verify PyPI shows the new version.
7. Install the PyPI wheel in a clean environment.

## After Publish

Check:

- `python -m pip index versions tensorstudio --no-cache-dir`
- Fresh install import smoke test.
- GitHub release notes link to the correct tag.
- Any exposed tokens have been rotated.
