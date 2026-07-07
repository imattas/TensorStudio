# Versioning

TensorStudio uses semantic versioning for public releases.

## Version Shape

```text
MAJOR.MINOR.PATCH
```

- Patch: bug fixes, docs, packaging fixes, small compatible ops.
- Minor: new public APIs, new modules, larger compatible feature groups.
- Major: breaking API or behavior changes.

## Places To Update

Before a release, update the version consistently in:

- `pyproject.toml`
- `CMakeLists.txt`
- `include/tensorstudio/version.hpp`
- `python/tensorstudio/_version.py`
- `tests/test_import.py`
- `README.md`
- Relevant docs pages.
- `CHANGELOG.md`

## Tag Format

Tags use a leading `v`:

```bash
git tag v1.9.0
git push origin v1.9.0
```

GitHub Actions uses the tag to build wheels and release artifacts.

## Compatibility Notes

Document behavior changes even when they are compatible. Users care about
gradient formulas, dtype promotion, serialization format, and install behavior.
