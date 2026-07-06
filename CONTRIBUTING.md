# Contributing

Thank you for helping improve TensorStudio.

## Development Setup

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
pre-commit install
```

## Checks

Run these before opening a pull request:

```bash
ruff check .
mypy python/tensorstudio
pytest
python -m build
```

## Guidelines

- Keep C++ tensor operations clear and maintainable.
- Add tests for user-visible behavior changes.
- Document limitations honestly.
- Do not introduce network calls in library code.
- Do not use `eval` or `exec`.
