# Project Layout

TensorStudio includes small project helpers for examples and local experiments.
They are not a replacement for a full experiment platform, but they provide a
repeatable place for configs, checkpoints, and training history.

## Suggested Layout

```text
my-project/
  config.json
  checkpoints/
  data/
  runs/
  exports/
```

Use:

- `config.json` for model and training settings.
- `checkpoints/` for TensorStudio checkpoint files.
- `data/` for local prepared datasets.
- `runs/` for metrics and logs.
- `exports/` for ONNX or other interchange artifacts.

## `ProjectConfig`

```python
from tensorstudio.project import ProjectConfig

config = ProjectConfig(
    name="linear-regression",
    version="0.1.0",
    seed=123,
    metadata={"owner": "local"},
)
```

Configuration should be serializable and boring. Keep live Python objects such
as models, optimizers, and datasets outside metadata.

## Reproducibility

Record:

- TensorStudio version.
- Python version.
- Dataset source and preprocessing.
- Random seed.
- Model class and hyperparameters.
- Optimizer settings.

This information makes bug reports and benchmark comparisons much easier to
reproduce.
