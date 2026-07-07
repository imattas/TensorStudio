# Checkpoints

Checkpoints capture model state and optional optimizer state so a training run
can be resumed or inspected later.

## State Dictionaries

State dictionaries are the safest default for model weights:

```python
from tensorstudio.project import save_state_dict, load_state_dict

save_state_dict(model, "checkpoints/model.tsnpz")
load_state_dict(model, "checkpoints/model.tsnpz", strict=True)
```

The state-dict path uses non-pickle NPZ files for tensors and flat model state.
Prefer it when you only need model parameters.

## Full Checkpoints

Use full checkpoints when optimizer state and metadata matter:

```python
from tensorstudio.project import save_checkpoint, load_checkpoint

save_checkpoint(
    model,
    "checkpoints/latest.tsckpt",
    optimizer=optimizer,
    metadata={"epoch": 10, "loss": 0.031},
)

metadata = load_checkpoint(model, "checkpoints/latest.tsckpt", optimizer=optimizer)
```

Full checkpoints may include trusted pickle data for Python optimizer state and
metadata. Only load them from sources you trust.

## Strict Loading

`strict=True` requires every saved parameter to match a current model parameter.
Use `strict=False` only during controlled model migration.

## Versioning Checkpoints

For long experiments, use names that show progression:

```text
checkpoints/
  epoch-001.tsckpt
  epoch-010.tsckpt
  best-validation.tsckpt
  latest.tsckpt
```

Keep one `latest` checkpoint for resume workflows and one `best` checkpoint for
deployment or comparison.
