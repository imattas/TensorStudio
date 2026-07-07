# Projects

`tensorstudio.project` provides a small workflow layer for complete local model
projects. It does not hide TensorStudio's eager training loop; it packages the
common pieces needed for repeatable experiments: project folders, config
loading, deterministic seeding, validation-aware training, callbacks, metrics,
and checkpoints.

## Workspace Layout

```python
from tensorstudio.project import Project, ProjectConfig

project = Project(
    "runs/linear",
    ProjectConfig(name="linear-regression", version="0.1.0", seed=7),
)

print(project.config_path)
print(project.checkpoints_dir)
print(project.artifacts_dir)
print(project.logs_dir)
```

Creating a project writes `tensorstudio.json` and creates:

- `checkpoints/`
- `artifacts/`
- `logs/`

The config saved by `Project` is JSON and can be loaded later:

```python
loaded = Project.load("runs/linear")
print(loaded.config.name)
```

Use `load_project_config` when you want to read `.json`, `.toml`, `.yaml`, or
`.yml` project configs outside an existing project folder.

## Trainer

The trainer runs the standard eager sequence:

```text
zero_grad -> forward -> loss -> backward -> optimizer step
```

```python
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, from_arrays, train_val_split
from tensorstudio.project import CSVLogger, LrLogger, Trainer, seed_everything

seed_everything(7)
dataset = from_arrays(
    [[0.0], [1.0], [2.0], [3.0]],
    [[1.0], [3.0], [5.0], [7.0]],
)
train_data, val_data = train_val_split(dataset, val_fraction=0.25, seed=7)

model = nn.Linear(1, 1)
optimizer = optim.SGD(model.parameters(), lr=0.05)
trainer = Trainer(model, optimizer, nn.MSELoss(), metric_fn=ts.metrics.mean_squared_error)
history = trainer.fit(
    DataLoader(train_data, batch_size=2),
    epochs=50,
    validation_loader=DataLoader(val_data, batch_size=1),
    callbacks=[LrLogger(), CSVLogger("runs/linear/logs/history.csv")],
)

print(history.last)
print(trainer.evaluate(DataLoader(val_data, batch_size=1)))
```

`History.records` stores one dictionary per epoch. Validation keys are prefixed
with `val_`. Simple function callbacks receive an epoch record, and object
callbacks can inspect the richer trainer context:

```python
def log(record: dict[str, float]) -> None:
    print(record["epoch"], record["loss"])

trainer.fit(loader, epochs=3, callbacks=[log])
```

For built-in callbacks, see [Trainer And Callbacks](trainer-and-callbacks.md).

## Checkpoints

For portable model weights, prefer NPZ state dictionaries:

```python
from tensorstudio.project import load_state_dict, save_state_dict

path = project.checkpoint_path("weights")
save_state_dict(model, path)
load_state_dict(model, path)
```

For trusted internal training resumes, full checkpoints can include optimizer
state and metadata:

```python
from tensorstudio.project import load_checkpoint, resume_checkpoint, save_checkpoint

save_checkpoint(model, "checkpoint.tsmodel", optimizer=optimizer, epoch=10)
checkpoint = load_checkpoint(model, "checkpoint.tsmodel", optimizer=optimizer)
next_epoch = resume_checkpoint(model, "checkpoint.tsmodel", optimizer=optimizer)
```

Full checkpoints use pickle through `tensorstudio.save` and must only be loaded
from trusted sources. NPZ weight files do not use pickle.

## Scope

The project utilities are intentionally lightweight:

- No experiment database.
- No multiprocessing launcher.
- No distributed training.
- No automatic mixed precision.

They are meant to make small projects and examples cleaner while keeping the
framework easy to inspect.
