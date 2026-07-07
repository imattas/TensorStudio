# Configs And Templates

TensorStudio project configs are small dictionaries intended for local runs,
examples, and reproducible experiments.

## JSON, TOML, And YAML

`load_config_dict` parses `.json`, `.toml`, `.yaml`, and `.yml` files into a
plain dictionary:

```python
from tensorstudio.project import load_config_dict

config = load_config_dict("experiment.yaml")
print(config["name"])
```

`load_project_config` validates the same files as `ProjectConfig`:

```python
from tensorstudio.project import load_project_config

project_config = load_project_config("tensorstudio.toml")
print(project_config.name, project_config.seed)
```

The validated fields are:

- `name`: required non-empty string
- `version`: optional non-empty string
- `seed`: optional integer
- `metadata`: optional mapping

`ProjectConfig.save()` writes JSON for portability.

## Deterministic Seeding

`seed_everything` seeds Python `random`, NumPy, and TensorStudio native random
creation helpers:

```python
from tensorstudio.project import seed_everything

seed_everything(7)
```

Use the same seed for `DataLoader(..., shuffle=True, seed=...)` and
`train_val_split(..., seed=...)` when you want deterministic data order as well
as deterministic random initialization.

## Project Templates

`create_project_template` writes a small runnable starter project:

```python
from tensorstudio.project import create_project_template

create_project_template("regression", "my-regression-run")
create_project_template("classification", "my-classification-run")
create_project_template("vision", "my-vision-run")
```

Available templates:

- `regression`
- `classification`
- `vision`

The generator refuses to overwrite existing files unless `overwrite=True` is
passed.

## Template Scope

Templates are intentionally compact. They create enough structure to start a
TensorStudio example quickly, but they do not generate packaging metadata,
deployment scripts, experiment databases, or cloud configuration.
