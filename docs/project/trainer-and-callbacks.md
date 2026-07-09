# Trainer And Callbacks

`tensorstudio.project.Trainer` is a transparent eager training loop. It is not a
large framework abstraction: each batch still performs the familiar sequence of
zero gradients, forward pass, loss computation, backward pass, and optimizer
step.

## Train And Validate

```python
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import DataLoader, from_arrays, train_val_split
from tensorstudio.project import Trainer

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
    epochs=20,
    validation_loader=DataLoader(val_data, batch_size=1),
)

print(history.last)
```

Training records include `loss`, optional `metric`, `epoch`, and validation
fields such as `val_loss` and `val_metric`.

## Learning-Rate Schedulers

Schedulers from `tensorstudio.optim` can be passed directly:

```python
scheduler = optim.StepLR(optimizer, step_size=10, gamma=0.5)
history = trainer.fit(loader, epochs=30, scheduler=scheduler)
```

The scheduler steps once per epoch after the training and validation records are
computed.

## Callback Context

Object callbacks may implement `on_epoch_end(context)`. The context contains:

- `model`
- `optimizer`
- `scheduler`
- `history`
- `record`
- `stop_training`

Simple function callbacks that accept an epoch record are still supported.

## Built-In Callbacks

```python
from tensorstudio.project import CSVLogger, CheckpointCallback, EarlyStopping, LrLogger

callbacks = [
    LrLogger(),
    CSVLogger("runs/linear/logs/history.csv"),
    CheckpointCallback("runs/linear/checkpoints/best.tsmodel", save_best_only=True),
    EarlyStopping(monitor="val_loss", patience=5),
]

history = trainer.fit(
    train_loader,
    epochs=100,
    validation_loader=val_loader,
    callbacks=callbacks,
)
```

`CSVLogger` writes epoch records, `LrLogger` records optimizer learning rates,
`CheckpointCallback` stores trusted full checkpoints, and `EarlyStopping`
requests training stop when a monitored value stops improving.

## Stop Behavior

`EarlyStopping` stops after `patience + 1` non-improving epochs. Callback state
is stored on the callback instance, so create a fresh callback object for each
independent training run.
