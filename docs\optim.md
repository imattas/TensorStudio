# Optimizers

`tensorstudio.optim` contains small Python-level optimizers for TensorStudio
parameters.

## Common Contract

Optimizers accept an iterable of tensors, usually `model.parameters()`.

```python
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

Every optimizer supports:

- `zero_grad()`
- `step()`
- `state_dict()`
- `load_state_dict()`

Gradients are read from each parameter's `grad` field. Parameters with no
gradient are skipped.

## Gradient Clipping

Clip gradients before `step()` when a training loop can produce very large
updates:

```python
total_norm = optim.clip_grad_norm_(model.parameters(), max_norm=1.0)
optim.clip_grad_value_(model.parameters(), clip_value=0.5)
```

`clip_grad_norm_` returns the original global L2 norm. Both helpers mutate
gradient tensors in place.

## SGD

```python
optimizer = optim.SGD(
    model.parameters(),
    lr=0.05,
    momentum=0.9,
    weight_decay=1e-4,
)
```

SGD supports learning rate, optional momentum, and coupled weight decay.

## Adam

```python
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.0,
)
```

Adam uses first and second moment estimates with bias correction. Weight decay is
coupled into the gradient.

## AdamW

```python
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-2)
```

AdamW applies decoupled weight decay.

## Learning-Rate Schedulers

TensorStudio includes minimal schedulers that mutate `optimizer.lr`:

```python
scheduler = optim.StepLR(optimizer, step_size=10, gamma=0.5)
for _ in range(100):
    optimizer.zero_grad()
    loss = loss_fn(model(x), y)
    loss.backward()
    optimizer.step()
    scheduler.step()
```

Available schedulers:

- `optim.StepLR(optimizer, step_size, gamma=0.1)`
- `optim.ExponentialLR(optimizer, gamma)`

Schedulers support `state_dict()` and `load_state_dict()`.

## State

Optimizer state dictionaries contain plain Python numbers and TensorStudio
tensors:

```python
state = optimizer.state_dict()
optimizer.load_state_dict(state)
```

The state format is intended for TensorStudio's own save/load path and may
evolve across major versions.

## Training Pattern

```python
for batch_x, batch_y in loader:
    optimizer.zero_grad()
    loss = loss_fn(model(batch_x), batch_y)
    loss.backward()
    optimizer.step()
```

Gradient accumulation is explicit: omit `zero_grad()` only if accumulation is
intended.
