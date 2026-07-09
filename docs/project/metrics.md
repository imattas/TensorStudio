# Metrics

`tensorstudio.metrics` contains small NumPy-backed metrics for evaluating model
outputs from TensorStudio tensors. Metrics return Python floats or NumPy arrays
and do not participate in autograd.

## Regression

```python
import tensorstudio as ts

prediction = ts.tensor([2.0, 4.0, 6.0])
target = ts.tensor([1.0, 5.0, 7.0])

print(ts.metrics.mean_absolute_error(prediction, target))
print(ts.metrics.mean_squared_error(prediction, target))
print(ts.metrics.root_mean_squared_error(prediction, target))
print(ts.metrics.r2_score(prediction, target))
```

## Classification

Classification metrics accept logits or probabilities with shape
`(batch, classes)`. The predicted label is the `argmax` over classes. A 1D
prediction is treated as binary probabilities with a threshold of `0.5`.

```python
logits = ts.tensor([[0.2, 0.8], [0.7, 0.3]])
target = ts.tensor([1, 0], dtype="int64")

print(ts.metrics.accuracy(logits, target))
print(ts.metrics.confusion_matrix(logits, target, num_classes=2))
print(ts.metrics.precision_recall_f1(logits, target, num_classes=2))
```

## Multilabel

Multilabel metrics threshold each output independently:

```python
prediction = ts.tensor([[0.8, 0.1, 0.6], [0.2, 0.7, 0.4]])
target = ts.tensor([[1, 0, 1], [0, 1, 0]], dtype="int64")

print(ts.metrics.multilabel_accuracy(prediction, target))
print(ts.metrics.hamming_loss(prediction, target))
print(ts.metrics.multilabel_f1(prediction, target))
```

## Trainer Integration

`Trainer` accepts one metric function:

```python
from tensorstudio.project import Trainer

trainer = Trainer(model, optimizer, loss_fn, metric_fn=ts.metrics.accuracy)
history = trainer.fit(train_loader, epochs=5, validation_loader=val_loader)
print(history.last["metric"], history.last["val_metric"])
```

For multiple metrics, compute them explicitly after `trainer.evaluate()` or use
a custom callback.
