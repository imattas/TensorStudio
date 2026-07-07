from __future__ import annotations

import csv
import random

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio import nn, optim
from tensorstudio.data import (
    DataLoader,
    dataset_summary,
    from_arrays,
    from_image_folder,
    from_tensors,
    train_val_split,
)
from tensorstudio.project import (
    CheckpointCallback,
    CSVLogger,
    EarlyStopping,
    LrLogger,
    ProjectConfig,
    Trainer,
    create_project_template,
    load_config_dict,
    load_project_config,
    resume_checkpoint,
    seed_everything,
)


def test_dataset_factories_split_and_summary(tmp_path) -> None:
    features = np.arange(12, dtype=np.float32).reshape(6, 2)
    labels = np.arange(6, dtype=np.int64)
    dataset = from_arrays(features, labels)
    train_a, val_a = train_val_split(dataset, val_fraction=0.33, seed=3)
    train_b, val_b = train_val_split(dataset, val_fraction=0.33, seed=3)
    tensor_dataset = from_tensors(ts.tensor(features), ts.tensor(labels, dtype="int64"))

    assert len(dataset) == 6
    assert len(train_a) == 4
    assert len(val_a) == 2
    assert train_a.indices == train_b.indices
    assert val_a.indices == val_b.indices
    assert len(tensor_dataset) == 6
    assert dataset_summary(dataset)["sample_parts"] == 2

    pytest.importorskip("PIL.Image")
    class_dir = tmp_path / "images" / "zero"
    class_dir.mkdir(parents=True)
    ts.vision.save_image(np.zeros((3, 3, 3), dtype=np.uint8), class_dir / "sample.png")
    image_dataset = from_image_folder(tmp_path / "images")
    image_summary = dataset_summary(image_dataset)

    assert image_summary["classes"] == ["zero"]
    assert image_summary["target_classes"] == [0]


def test_metrics_package() -> None:
    prediction = ts.tensor([[0.1, 0.9], [2.0, 1.0], [0.2, 0.8]])
    target = ts.tensor([1, 0, 0], dtype="int64")
    regression_pred = ts.tensor([2.0, 4.0, 6.0])
    regression_target = ts.tensor([1.0, 5.0, 7.0])
    multilabel_pred = ts.tensor([[0.8, 0.1, 0.6], [0.2, 0.7, 0.4]])
    multilabel_target = ts.tensor([[1, 0, 1], [0, 1, 0]], dtype="int64")

    assert ts.metrics.accuracy(prediction, target) == pytest.approx(2 / 3)
    np.testing.assert_array_equal(
        ts.metrics.confusion_matrix(prediction, target, num_classes=2),
        np.array([[1, 1], [0, 1]], dtype=np.int64),
    )
    scores = ts.metrics.precision_recall_f1(prediction, target, num_classes=2)
    assert scores["precision_macro"] == pytest.approx(0.75)
    assert scores["recall_macro"] == pytest.approx(0.75)
    assert scores["f1_macro"] == pytest.approx(2 / 3)
    assert ts.metrics.mean_absolute_error(regression_pred, regression_target) == pytest.approx(1.0)
    assert ts.metrics.mean_squared_error(regression_pred, regression_target) == pytest.approx(1.0)
    assert ts.metrics.root_mean_squared_error(regression_pred, regression_target) == pytest.approx(
        1.0
    )
    assert ts.metrics.r2_score(regression_pred, regression_target) == pytest.approx(0.8392857143)
    assert ts.metrics.multilabel_accuracy(multilabel_pred, multilabel_target) == pytest.approx(1.0)
    assert ts.metrics.hamming_loss(multilabel_pred, multilabel_target) == pytest.approx(0.0)
    assert ts.metrics.multilabel_f1(multilabel_pred, multilabel_target) == pytest.approx(1.0)


def test_project_config_json_toml_yaml(tmp_path) -> None:
    json_path = tmp_path / "project.json"
    toml_path = tmp_path / "project.toml"
    yaml_path = tmp_path / "project.yaml"
    json_path.write_text('{"name": "json-demo", "seed": 7}\n', encoding="utf-8")
    toml_path.write_text('name = "toml-demo"\nseed = 11\n', encoding="utf-8")
    yaml_path.write_text("name: yaml-demo\nseed: 13\nmetadata:\n  task: vision\n", encoding="utf-8")

    assert load_project_config(json_path).name == "json-demo"
    assert load_project_config(toml_path).seed == 11
    assert load_project_config(yaml_path).metadata["task"] == "vision"
    assert load_config_dict(yaml_path)["name"] == "yaml-demo"

    config = ProjectConfig(name="roundtrip", seed=5)
    config_path = tmp_path / "saved.json"
    config.save(config_path)
    assert ProjectConfig.load(config_path).name == "roundtrip"


def test_trainer_validation_callbacks_and_checkpoint_resume(tmp_path) -> None:
    seed_everything(21)
    x = ts.tensor([[0.0], [1.0], [2.0], [3.0]])
    y = ts.tensor([[1.0], [3.0], [5.0], [7.0]])
    train_loader = DataLoader(from_tensors(x, y), batch_size=2, shuffle=False)
    val_loader = DataLoader(from_tensors(x, y), batch_size=4, shuffle=False)

    model = nn.Linear(1, 1)
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.1)
    scheduler = optim.StepLR(optimizer, step_size=2, gamma=0.5)
    trainer = Trainer(model, optimizer, nn.MSELoss(), metric_fn=ts.metrics.mean_squared_error)
    csv_path = tmp_path / "logs" / "history.csv"
    checkpoint_path = tmp_path / "checkpoints" / "best.tsmodel"
    callbacks = [
        LrLogger(),
        CSVLogger(csv_path),
        CheckpointCallback(checkpoint_path, save_best_only=True),
        EarlyStopping(monitor="val_loss", patience=10),
    ]

    history = trainer.fit(
        train_loader,
        epochs=4,
        validation_loader=val_loader,
        scheduler=scheduler,
        callbacks=callbacks,
    )

    assert history.last["epoch"] == 4.0
    assert "val_loss" in history.last
    assert "lr" in history.last
    assert csv_path.exists()
    assert checkpoint_path.exists()
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert "lr" in rows[0]

    restored = nn.Linear(1, 1)
    restored_optimizer = optim.SGD(restored.parameters(), lr=0.5, momentum=0.1)
    restored_scheduler = optim.StepLR(restored_optimizer, step_size=2, gamma=0.5)
    next_epoch = resume_checkpoint(
        restored,
        checkpoint_path,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
    )

    assert next_epoch >= 2
    assert restored_optimizer.lr < 0.5
    assert restored_scheduler.last_epoch >= 1


def test_seed_everything_and_project_templates(tmp_path) -> None:
    seed_everything(99)
    python_first = random.random()
    numpy_first = float(np.random.rand())
    tensor_first = ts.rand((2,), seed=None).tolist()
    seed_everything(99)

    assert random.random() == pytest.approx(python_first)
    assert float(np.random.rand()) == pytest.approx(numpy_first)
    assert ts.rand((2,), seed=None).tolist() == pytest.approx(tensor_first)

    project_root = create_project_template("regression", tmp_path / "regression")
    assert (project_root / "README.md").exists()
    assert (project_root / "train.py").exists()
    with pytest.raises(FileExistsError):
        create_project_template("regression", project_root)
    create_project_template("classification", tmp_path / "classification")
    create_project_template("vision", tmp_path / "vision")
