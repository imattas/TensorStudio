from __future__ import annotations

import numpy as np
import pytest
import tensorstudio as ts


def test_to_tensor_scales_hwc_uint8_to_chw_float() -> None:
    image = np.array(
        [
            [[0, 128, 255], [255, 0, 128]],
            [[32, 64, 96], [16, 8, 4]],
        ],
        dtype=np.uint8,
    )

    result = ts.vision.to_tensor(image)

    assert result.shape == (3, 2, 2)
    assert result.dtype == "float32"
    np.testing.assert_allclose(result.numpy(), np.transpose(image, (2, 0, 1)) / 255.0)


def test_vision_normalize_crop_and_resize() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    cropped = ts.vision.center_crop(image, (2, 2))
    resized = ts.vision.resize_nearest(image, (2, 2))
    chw = ts.vision.to_tensor(image)
    normalized = ts.vision.normalize(chw, mean=(0.5, 0.25, 0.0), std=(0.5, 0.5, 1.0))

    assert cropped.shape == (2, 2, 3)
    assert resized.shape == (2, 2, 3)
    assert normalized.shape == chw.shape
    expected = (chw.numpy() - np.array([0.5, 0.25, 0.0], dtype=np.float32).reshape(3, 1, 1)) / (
        np.array([0.5, 0.5, 1.0], dtype=np.float32).reshape(3, 1, 1)
    )
    np.testing.assert_allclose(normalized.numpy(), expected, rtol=1e-6)


def test_expanded_transforms_and_compose() -> None:
    image = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)

    bilinear = ts.vision.resize_bilinear(image, (6, 8))
    padded = ts.vision.pad(image, (1, 2))
    random_cropped = ts.vision.random_crop(image, (2, 2), seed=1)
    flipped = ts.vision.horizontal_flip(image)
    gray = ts.vision.rgb_to_grayscale(image)
    rgb = ts.vision.grayscale_to_rgb(gray)
    pipeline = ts.vision.Compose(
        [
            ts.vision.Resize((4, 4), interpolation="nearest"),
            ts.vision.CenterCrop((2, 2)),
            ts.vision.ToTensor(),
            ts.vision.Normalize(0.5, 0.5),
        ]
    )
    transformed = pipeline(image)

    assert bilinear.shape == (6, 8, 3)
    assert padded.shape == (5, 8, 3)
    assert random_cropped.shape == (2, 2, 3)
    np.testing.assert_array_equal(flipped[:, 0], image[:, -1])
    assert gray.shape == (3, 4, 1)
    assert rgb.shape == image.shape
    assert isinstance(transformed, ts.Tensor)
    assert transformed.shape == (3, 2, 2)


def test_image_io_grid_visualization_and_imagefolder(tmp_path) -> None:
    pytest.importorskip("PIL.Image")
    cat_dir = tmp_path / "cat"
    dog_dir = tmp_path / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()
    cat = np.full((4, 4, 3), [255, 0, 0], dtype=np.uint8)
    dog = np.full((4, 4, 3), [0, 0, 255], dtype=np.uint8)
    ts.vision.save_image(cat, cat_dir / "cat.png")
    ts.vision.save_image(dog, dog_dir / "dog.png")

    loaded = ts.vision.load_image(cat_dir / "cat.png")
    dataset = ts.vision.ImageFolder(tmp_path)
    image, target = dataset[0]
    loader = ts.data.DataLoader(dataset, batch_size=2)
    batch_images, batch_targets = next(iter(loader))
    grid = ts.vision.make_grid(batch_images, nrow=2, padding=1)
    boxed = ts.vision.draw_bounding_boxes(loaded, [[0, 0, 3, 3]], labels=["cat"])

    assert dataset.classes == ["cat", "dog"]
    assert loaded.shape == (4, 4, 3)
    assert isinstance(image, ts.Tensor)
    assert target in {0, 1}
    assert batch_images.shape == (2, 3, 4, 4)
    assert batch_targets.shape == (2,)
    assert grid.shape == (4, 9, 3)
    assert boxed.shape == (4, 4, 3)


def test_image_manifest_dataset_and_validation(tmp_path) -> None:
    pytest.importorskip("PIL.Image")
    cat_dir = tmp_path / "cat"
    dog_dir = tmp_path / "dog"
    cat_dir.mkdir()
    dog_dir.mkdir()
    cat = np.full((3, 3, 3), [255, 0, 0], dtype=np.uint8)
    dog = np.full((3, 3, 3), [0, 0, 255], dtype=np.uint8)
    ts.vision.save_image(cat, cat_dir / "cat.png")
    ts.vision.save_image(dog, dog_dir / "dog.png")
    manifest_path = tmp_path / "manifest.json"

    manifest = ts.vision.build_image_manifest(tmp_path, manifest_path)
    loaded = ts.vision.load_image_manifest(manifest_path)
    validation = ts.vision.validate_image_manifest(loaded)
    dataset = ts.vision.ImageManifestDataset(manifest_path)
    images, targets = next(iter(ts.data.DataLoader(dataset, batch_size=2)))

    assert manifest["sample_count"] == 2
    assert loaded["classes"] == ["cat", "dog"]
    assert all(sample["sha256"] for sample in loaded["samples"])
    assert validation["valid"] is True
    assert dataset.classes == ["cat", "dog"]
    assert images.shape == (2, 3, 3, 3)
    assert targets.shape == (2,)

    (cat_dir / "cat.png").write_bytes(b"changed")
    invalid = ts.vision.validate_image_manifest(loaded)
    assert invalid["valid"] is False
    assert invalid["checksum_mismatches"] == ["cat/cat.png"]


def test_vision_metrics() -> None:
    logits = ts.tensor([[0.1, 0.9, 0.0], [2.0, 1.0, 0.0], [0.0, 0.1, 0.2]])
    target = ts.tensor([1, 2, 2], dtype="int64")

    assert ts.vision.accuracy(logits, target) == pytest.approx(2 / 3)
    assert ts.vision.top_k_accuracy(logits, target, k=2) == pytest.approx(2 / 3)
    assert ts.vision.top_k_accuracy(logits, target, k=(1, 3)) == pytest.approx((2 / 3, 1.0))
    np.testing.assert_array_equal(
        ts.vision.confusion_matrix(logits, target, num_classes=3),
        np.array([[0, 0, 0], [0, 1, 0], [1, 0, 1]]),
    )
    iou = ts.vision.box_iou(
        np.array([[0.0, 0.0, 2.0, 2.0]]),
        np.array([[1.0, 1.0, 3.0, 3.0], [0.0, 0.0, 2.0, 2.0]]),
    )
    np.testing.assert_allclose(iou, np.array([[1 / 7, 1.0]]))


def test_tiny_conv_classifier_forward_backward() -> None:
    ts.manual_seed(0)
    model = ts.vision.TinyConvClassifier((1, 4, 4), num_classes=2, hidden_channels=2)
    inputs = ts.randn((2, 1, 4, 4), seed=123, requires_grad=True)
    labels = ts.tensor([0, 1], dtype="int64")

    logits = model(inputs)
    loss = ts.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (2, 2)
    assert loss.item() > 0.0
    assert inputs.grad is not None
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_configurable_image_classifier_forward_backward() -> None:
    ts.manual_seed(0)
    model = ts.vision.ImageClassifier((1, 8, 8), num_classes=3, channels=(2, 4))
    inputs = ts.randn((2, 1, 8, 8), seed=456, requires_grad=True)
    labels = ts.tensor([0, 2], dtype="int64")

    logits = model(inputs)
    loss = ts.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    assert logits.shape == (2, 3)
    assert inputs.grad is not None
    assert model.parameter_count() > 0
