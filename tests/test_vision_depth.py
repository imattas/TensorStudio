from __future__ import annotations

import json

import numpy as np
import pytest
import tensorstudio as ts


def test_batched_and_advanced_transforms() -> None:
    image = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    batch = np.stack([image, image], axis=0)

    resized = ts.vision.batch_resize(batch, (2, 2), interpolation="nearest")
    cropped = ts.vision.batch_center_crop(batch, (2, 2))
    jittered = ts.vision.color_jitter(image, brightness=0.1, contrast=0.1, seed=1)
    random_crop = ts.vision.random_resized_crop(image, (3, 3), seed=2)
    rotated = ts.vision.rotate(image, 20.0)
    affine = ts.vision.affine(image, angle=10.0, translate=(1.0, 0.0), scale=1.0)
    cutout = ts.vision.cutout(image, (2, 2), value=0, seed=3)

    assert resized.shape == (2, 2, 2, 3)
    assert cropped.shape == (2, 2, 2, 3)
    assert jittered.shape == image.shape
    assert random_crop.shape == (3, 3, 3)
    assert rotated.shape == image.shape
    assert affine.shape == image.shape
    assert np.count_nonzero(cutout == 0) >= 4


def test_mixup_cutmix_and_transform_classes() -> None:
    images = ts.tensor(
        np.arange(2 * 3 * 4 * 4, dtype=np.float32).reshape(2, 3, 4, 4).tolist()
    )
    targets = ts.tensor([0, 1], dtype="int64")

    mixed, target_a, target_b, lam = ts.vision.mixup(images, targets, alpha=0.4, seed=4)
    cut, cut_a, cut_b, cut_lam = ts.vision.cutmix(images, targets, alpha=1.0, seed=5)
    pipeline = ts.vision.Compose(
        [
            ts.vision.ColorJitter(brightness=0.1, seed=1),
            ts.vision.RandomResizedCrop((4, 4), seed=2),
            ts.vision.RandomRotation(5.0, seed=3),
            ts.vision.Cutout((1, 1), seed=4),
        ]
    )
    transformed = pipeline(np.zeros((4, 4, 3), dtype=np.uint8))

    assert mixed.shape == images.shape
    assert cut.shape == images.shape
    assert target_a.tolist() == [0, 1]
    assert target_b.shape == targets.shape
    assert cut_a.tolist() == [0, 1]
    assert cut_b.shape == targets.shape
    assert 0.0 <= lam <= 1.0
    assert 0.0 <= cut_lam <= 1.0
    assert transformed.shape == (4, 4, 3)


def test_detection_utilities() -> None:
    boxes = np.array([[0.0, 0.0, 2.0, 2.0], [0.2, 0.2, 2.2, 2.2], [5.0, 5.0, 7.0, 7.0]])
    scores = np.array([0.9, 0.8, 0.7])
    anchors = np.array([[0.0, 0.0, 2.0, 2.0], [4.0, 4.0, 8.0, 8.0]])
    targets = np.array([[0.5, 0.5, 2.5, 2.5], [5.0, 5.0, 9.0, 9.0]])

    iou = ts.vision.box_iou(boxes[:1], boxes[1:])
    giou = ts.vision.generalized_box_iou(boxes[:1], boxes[1:])
    diou = ts.vision.distance_box_iou(boxes[:1], boxes[1:])
    kept = ts.vision.nms(boxes, scores, iou_threshold=0.5)
    encoded = ts.vision.encode_boxes(targets, anchors)
    decoded = ts.vision.decode_boxes(encoded, anchors)
    generated = ts.vision.generate_anchors((2, 2), stride=4, sizes=(4.0,), ratios=(1.0,))

    assert ts.vision.box_area(boxes).tolist() == pytest.approx([4.0, 4.0, 4.0])
    assert iou.shape == (1, 2)
    assert giou.shape == iou.shape
    assert diou.shape == iou.shape
    assert kept == [0, 2]
    np.testing.assert_allclose(decoded, targets, rtol=1e-6, atol=1e-6)
    assert generated.shape == (4, 4)


def test_segmentation_helpers() -> None:
    pred = np.array([[0, 1, 1], [0, 2, 2]])
    target = np.array([[0, 1, 0], [0, 2, 2]])
    one_hot = ts.vision.mask_to_one_hot(target, num_classes=3)
    restored = ts.vision.one_hot_to_mask(one_hot)
    boxes = ts.vision.masks_to_boxes(target == 2)
    resized = ts.vision.resize_mask(target, (3, 6))
    cropped = ts.vision.center_crop_mask(target, (1, 2))
    random_crop = ts.vision.random_crop_mask(target, (1, 2), seed=1)

    np.testing.assert_allclose(ts.vision.mask_iou(pred, target, num_classes=3), [2 / 3, 0.5, 1.0])
    assert ts.vision.mean_mask_iou(pred, target, num_classes=3) == pytest.approx(13 / 18)
    np.testing.assert_array_equal(restored, target)
    np.testing.assert_array_equal(boxes, np.array([[1.0, 1.0, 2.0, 1.0]], dtype=np.float32))
    assert resized.shape == (3, 6)
    assert cropped.shape == (1, 2)
    assert random_crop.shape == (1, 2)


def test_detection_and_segmentation_folder_datasets(tmp_path) -> None:
    pytest.importorskip("PIL.Image")
    root = tmp_path / "detection"
    image_dir = root / "images"
    annotation_dir = root / "annotations"
    image_dir.mkdir(parents=True)
    annotation_dir.mkdir()
    ts.vision.save_image(np.zeros((4, 4, 3), dtype=np.uint8), image_dir / "sample.png")
    (annotation_dir / "sample.json").write_text(
        json.dumps({"boxes": [[0, 0, 2, 2]], "labels": [3]}),
        encoding="utf-8",
    )
    detection = ts.vision.DetectionFolder(root)
    image, target = detection[0]

    seg_root = tmp_path / "segmentation"
    (seg_root / "images").mkdir(parents=True)
    (seg_root / "masks").mkdir()
    ts.vision.save_image(np.ones((4, 4, 3), dtype=np.uint8), seg_root / "images" / "mask.png")
    ts.vision.save_image(np.ones((4, 4), dtype=np.uint8), seg_root / "masks" / "mask.png")
    segmentation = ts.data.from_segmentation_folder(seg_root)
    seg_image, mask = segmentation[0]

    assert isinstance(image, ts.Tensor)
    assert target["boxes"].shape == (1, 4)
    assert target["labels"].tolist() == [3]
    assert isinstance(seg_image, ts.Tensor)
    assert mask.shape == (4, 4)


def test_vision_model_blocks_and_visualization() -> None:
    ts.manual_seed(0)
    residual = ts.vision.ResidualBlock(2)
    depthwise = ts.vision.DepthwiseSeparableBlock(2, 4)
    unet = ts.vision.CompactUNet(1, num_classes=2, base_channels=2)

    x = ts.randn((1, 2, 4, 4), seed=1, requires_grad=True)
    residual_out = residual(x)
    depthwise_out = depthwise(x)
    logits = unet(ts.randn((1, 1, 8, 8), seed=2, requires_grad=True))
    loss = logits.mean()
    loss.backward()

    image = np.zeros((6, 6, 3), dtype=np.uint8)
    predictions = ts.vision.draw_predictions(
        image,
        [[1, 1, 4, 4]],
        labels=["object"],
        scores=[0.9],
    )
    overlay = ts.vision.overlay_mask(image, np.ones((6, 6), dtype=np.uint8))
    features = ts.vision.feature_map_grid(ts.randn((1, 4, 3, 3), seed=3), nrow=2)

    assert residual_out.shape == x.shape
    assert depthwise_out.shape == (1, 4, 4, 4)
    assert logits.shape == (1, 2, 8, 8)
    assert all(parameter.grad is not None for parameter in unet.parameters())
    assert predictions.shape == image.shape
    assert overlay.shape == image.shape
    assert features.shape == (6, 6)
