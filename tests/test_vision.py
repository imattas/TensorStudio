from __future__ import annotations

import numpy as np
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
