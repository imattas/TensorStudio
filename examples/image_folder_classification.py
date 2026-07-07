from __future__ import annotations

from tempfile import TemporaryDirectory

import numpy as np
import tensorstudio as ts
from tensorstudio import nn, optim


def _write_demo_dataset(root: str) -> None:
    for class_name, color in {
        "blue": (0, 0, 255),
        "red": (255, 0, 0),
    }.items():
        class_dir = f"{root}/{class_name}"
        import os

        os.makedirs(class_dir, exist_ok=True)
        for index in range(2):
            image = np.zeros((8, 8, 3), dtype=np.uint8)
            image[:, :] = color
            image[index : index + 2, index : index + 2] = 255
            ts.vision.save_image(image, f"{class_dir}/{index}.png")


def main() -> None:
    ts.manual_seed(0)
    with TemporaryDirectory() as root:
        _write_demo_dataset(root)
        transform = ts.vision.Compose(
            [
                ts.vision.Resize((8, 8)),
                ts.vision.ToTensor(),
                ts.vision.Normalize(0.5, 0.5),
            ]
        )
        dataset = ts.vision.ImageFolder(root, transform=transform)
        loader = ts.data.DataLoader(dataset, batch_size=4, shuffle=True, seed=0)
        model = ts.vision.ImageClassifier((3, 8, 8), num_classes=2, channels=(4,))
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        loss_fn = nn.CrossEntropyLoss()

        for images, labels in loader:
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels.astype("int64"))
            loss.backward()
            optimizer.step()

        accuracy = ts.vision.accuracy(model(images), labels.astype("int64"))
        print("classes:", dataset.classes)
        print("loss:", loss.item())
        print("accuracy:", accuracy)


if __name__ == "__main__":
    main()
