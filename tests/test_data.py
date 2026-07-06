from __future__ import annotations

import numpy as np
import tensorstudio as ts
from tensorstudio.data import DataLoader, TensorDataset


def test_tensor_dataset_and_dataloader_batches() -> None:
    features = ts.tensor([[0.0], [1.0], [2.0], [3.0], [4.0]])
    targets = ts.tensor([[1.0], [3.0], [5.0], [7.0], [9.0]])
    dataset = TensorDataset(features, targets)
    loader = DataLoader(dataset, batch_size=2)

    batches = list(loader)

    assert len(dataset) == 5
    assert len(batches) == 3
    np.testing.assert_allclose(batches[0][0].numpy(), np.array([[0.0], [1.0]], dtype=np.float32))
    np.testing.assert_allclose(batches[0][1].numpy(), np.array([[1.0], [3.0]], dtype=np.float32))


def test_dataloader_shuffle_is_deterministic_and_drop_last() -> None:
    dataset = TensorDataset(ts.arange(6))
    first = [
        batch[0].tolist() for batch in DataLoader(dataset, batch_size=2, shuffle=True, seed=7)
    ]
    second = [
        batch[0].tolist() for batch in DataLoader(dataset, batch_size=2, shuffle=True, seed=7)
    ]
    dropped = list(DataLoader(dataset, batch_size=4, drop_last=True))

    assert first == second
    assert len(dropped) == 1
    assert dropped[0][0].shape == (4,)
