"""Small data utilities."""

from __future__ import annotations

from .dataloader import DataLoader
from .dataset import ArrayDataset, Dataset, Subset, TensorDataset, dataset_summary
from .factories import (
    from_arrays,
    from_detection_folder,
    from_image_folder,
    from_segmentation_folder,
    from_tensors,
)
from .manifest import (
    CachedDataset,
    DatasetManifest,
    ManifestEntry,
    build_dataset_manifest,
    cache_dataset,
    file_sha256,
    load_dataset_manifest,
    save_dataset_manifest,
    validate_dataset_manifest,
)
from .public_formats import (
    CSVDataset,
    JSONLDataset,
    LibSVMDataset,
    TextLineDataset,
    from_csv,
    from_jsonl,
    from_libsvm,
    from_text_lines,
)
from .splitting import train_val_split

__all__ = [
    "ArrayDataset",
    "CachedDataset",
    "CSVDataset",
    "DataLoader",
    "Dataset",
    "DatasetManifest",
    "JSONLDataset",
    "LibSVMDataset",
    "ManifestEntry",
    "Subset",
    "TensorDataset",
    "TextLineDataset",
    "build_dataset_manifest",
    "cache_dataset",
    "dataset_summary",
    "file_sha256",
    "from_arrays",
    "from_csv",
    "from_detection_folder",
    "from_image_folder",
    "from_jsonl",
    "from_libsvm",
    "from_segmentation_folder",
    "from_text_lines",
    "from_tensors",
    "load_dataset_manifest",
    "save_dataset_manifest",
    "train_val_split",
    "validate_dataset_manifest",
]
