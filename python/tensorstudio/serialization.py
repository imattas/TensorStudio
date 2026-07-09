"""Serialization helpers.

TensorStudio supports two file paths:

* ``save``/``load`` use pickle for complete internal object roundtrips. Loading
  pickle data from untrusted sources is unsafe because pickle can execute
  arbitrary code during deserialization.
* ``save_npz``/``load_npz`` use NumPy's non-pickle ``.npz`` container for
  tensors and flat ``state_dict`` mappings. This is the safer interchange format
  for numeric weights.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np

from ._version import __version__
from .tensor import Tensor, from_numpy
from .typing import PathLikeStr

_NPZ_METADATA_KEY = "__tensorstudio_metadata__"
_NPZ_FORMAT = "tensorstudio.npz"


def save(obj: Any, path: PathLikeStr) -> None:
    """Serialize a TensorStudio object to a file with pickle."""

    with Path(path).open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load(path: PathLikeStr) -> Any:
    """Load a TensorStudio object from a pickle file.

    Only load files you created or otherwise trust.
    """

    with Path(path).open("rb") as file:
        return pickle.load(file)  # noqa: S301


def save_npz(
    obj: Tensor | dict[str, Tensor],
    path: PathLikeStr,
    *,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a tensor or flat state_dict to a portable ``.npz`` file.

    The archive stores raw NumPy arrays and JSON metadata with
    ``allow_pickle=False`` compatibility. It is intended for TensorStudio tensor
    values, not arbitrary Python objects.
    """

    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    if isinstance(obj, Tensor):
        arrays["arr_0"] = obj.numpy()
        entries.append(
            {
                "name": "",
                "key": "arr_0",
                "dtype": obj.dtype,
                "shape": list(obj.shape),
                "requires_grad": obj.requires_grad,
            }
        )
        kind = "tensor"
    elif isinstance(obj, dict):
        kind = "state_dict"
        for index, (name, value) in enumerate(sorted(obj.items())):
            if not isinstance(name, str):
                raise TypeError("save_npz state_dict keys must be strings")
            if not isinstance(value, Tensor):
                raise TypeError("save_npz state_dict values must be Tensor objects")
            key = f"arr_{index}"
            arrays[key] = value.numpy()
            entries.append(
                {
                    "name": name,
                    "key": key,
                    "dtype": value.dtype,
                    "shape": list(value.shape),
                    "requires_grad": value.requires_grad,
                }
            )
    else:
        raise TypeError("save_npz expects a Tensor or dict[str, Tensor]")

    archive_metadata = {
        "format": _NPZ_FORMAT,
        "version": 2,
        "tensorstudio_version": __version__,
        "kind": kind,
        "tensor_count": len(entries),
        "tensors": entries,
        "metadata": dict(metadata or {}),
    }
    arrays[_NPZ_METADATA_KEY] = np.array(json.dumps(archive_metadata, sort_keys=True))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as file:
        savez_compressed: Any = np.savez_compressed
        savez_compressed(file, **arrays)


def load_npz(path: PathLikeStr) -> Tensor | dict[str, Tensor]:
    """Load a tensor or flat state_dict saved with :func:`save_npz`."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if _NPZ_METADATA_KEY not in archive:
            raise ValueError("not a TensorStudio npz archive: missing metadata")
        metadata = json.loads(str(archive[_NPZ_METADATA_KEY].item()))
        if metadata.get("format") != _NPZ_FORMAT:
            raise ValueError("not a TensorStudio npz archive")
        if metadata.get("version") not in {1, 2}:
            raise ValueError(f"unsupported TensorStudio npz version: {metadata.get('version')}")

        tensors: dict[str, Tensor] = {}
        for entry in metadata.get("tensors", []):
            key = entry["key"]
            if key not in archive:
                raise ValueError(f"TensorStudio npz archive is missing array {key!r}")
            value = from_numpy(np.array(archive[key], copy=True))
            if bool(entry.get("requires_grad", False)):
                value.requires_grad = True
            tensors[str(entry["name"])] = value

        kind = metadata.get("kind")
        if kind == "tensor":
            if "" not in tensors:
                raise ValueError("TensorStudio tensor npz archive is missing tensor payload")
            return tensors[""]
        if kind == "state_dict":
            return tensors
        raise ValueError(f"unsupported TensorStudio npz archive kind: {kind!r}")


def load_npz_metadata(path: PathLikeStr) -> dict[str, Any]:
    """Return TensorStudio NPZ archive metadata without loading tensor values."""

    with np.load(Path(path), allow_pickle=False) as archive:
        if _NPZ_METADATA_KEY not in archive:
            raise ValueError("not a TensorStudio npz archive: missing metadata")
        metadata = json.loads(str(archive[_NPZ_METADATA_KEY].item()))
        if metadata.get("format") != _NPZ_FORMAT:
            raise ValueError("not a TensorStudio npz archive")
        return dict(metadata)


def save_safetensors(
    tensors: dict[str, Tensor],
    path: PathLikeStr,
    *,
    metadata: dict[str, str] | None = None,
) -> None:
    """Save a tensor mapping to the SafeTensors format.

    This requires the optional dependency:
    ``python -m pip install 'tensorstudio[safetensors]'``.
    """

    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SafeTensors support requires: python -m pip install 'tensorstudio[safetensors]'"
        ) from exc
    arrays = _tensor_mapping_to_arrays(tensors)
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    safe_metadata = {"format": "tensorstudio.safetensors", "tensorstudio_version": __version__}
    safe_metadata.update(metadata or {})
    save_file(arrays, str(output_path), metadata=safe_metadata)


def load_safetensors(path: PathLikeStr) -> dict[str, Tensor]:
    """Load a SafeTensors tensor mapping as TensorStudio tensors."""

    try:
        from safetensors.numpy import load_file
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SafeTensors support requires: python -m pip install 'tensorstudio[safetensors]'"
        ) from exc
    arrays = load_file(str(path))
    return {name: from_numpy(np.array(value, copy=True)) for name, value in arrays.items()}


def inspect_model_metadata(
    path: PathLikeStr,
    *,
    trusted_pickle: bool = False,
) -> dict[str, Any]:
    """Inspect metadata for supported TensorStudio, SafeTensors, and ONNX files.

    Trusted pickle checkpoints require ``trusted_pickle=True`` because reading
    pickle files can execute arbitrary code.
    """

    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix in {".npz", ".tsnpz"}:
        return load_npz_metadata(input_path)
    if suffix == ".safetensors":
        return _inspect_safetensors_metadata(input_path)
    if suffix == ".onnx":
        from tensorstudio.interchange import inspect_onnx

        return inspect_onnx(input_path)
    if suffix in {".tsmodel", ".pkl", ".pickle"}:
        if not trusted_pickle:
            raise ValueError("trusted_pickle=True is required to inspect pickle checkpoints")
        checkpoint = load(input_path)
        if not isinstance(checkpoint, dict):
            raise ValueError("checkpoint metadata expects a dictionary payload")
        model_state = checkpoint.get("model")
        model_tensors = len(model_state) if isinstance(model_state, dict) else 0
        return {
            "format": "tensorstudio.checkpoint",
            "tensorstudio_version": checkpoint.get("tensorstudio_version"),
            "epoch": checkpoint.get("epoch"),
            "has_optimizer": "optimizer" in checkpoint,
            "has_scheduler": "scheduler" in checkpoint,
            "model_tensor_count": model_tensors,
            "metadata": checkpoint.get("metadata", {}),
        }
    raise ValueError(f"unsupported metadata format for {input_path}")


def _tensor_mapping_to_arrays(tensors: dict[str, Tensor]) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for name, value in tensors.items():
        if not isinstance(name, str):
            raise TypeError("tensor mapping keys must be strings")
        if not isinstance(value, Tensor):
            raise TypeError("tensor mapping values must be Tensor objects")
        arrays[name] = value.numpy()
    return arrays


def _inspect_safetensors_metadata(path: Path) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "SafeTensors support requires: python -m pip install 'tensorstudio[safetensors]'"
        ) from exc
    with safe_open(str(path), framework="numpy") as handle:
        keys = list(handle.keys())
        tensors = [
            {
                "name": key,
                "shape": list(handle.get_tensor(key).shape),
                "dtype": str(handle.get_tensor(key).dtype),
            }
            for key in keys
        ]
        return {
            "format": "safetensors",
            "metadata": dict(handle.metadata() or {}),
            "tensor_count": len(keys),
            "tensors": tensors,
        }


__all__ = [
    "inspect_model_metadata",
    "load",
    "load_npz",
    "load_npz_metadata",
    "load_safetensors",
    "save",
    "save_npz",
    "save_safetensors",
]
