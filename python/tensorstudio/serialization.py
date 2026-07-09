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


def save_npz(obj: Tensor | dict[str, Tensor], path: PathLikeStr) -> None:
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

    metadata = {
        "format": _NPZ_FORMAT,
        "version": 1,
        "kind": kind,
        "tensors": entries,
    }
    arrays[_NPZ_METADATA_KEY] = np.array(json.dumps(metadata, sort_keys=True))

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
        if metadata.get("version") != 1:
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


def inspect_npz(path: PathLikeStr) -> dict[str, Any]:
    """Inspect TensorStudio NPZ metadata without constructing tensors."""

    archive_path = Path(path)
    with np.load(archive_path, allow_pickle=False) as archive:
        files = list(archive.files)
        if _NPZ_METADATA_KEY not in archive:
            raise ValueError("not a TensorStudio npz archive: missing metadata")
        metadata = json.loads(str(archive[_NPZ_METADATA_KEY].item()))
        if metadata.get("format") != _NPZ_FORMAT:
            raise ValueError("not a TensorStudio npz archive")

        tensor_entries: list[dict[str, Any]] = []
        missing_arrays: list[str] = []
        expected_arrays = {_NPZ_METADATA_KEY}
        for entry in metadata.get("tensors", []):
            key = str(entry.get("key"))
            expected_arrays.add(key)
            array_present = key in archive
            if not array_present:
                missing_arrays.append(key)
                array_dtype = None
                array_shape = None
            else:
                array = archive[key]
                array_dtype = str(array.dtype)
                array_shape = list(array.shape)
            tensor_entries.append(
                {
                    "name": str(entry.get("name", "")),
                    "key": key,
                    "dtype": str(entry.get("dtype")),
                    "shape": list(entry.get("shape", [])),
                    "requires_grad": bool(entry.get("requires_grad", False)),
                    "array_present": array_present,
                    "array_dtype": array_dtype,
                    "array_shape": array_shape,
                }
            )

        extra_arrays = sorted(name for name in files if name not in expected_arrays)
        version = metadata.get("version")
        kind = metadata.get("kind")
        compatible = version == 1 and kind in {"tensor", "state_dict"} and not missing_arrays
        reason = ""
        if version != 1:
            reason = f"unsupported TensorStudio npz version: {version}"
        elif kind not in {"tensor", "state_dict"}:
            reason = f"unsupported TensorStudio npz archive kind: {kind!r}"
        elif missing_arrays:
            reason = "TensorStudio npz archive is missing one or more array payloads"

        return {
            "path": str(archive_path),
            "format": metadata.get("format"),
            "version": version,
            "kind": kind,
            "tensor_count": len(tensor_entries),
            "tensors": tensor_entries,
            "archive_arrays": files,
            "missing_arrays": missing_arrays,
            "extra_arrays": extra_arrays,
            "compatible": compatible,
            "reason": reason,
        }


def check_npz_compatibility(path: PathLikeStr) -> dict[str, Any]:
    """Return versioned TensorStudio NPZ compatibility metadata."""

    info = inspect_npz(path)
    return {
        "path": info["path"],
        "format": info["format"],
        "version": info["version"],
        "kind": info["kind"],
        "compatible": info["compatible"],
        "tensor_count": info["tensor_count"],
        "missing_arrays": info["missing_arrays"],
        "extra_arrays": info["extra_arrays"],
        "reason": info["reason"],
    }


__all__ = [
    "check_npz_compatibility",
    "inspect_npz",
    "load",
    "load_npz",
    "save",
    "save_npz",
]
