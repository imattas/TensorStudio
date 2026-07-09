"""Safe model-format metadata inspection helpers."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

from tensorstudio.typing import PathLikeStr

from .onnx import inspect_onnx

_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_TFLITE_IDENTIFIER = b"TFL3"
_KERAS_JSON_LIMIT = 5 * 1024 * 1024


def _read_zip_json(
    archive: zipfile.ZipFile,
    name: str,
    *,
    max_bytes: int = _KERAS_JSON_LIMIT,
) -> dict[str, Any] | None:
    try:
        info = archive.getinfo(name)
    except KeyError:
        return None
    if info.file_size > max_bytes:
        raise ValueError(
            f"Keras archive JSON entry {name!r} is too large to inspect safely "
            f"({info.file_size} bytes)"
        )
    with archive.open(info) as file:
        data = file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"Keras archive JSON entry {name!r} exceeds safe inspection limit")
    value = json.loads(data.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Keras archive JSON entry {name!r} must contain a JSON object")
    return value


def _keras_layer_summary(config: dict[str, Any]) -> dict[str, Any]:
    model_config = config.get("config")
    if not isinstance(model_config, dict):
        return {"layer_count": None, "layer_classes": []}
    layers = model_config.get("layers")
    if not isinstance(layers, list):
        return {"layer_count": None, "layer_classes": []}
    layer_classes: list[str] = []
    for layer in layers:
        if isinstance(layer, dict):
            class_name = layer.get("class_name")
            if isinstance(class_name, str):
                layer_classes.append(class_name)
    return {
        "layer_count": len(layers),
        "layer_classes": layer_classes,
    }


def inspect_keras(path: PathLikeStr) -> dict[str, Any]:
    """Inspect a Keras ``.keras`` archive without extracting or loading code."""

    archive_path = Path(path)
    if not archive_path.is_file():
        raise FileNotFoundError(f"Keras archive not found: {archive_path}")
    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Keras archive is not a valid zip file: {archive_path}")

    with zipfile.ZipFile(archive_path) as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        metadata = _read_zip_json(archive, "metadata.json")
        config = _read_zip_json(archive, "config.json")
        weight_entries = [
            name for name in names if name.endswith(".weights.h5") or name.endswith(".h5")
        ]
        summary = _keras_layer_summary(config or {})
        return {
            "path": str(archive_path),
            "format": "keras",
            "container": "zip",
            "safe_metadata_only": True,
            "file_count": len(members),
            "total_uncompressed_size": sum(member.file_size for member in members),
            "entries": names,
            "has_metadata_json": metadata is not None,
            "has_config_json": config is not None,
            "has_weights": bool(weight_entries),
            "weight_entries": weight_entries,
            "keras_version": metadata.get("keras_version") if metadata else None,
            "date_saved": metadata.get("date_saved") if metadata else None,
            "model_class": config.get("class_name") if config else None,
            "registered_name": config.get("registered_name") if config else None,
            "build_config": config.get("build_config") if config else None,
            **summary,
        }


def inspect_saved_model(path: PathLikeStr) -> dict[str, Any]:
    """Inspect TensorFlow SavedModel directory structure without loading TensorFlow."""

    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"SavedModel directory not found: {root}")
    saved_model_pb = root / "saved_model.pb"
    saved_model_pbtxt = root / "saved_model.pbtxt"
    variables_dir = root / "variables"
    assets_dir = root / "assets"
    variables = sorted(
        str(item.relative_to(root))
        for item in variables_dir.glob("*")
        if item.is_file()
    ) if variables_dir.is_dir() else []
    assets = sorted(
        str(item.relative_to(root))
        for item in assets_dir.rglob("*")
        if item.is_file()
    ) if assets_dir.is_dir() else []
    fingerprint = root / "fingerprint.pb"
    return {
        "path": str(root),
        "format": "tensorflow_saved_model",
        "container": "directory",
        "safe_metadata_only": True,
        "has_saved_model_pb": saved_model_pb.is_file(),
        "has_saved_model_pbtxt": saved_model_pbtxt.is_file(),
        "has_variables_dir": variables_dir.is_dir(),
        "has_assets_dir": assets_dir.is_dir(),
        "has_fingerprint": fingerprint.is_file(),
        "saved_model_pb_size": saved_model_pb.stat().st_size if saved_model_pb.is_file() else None,
        "saved_model_pbtxt_size": (
            saved_model_pbtxt.stat().st_size if saved_model_pbtxt.is_file() else None
        ),
        "variables": variables,
        "assets": assets,
        "loadable_by_tensorstudio": False,
        "reason": (
            "TensorStudio inspects SavedModel structure but does not import SavedModel graphs"
        ),
    }


def _inspect_hdf5_with_h5py(path: Path) -> dict[str, Any] | None:
    try:
        import h5py
    except ImportError:
        return None

    datasets: list[dict[str, Any]] = []
    groups: list[str] = []
    attrs: dict[str, Any] = {}
    try:
        with h5py.File(path, "r") as file:
            for key, value in file.attrs.items():
                if isinstance(value, bytes):
                    attrs[key] = value.decode("utf-8", errors="replace")
                elif hasattr(value, "tolist"):
                    attrs[key] = value.tolist()
                else:
                    attrs[key] = value

            def visit(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Dataset):
                    datasets.append(
                        {
                            "name": name,
                            "shape": list(obj.shape),
                            "dtype": str(obj.dtype),
                        }
                    )
                elif isinstance(obj, h5py.Group):
                    groups.append(name)

            file.visititems(visit)
    except OSError as exc:
        return {
            "h5py_parse_error": str(exc),
            "groups": [],
            "datasets": [],
            "dataset_count": 0,
            "group_count": 0,
        }

    return {
        "attributes": attrs,
        "groups": groups,
        "datasets": datasets,
        "dataset_count": len(datasets),
        "group_count": len(groups),
    }


def inspect_hdf5(path: PathLikeStr) -> dict[str, Any]:
    """Inspect HDF5/Keras weight-file metadata without loading model code."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"HDF5 file not found: {file_path}")
    with file_path.open("rb") as file:
        signature = file.read(len(_HDF5_SIGNATURE))
    if signature != _HDF5_SIGNATURE:
        raise ValueError(f"file does not have an HDF5 signature: {file_path}")

    optional = _inspect_hdf5_with_h5py(file_path)
    result: dict[str, Any] = {
        "path": str(file_path),
        "format": "hdf5",
        "container": "hdf5",
        "safe_metadata_only": True,
        "has_hdf5_signature": True,
        "size": file_path.stat().st_size,
        "h5py_available": optional is not None,
        "loadable_by_tensorstudio": False,
        "reason": "TensorStudio can inspect HDF5 metadata but does not import HDF5 weights",
    }
    if optional is not None:
        result.update(optional)
    return result


def inspect_tflite(path: PathLikeStr) -> dict[str, Any]:
    """Inspect TensorFlow Lite flatbuffer identity without executing a model."""

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"TFLite file not found: {file_path}")
    with file_path.open("rb") as file:
        header = file.read(8)
    if len(header) < 8 or header[4:8] != _TFLITE_IDENTIFIER:
        raise ValueError(f"file does not have a TensorFlow Lite flatbuffer identifier: {file_path}")
    return {
        "path": str(file_path),
        "format": "tensorflow_lite",
        "container": "flatbuffer",
        "safe_metadata_only": True,
        "has_tflite_identifier": True,
        "size": file_path.stat().st_size,
        "loadable_by_tensorstudio": False,
        "reason": (
            "TensorStudio can identify TFLite files but does not import or execute TFLite graphs"
        ),
    }


def inspect_model_format(path: PathLikeStr) -> dict[str, Any]:
    """Inspect known model-file metadata without executing untrusted model code."""

    target = Path(path)
    if target.is_dir():
        return inspect_saved_model(target)
    suffix = target.suffix.lower()
    if suffix == ".onnx":
        info = inspect_onnx(target)
        info["format"] = "onnx"
        info["safe_metadata_only"] = True
        return info
    if suffix == ".keras":
        return inspect_keras(target)
    if suffix in {".h5", ".hdf5"}:
        return inspect_hdf5(target)
    if suffix == ".tflite":
        return inspect_tflite(target)
    if target.is_file():
        with target.open("rb") as file:
            signature = file.read(max(len(_HDF5_SIGNATURE), 8))
        if signature.startswith(_HDF5_SIGNATURE):
            return inspect_hdf5(target)
        if len(signature) >= 8 and signature[4:8] == _TFLITE_IDENTIFIER:
            return inspect_tflite(target)
    raise ValueError(
        f"unsupported model format for safe inspection: {target}. "
        "Expected an ONNX file, Keras .keras archive, HDF5 file, TFLite file, "
        "or SavedModel directory."
    )


__all__ = [
    "inspect_hdf5",
    "inspect_keras",
    "inspect_model_format",
    "inspect_saved_model",
    "inspect_tflite",
]
