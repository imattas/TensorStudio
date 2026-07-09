import json

import pytest
import tensorstudio as ts


def test_kernel_manifest_json_discovery_and_registration(tmp_path) -> None:
    manifest_path = tmp_path / "tensorstudio-kernel.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": {
                    "name": "manifest_demo",
                    "compiled": True,
                    "available": False,
                    "device_count": 0,
                    "reason": "metadata only",
                },
                "library": "ignored_library.dll",
                "kernels": [
                    {
                        "op": "add",
                        "dtypes": ["float32", "float64"],
                        "memory_space": "unified",
                        "execution_mode": "async",
                        "forward": True,
                        "backward": True,
                        "available": True,
                        "deterministic": False,
                        "priority": 9,
                    }
                ],
                "module": "this.must.not.be.imported",
            }
        ),
        encoding="utf-8",
    )

    manifests = ts.discover_kernel_manifests(tmp_path)

    assert len(manifests) == 1
    manifest = manifests[0]
    assert manifest.backend == "manifest_demo"
    assert manifest.library == "ignored_library.dll"
    assert manifest.kernels[0].op == "add"
    assert manifest.kernels[0].dtypes == ("float32", "float64")
    assert manifest.kernels[0].memory_space == "unified"
    assert manifest.kernels[0].execution_mode == "async"
    assert manifest.kernels[0].deterministic is False
    assert manifest.kernels[0].priority == 9

    ts.unregister_backend("manifest_demo")
    try:
        ts.register_kernel_manifest(manifest)
        kernels = ts.backend_kernel_info("manifest_demo:0")
        assert kernels[0]["backend"] == "manifest_demo"
        assert kernels[0]["op"] == "add"
        assert "float64" in kernels[0]["dtypes"]
        assert kernels[0]["memory_space"] == "unified"
        assert kernels[0]["execution_mode"] == "async"
        assert kernels[0]["deterministic"] is False
        assert kernels[0]["priority"] == 9
        assert ts.backend_supports_kernel("add", "manifest_demo:0", "float32") is False
    finally:
        ts.unregister_backend("manifest_demo")


def test_kernel_manifest_toml_validation(tmp_path) -> None:
    manifest_path = tmp_path / "tensorstudio-kernel.toml"
    manifest_path.write_text(
        """
schema_version = 1
library = "ignored.so"

[backend]
name = "toml_demo"
compiled = true
available = false
device_count = 0
reason = "metadata only"

[[kernels]]
op = "matmul"
dtypes = ["float32"]
memory_space = "device"
execution_mode = "sync"
forward = true
backward = false
available = true
deterministic = true
priority = 11
""".strip(),
        encoding="utf-8",
    )

    manifest = ts.load_kernel_manifest(manifest_path)

    assert manifest.backend == "toml_demo"
    assert manifest.kernels[0].op == "matmul"
    assert manifest.kernels[0].dtypes == ("float32",)
    assert manifest.kernels[0].priority == 11


def test_kernel_manifest_rejects_executable_claims(tmp_path) -> None:
    manifest_path = tmp_path / "tensorstudio-kernel.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "backend": {
                    "name": "unsafe_demo",
                    "available": True,
                    "device_count": 1,
                },
                "kernels": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="cannot mark external backends available"):
        ts.validate_kernel_manifest(manifest_path)
