import pytest
import tensorstudio as ts
from tensorstudio.errors import DeviceError


def test_device_descriptors_and_backend_info() -> None:
    cpu = ts.device("cpu")

    assert str(cpu) == "cpu"
    assert repr(cpu) == "Device('cpu')"
    assert cpu.type == "cpu"
    assert cpu.index == 0
    assert cpu.is_cpu is True
    assert cpu == ts.Device("cpu")
    assert ts.device_count("cpu") == 1
    assert ts.is_available(cpu) is True
    assert [str(device) for device in ts.available_devices()] == ["cpu"]

    info = {item["name"]: item for item in ts.backend_info()}
    assert info["cpu"]["available"] is True
    assert info["cuda"]["available"] is False
    assert info["metal"]["available"] is False


def test_tensor_device_transfer_api_keeps_dtype_casts() -> None:
    x = ts.tensor([1, 2, 3], device="cpu")

    assert x.device == "cpu"
    assert x.cpu().device == "cpu"
    assert x.to_device("cpu").tolist() == [1, 2, 3]
    assert x.to(ts.device("cpu")).device == "cpu"
    assert x.to("float64").dtype == "float64"


def test_unavailable_accelerators_fail_clearly() -> None:
    assert ts.cuda_is_available() is False
    assert ts.metal_is_available() is False
    assert ts.device_count("cuda") == 0

    with pytest.raises(DeviceError, match="not available"):
        ts.zeros((2,), device="cuda")
    with pytest.raises(DeviceError, match="not available"):
        ts.ones((2,)).to("metal:0")
    with pytest.raises(DeviceError, match="cpu device only supports index 0"):
        ts.device("cpu:1")
