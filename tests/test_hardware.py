import gc

import numpy as np
import pytest
import tensorstudio as ts
from tensorstudio.errors import DeviceError


def test_device_descriptors_and_backend_info() -> None:
    cpu = ts.device("cpu")

    assert str(cpu) == "cpu"
    assert repr(cpu) == "Device('cpu')"
    assert cpu.type == "cpu"
    assert cpu.backend == "cpu"
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

    transfer = ts.device_transfer_info("cpu", cpu)
    assert transfer["source"] == "cpu"
    assert transfer["target"] == "cpu"
    assert transfer["supported"] is True
    assert transfer["direct"] is True
    assert ts.can_transfer("cpu", "cpu") is True


def test_allocator_and_op_interface_metadata() -> None:
    allocators = {item["device"]: item for item in ts.backend_allocator_info()}

    assert allocators["cpu"]["memory_space"] == "host"
    assert allocators["cpu"]["can_allocate"] is True
    assert allocators["cpu"]["alignment"] >= 1
    assert allocators["cuda:0"]["can_allocate"] is False
    assert "not enabled" in allocators["cuda:0"]["reason"]

    cuda_allocator = ts.backend_allocator_info("cuda:0")[0]
    assert cuda_allocator["device"] == "cuda:0"
    assert cuda_allocator["memory_space"] == "device"

    ops = {item["op"]: item for item in ts.backend_op_info()}
    assert ops["matmul"]["category"] == "linear_algebra"
    assert ops["matmul"]["differentiable"] is True
    assert ops["conv2d"]["shape_inference"] is True
    assert ts.backend_op_info("relu")["category"] == "activation"


def test_storage_telemetry_tracks_native_allocations() -> None:
    ts.reset_storage_telemetry()
    baseline = ts.storage_telemetry()

    a = ts.zeros((16,), dtype="float32")
    b = ts.ones((8,), dtype="float64")
    during = ts.storage_telemetry()

    assert during["total_allocations"] >= 2
    assert during["total_bytes_allocated"] >= 16 * 4 + 8 * 8
    assert during["active_allocations"] >= baseline["active_allocations"] + 2
    assert during["active_bytes"] >= baseline["active_bytes"] + 16 * 4 + 8 * 8
    assert during["peak_active_allocations"] >= during["active_allocations"]
    assert during["peak_active_bytes"] >= during["active_bytes"]

    del a, b
    gc.collect()
    after = ts.storage_telemetry()

    assert after["active_allocations"] <= during["active_allocations"]
    assert after["active_bytes"] <= during["active_bytes"]


def test_runtime_capability_metadata() -> None:
    runtimes = {item["device"]: item for item in ts.backend_runtime_info()}

    assert runtimes["cpu"]["executable"] is True
    assert runtimes["cpu"]["supports_eager"] is True
    assert runtimes["cpu"]["supports_graph"] is False
    assert runtimes["cpu"]["supports_compiler"] is False
    assert runtimes["cpu"]["supports_host_fallback"] is False

    assert runtimes["cuda:0"]["executable"] is False
    assert runtimes["cuda:0"]["supports_host_fallback"] is True
    assert "CUDA runtime" in runtimes["cuda:0"]["reason"]

    selected = ts.backend_runtime_info("cpu")[0]
    assert selected["device"] == "cpu"
    assert selected["backend"] == "cpu"


def test_device_properties_logical_devices_and_execution_plans() -> None:
    properties = {item["device"]: item for item in ts.backend_device_properties()}

    assert properties["cpu"]["available"] is True
    assert properties["cpu"]["architecture"] == "portable-cpu"
    assert properties["cpu"]["unified_addressing"] is True
    assert properties["cpu"]["max_threads_per_block"] >= 1
    assert properties["cuda:0"]["available"] is False
    assert properties["cuda:0"]["vendor"] == "nvidia"

    cpu_properties = ts.backend_device_properties("cpu")[0]
    assert cpu_properties["device"] == "cpu"

    logical_devices = ts.logical_device_info()
    assert logical_devices == [
        {
            "physical_device": "cpu",
            "logical_device": "cpu",
            "memory_limit_bytes": 0,
            "priority": 0,
            "default_device": True,
            "reason": "CPU exposes one default logical device backed by host storage",
        },
    ]
    assert ts.logical_device_info("cuda:0") == []

    cpu_plan = ts.backend_execution_plan("add", "cpu", "float32", input_devices=["cpu", "cpu"])
    assert cpu_plan["requested"] == "cpu"
    assert cpu_plan["selected"] == "cpu"
    assert cpu_plan["executable"] is True
    assert cpu_plan["uses_fallback"] is False
    assert cpu_plan["requires_transfer"] is False
    assert len(cpu_plan["input_transfers"]) == 2

    cuda_plan = ts.backend_execution_plan("add", "cuda:0", "float32")
    assert cuda_plan["requested"] == "cuda:0"
    assert cuda_plan["selected"] == "cpu"
    assert cuda_plan["executable"] is True
    assert cuda_plan["uses_fallback"] is True
    assert cuda_plan["input_devices"] == ["cpu"]

    blocked_plan = ts.backend_execution_plan("add", "cpu", "float32", input_devices=["cuda:0"])
    assert blocked_plan["executable"] is False
    assert blocked_plan["requires_transfer"] is True
    assert blocked_plan["input_transfers"][0]["supported"] is False


def test_tensor_to_device_cpu_alias_copy_and_unavailable_backend() -> None:
    x = ts.arange(3, dtype="float64")
    alias = x.to_device("cpu")
    copied = ts.to_device(x, ts.device("cpu"), copy=True)

    x._assign(ts.ones(x.shape, dtype="float64"))

    assert alias.device == "cpu"
    assert copied.device == "cpu"
    assert alias.tolist() == [1.0, 1.0, 1.0]
    assert copied.tolist() == [0.0, 1.0, 2.0]

    with pytest.raises(DeviceError, match="non-CPU tensor storage"):
        x.to_device("cuda:0")


def test_device_scope_and_creation_device_contracts() -> None:
    assert str(ts.current_device()) == "cpu"

    direct_cases = [
        ts.tensor([1.0, 2.0], device="cpu"),
        ts.from_numpy(np.array([1.0, 2.0], dtype=np.float32), device=ts.device("cpu")),
        ts.zeros((2,), device="cpu"),
        ts.ones((2,), device="cpu"),
        ts.empty((2,), device="cpu"),
        ts.full((2,), 3.0, device="cpu"),
        ts.rand((2,), seed=1, device="cpu"),
        ts.randn((2,), seed=1, device="cpu"),
        ts.arange(2, device="cpu"),
        ts.eye(2, device="cpu"),
        ts.linspace(0.0, 1.0, 2, device="cpu"),
    ]
    assert {item.device for item in direct_cases} == {"cpu"}

    with ts.device_scope("cpu") as scoped:
        assert str(scoped) == "cpu"
        assert str(ts.current_device()) == "cpu"
        assert ts.zeros((1,)).device == "cpu"

    assert str(ts.current_device()) == "cpu"

    base = ts.ones((2,), requires_grad=True)
    with ts.device_scope("cuda:0"):
        assert str(ts.current_device()) == "cuda:0"
        same_device = ts.zeros_like(base)
        assert same_device.device == "cpu"
        with pytest.raises(DeviceError, match="non-CPU tensor storage"):
            ts.zeros((1,))
        with pytest.raises(DeviceError, match="non-CPU tensor storage"):
            ts.ones_like(base, device="cuda:0")

    assert str(ts.current_device()) == "cpu"

    ts.set_default_device("cpu")
    try:
        assert ts.full((1,), 2.0).device == "cpu"
    finally:
        ts.reset_default_device()
    assert str(ts.current_device()) == "cpu"


def test_backend_kernel_registry_and_placement_metadata() -> None:
    kernels = ts.backend_kernel_info()
    cpu_kernels = ts.backend_kernel_info("cpu")
    ops = {item["op"]: item for item in cpu_kernels}

    assert len(kernels) >= len(cpu_kernels) > 0
    assert ops["add"]["backend"] == "cpu"
    assert "float32" in ops["add"]["dtypes"]
    assert ops["add"]["memory_space"] == "host"
    assert ops["add"]["execution_mode"] == "sync"
    assert ops["add"]["deterministic"] is True
    assert ops["add"]["priority"] == 100
    assert ops["add"]["backward"] is True
    assert ts.backend_supports_kernel("add", "cpu", "float32") is True
    assert ts.backend_supports_kernel("conv2d", "cpu", "float64") is True
    assert ts.backend_supports_kernel("add", "cuda:0", "float32") is False
    assert ts.backend_supports_kernel("definitely_missing", "cpu", "float32") is False

    cpu_placement = ts.kernel_placement_info("add", "cpu", "float32")
    assert cpu_placement["requested"] == "cpu"
    assert cpu_placement["selected"] == "cpu"
    assert cpu_placement["supported"] is True
    assert "registered" in cpu_placement["reason"]

    cuda_placement = ts.kernel_placement_info("add", "cuda:0", "float32")
    assert cuda_placement["requested"] == "cuda:0"
    assert cuda_placement["selected"] == "cpu"
    assert cuda_placement["supported"] is False
    assert cuda_placement["fallback_to_cpu"] is True
    assert cuda_placement["requires_transfer"] is False
    assert "not available" in cuda_placement["reason"]


def test_external_backend_descriptor_registration_is_safe() -> None:
    ts.unregister_backend("demo_accel")
    try:
        ts.register_backend("demo_accel", reason="test descriptor only")
        ts.register_backend_kernel(
            "demo_accel",
            "add",
            ["float32"],
            backward=True,
            memory_space="unified",
            execution_mode="async",
            deterministic=False,
            priority=7,
        )

        descriptor = ts.device("demo_accel:0")
        assert descriptor.type == "plugin"
        assert descriptor.backend == "demo_accel"
        assert ts.is_available(descriptor) is False

        info = {item["name"]: item for item in ts.backend_info()}
        assert info["demo_accel"]["compiled"] is True
        assert info["demo_accel"]["available"] is False
        assert info["demo_accel"]["device_count"] == 0

        kernels = ts.backend_kernel_info("demo_accel:0")
        assert kernels[0]["op"] == "add"
        assert kernels[0]["backend"] == "demo_accel"
        assert kernels[0]["available"] is True
        assert kernels[0]["memory_space"] == "unified"
        assert kernels[0]["execution_mode"] == "async"
        assert kernels[0]["deterministic"] is False
        assert kernels[0]["priority"] == 7
        assert ts.backend_supports_kernel("add", descriptor, "float32") is False
        runtime = ts.backend_runtime_info(descriptor)[0]
        assert runtime["executable"] is False
        assert runtime["compiler"] == "external plugin descriptor"
        assert "descriptor-only" in runtime["reason"]
    finally:
        ts.unregister_backend("demo_accel")


def test_unavailable_accelerators_fail_clearly() -> None:
    assert ts.cuda_is_available() is False
    assert ts.metal_is_available() is False
    assert ts.device_count("cuda") == 0
    cuda_transfer = ts.device_transfer_info("cpu", "cuda:0")
    assert cuda_transfer["supported"] is False
    assert "not available" in cuda_transfer["reason"]
    assert ts.can_transfer("cpu", "cuda:0") is False

    with pytest.raises(DeviceError, match="not available"):
        ts.require_device("cuda")
    with pytest.raises(DeviceError, match="cpu device only supports index 0"):
        ts.device("cpu:1")
    with pytest.raises(DeviceError, match="backend storage hooks"):
        ts.register_backend("unsafe_demo", available=True, device_count=1)
