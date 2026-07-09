#include "bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensorstudio/device.hpp"
#include "tensorstudio/dtype.hpp"
#include "tensorstudio/errors.hpp"
#include "tensorstudio/storage.hpp"
#include "tensorstudio/version.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_C, module) {
  module.doc() = "TensorStudio C++ tensor and autograd extension";
  module.attr("__version__") = tensorstudio::version;

  static py::exception<tensorstudio::TensorStudioError> base_error(module, "TensorStudioError");
  py::register_exception<tensorstudio::ShapeError>(module, "ShapeError", base_error.ptr());
  py::register_exception<tensorstudio::DTypeError>(module, "DTypeError", base_error.ptr());
  py::register_exception<tensorstudio::DeviceError>(module, "DeviceError", base_error.ptr());
  py::register_exception<tensorstudio::AutogradError>(module, "AutogradError", base_error.ptr());

  py::class_<tensorstudio::Device>(module, "Device")
      .def(py::init([](const std::string& spec) { return tensorstudio::parse_device(spec); }),
           py::arg("spec") = "cpu")
      .def_property_readonly("type", [](const tensorstudio::Device& self) {
        return tensorstudio::device_type_name(self.type());
      })
      .def_property_readonly("backend", &tensorstudio::Device::backend)
      .def_property_readonly("index", &tensorstudio::Device::index)
      .def_property_readonly("is_cpu", &tensorstudio::Device::is_cpu)
      .def_property_readonly("is_accelerator", &tensorstudio::Device::is_accelerator)
      .def("__str__", &tensorstudio::Device::str)
      .def("__repr__", [](const tensorstudio::Device& self) {
        return "Device('" + self.str() + "')";
      })
      .def("__eq__", [](const tensorstudio::Device& self, const tensorstudio::Device& other) {
        return self == other;
      })
      .def("__ne__", [](const tensorstudio::Device& self, const tensorstudio::Device& other) {
        return self != other;
      });

  auto device_from_py = [](py::object object) {
    if (py::isinstance<tensorstudio::Device>(object)) {
      return py::cast<tensorstudio::Device>(object);
    }
    if (py::isinstance<py::str>(object)) {
      return tensorstudio::parse_device(py::cast<std::string>(object));
    }
    throw tensorstudio::DeviceError("device must be a TensorStudio Device or a string such as 'cpu' or 'cuda:0'");
  };
  auto backend_info_dict = [](const tensorstudio::BackendInfo& info) {
    py::dict item;
    item["name"] = info.name;
    item["device_type"] = tensorstudio::device_type_name(info.type);
    item["compiled"] = info.compiled;
    item["available"] = info.available;
    item["device_count"] = info.device_count;
    item["reason"] = info.reason;
    return item;
  };
  auto kernel_info_dict = [](const tensorstudio::BackendKernelInfo& info) {
    py::dict item;
    item["backend"] = info.backend;
    item["device_type"] = tensorstudio::device_type_name(info.type);
    item["op"] = info.op;
    py::list dtypes;
    for (const tensorstudio::DType dtype : info.dtypes) {
      dtypes.append(tensorstudio::dtype_name(dtype));
    }
    item["dtypes"] = dtypes;
    item["memory_space"] = tensorstudio::memory_space_name(info.memory_space);
    item["execution_mode"] = tensorstudio::kernel_execution_mode_name(info.execution_mode);
    item["forward"] = info.forward;
    item["backward"] = info.backward;
    item["available"] = info.available;
    item["deterministic"] = info.deterministic;
    item["priority"] = info.priority;
    item["reason"] = info.reason;
    return item;
  };
  auto allocator_info_dict = [](const tensorstudio::BackendAllocatorInfo& info) {
    py::dict item;
    item["device"] = info.device.str();
    item["memory_space"] = tensorstudio::memory_space_name(info.memory_space);
    item["can_allocate"] = info.can_allocate;
    item["supports_pinned_host"] = info.supports_pinned_host;
    item["supports_unified_memory"] = info.supports_unified_memory;
    item["supports_streams"] = info.supports_streams;
    item["alignment"] = info.alignment;
    item["reason"] = info.reason;
    return item;
  };
  auto runtime_info_dict = [](const tensorstudio::BackendRuntimeInfo& info) {
    py::dict item;
    item["device"] = info.device.str();
    item["backend"] = info.device.backend();
    item["device_type"] = tensorstudio::device_type_name(info.device.type());
    item["executable"] = info.executable;
    item["supports_eager"] = info.supports_eager;
    item["supports_graph"] = info.supports_graph;
    item["supports_compiler"] = info.supports_compiler;
    item["supports_streams"] = info.supports_streams;
    item["supports_events"] = info.supports_events;
    item["supports_peer_access"] = info.supports_peer_access;
    item["supports_host_fallback"] = info.supports_host_fallback;
    item["compiler"] = info.compiler;
    item["reason"] = info.reason;
    return item;
  };
  auto device_properties_dict = [](const tensorstudio::BackendDeviceProperties& info) {
    py::dict item;
    item["device"] = info.device.str();
    item["backend"] = info.device.backend();
    item["device_type"] = tensorstudio::device_type_name(info.device.type());
    item["name"] = info.name;
    item["vendor"] = info.vendor;
    item["architecture"] = info.architecture;
    item["total_memory_bytes"] = info.total_memory_bytes;
    item["free_memory_bytes"] = info.free_memory_bytes;
    item["max_threads_per_block"] = info.max_threads_per_block;
    item["max_shared_memory_bytes"] = info.max_shared_memory_bytes;
    item["supports_fp16"] = info.supports_fp16;
    item["supports_bf16"] = info.supports_bf16;
    item["supports_tf32"] = info.supports_tf32;
    item["unified_addressing"] = info.unified_addressing;
    item["available"] = info.available;
    item["reason"] = info.reason;
    return item;
  };
  auto logical_device_info_dict = [](const tensorstudio::LogicalDeviceInfo& info) {
    py::dict item;
    item["physical_device"] = info.physical_device.str();
    item["logical_device"] = info.logical_device.str();
    item["memory_limit_bytes"] = info.memory_limit_bytes;
    item["priority"] = info.priority;
    item["default_device"] = info.default_device;
    item["reason"] = info.reason;
    return item;
  };
  auto op_info_dict = [](const tensorstudio::BackendOpInfo& info) {
    py::dict item;
    item["op"] = info.op;
    item["min_inputs"] = info.min_inputs;
    item["max_inputs"] = info.max_inputs;
    item["differentiable"] = info.differentiable;
    item["shape_inference"] = info.shape_inference;
    item["stateful"] = info.stateful;
    item["category"] = info.category;
    item["notes"] = info.notes;
    return item;
  };
  auto dtypes_from_py = [](const std::vector<std::string>& names) {
    std::vector<tensorstudio::DType> dtypes;
    dtypes.reserve(names.size());
    for (const std::string& name : names) {
      dtypes.push_back(tensorstudio::dtype_from_string(name));
    }
    return dtypes;
  };
  auto devices_from_py = [device_from_py](py::object object) {
    std::vector<tensorstudio::Device> devices;
    if (object.is_none()) {
      return devices;
    }
    if (py::isinstance<py::str>(object) || py::isinstance<tensorstudio::Device>(object)) {
      devices.push_back(device_from_py(std::move(object)));
      return devices;
    }
    if (!py::isinstance<py::sequence>(object)) {
      throw tensorstudio::DeviceError("input_devices must be None, a device, or a sequence of devices");
    }
    for (py::handle item : py::reinterpret_borrow<py::sequence>(object)) {
      devices.push_back(device_from_py(py::reinterpret_borrow<py::object>(item)));
    }
    return devices;
  };
  auto storage_telemetry_dict = []() {
    const tensorstudio::StorageTelemetry telemetry = tensorstudio::storage_telemetry();
    py::dict item;
    item["total_allocations"] = telemetry.total_allocations;
    item["active_allocations"] = telemetry.active_allocations;
    item["peak_active_allocations"] = telemetry.peak_active_allocations;
    item["total_bytes_allocated"] = telemetry.total_bytes_allocated;
    item["active_bytes"] = telemetry.active_bytes;
    item["peak_active_bytes"] = telemetry.peak_active_bytes;
    return item;
  };

  module.def("device", &tensorstudio::parse_device, py::arg("spec") = "cpu");
  module.def("is_device_available", [device_from_py](py::object spec) {
    return tensorstudio::is_device_available(device_from_py(std::move(spec)));
  }, py::arg("device"));
  module.def("device_count", [device_from_py](py::object spec) {
    return tensorstudio::device_count(device_from_py(std::move(spec)));
  }, py::arg("device") = "cpu");
  module.def("available_devices", []() {
    py::list result;
    for (const tensorstudio::Device& device : tensorstudio::available_devices()) {
      result.append(device);
    }
    return result;
  });
  module.def("backend_info", [backend_info_dict]() {
    py::list result;
    for (const tensorstudio::BackendInfo& info : tensorstudio::backend_info()) {
      result.append(backend_info_dict(info));
    }
    return result;
  });
  module.def("backend_allocator_info", [device_from_py, allocator_info_dict](py::object device) {
    py::list result;
    if (device.is_none()) {
      for (const tensorstudio::BackendAllocatorInfo& info : tensorstudio::backend_allocator_info()) {
        result.append(allocator_info_dict(info));
      }
      return result;
    }
    result.append(allocator_info_dict(tensorstudio::backend_allocator_info(device_from_py(std::move(device)))));
    return result;
  }, py::arg("device") = py::none());
  module.def("backend_runtime_info", [device_from_py, runtime_info_dict](py::object device) {
    py::list result;
    if (device.is_none()) {
      for (const tensorstudio::BackendRuntimeInfo& info : tensorstudio::backend_runtime_info()) {
        result.append(runtime_info_dict(info));
      }
      return result;
    }
    result.append(runtime_info_dict(tensorstudio::backend_runtime_info(device_from_py(std::move(device)))));
    return result;
  }, py::arg("device") = py::none());
  module.def("backend_device_properties", [device_from_py, device_properties_dict](py::object device) {
    py::list result;
    if (device.is_none()) {
      for (const tensorstudio::BackendDeviceProperties& info : tensorstudio::backend_device_properties()) {
        result.append(device_properties_dict(info));
      }
      return result;
    }
    result.append(device_properties_dict(tensorstudio::backend_device_properties(device_from_py(std::move(device)))));
    return result;
  }, py::arg("device") = py::none());
  module.def("logical_device_info", [device_from_py, logical_device_info_dict](py::object device) {
    py::list result;
    if (device.is_none()) {
      for (const tensorstudio::LogicalDeviceInfo& info : tensorstudio::logical_device_info()) {
        result.append(logical_device_info_dict(info));
      }
      return result;
    }
    for (const tensorstudio::LogicalDeviceInfo& info : tensorstudio::logical_device_info(device_from_py(std::move(device)))) {
      result.append(logical_device_info_dict(info));
    }
    return result;
  }, py::arg("device") = py::none());
  module.def("backend_op_info", [op_info_dict](py::object op) -> py::object {
    if (op.is_none()) {
      py::list result;
      for (const tensorstudio::BackendOpInfo& info : tensorstudio::backend_op_info()) {
        result.append(op_info_dict(info));
      }
      return result;
    }
    return op_info_dict(tensorstudio::backend_op_info(py::cast<std::string>(op)));
  }, py::arg("op") = py::none());
  module.def("backend_kernel_info", [device_from_py, kernel_info_dict](py::object device) {
    py::list result;
    if (device.is_none()) {
      for (const tensorstudio::BackendKernelInfo& info : tensorstudio::backend_kernel_info()) {
        result.append(kernel_info_dict(info));
      }
      return result;
    }
    for (const tensorstudio::BackendKernelInfo& info : tensorstudio::backend_kernel_info(device_from_py(std::move(device)))) {
      result.append(kernel_info_dict(info));
    }
    return result;
  }, py::arg("device") = py::none());
  module.def("backend_supports_kernel", [device_from_py](const std::string& op, py::object device, const std::string& dtype) {
    return tensorstudio::backend_supports_kernel(
        device_from_py(std::move(device)),
        op,
        tensorstudio::dtype_from_string(dtype));
  }, py::arg("op"), py::arg("device") = "cpu", py::arg("dtype") = "float32");
  module.def("kernel_placement_info", [device_from_py](const std::string& op, py::object device, const std::string& dtype) {
    const tensorstudio::KernelPlacementInfo info = tensorstudio::kernel_placement_info(
        device_from_py(std::move(device)),
        op,
        tensorstudio::dtype_from_string(dtype));
    py::dict result;
    result["op"] = info.op;
    result["dtype"] = tensorstudio::dtype_name(info.dtype);
    result["requested"] = info.requested.str();
    result["selected"] = info.selected.str();
    result["supported"] = info.supported;
    result["fallback_to_cpu"] = info.fallback_to_cpu;
    result["requires_transfer"] = info.requires_transfer;
    result["reason"] = info.reason;
    return result;
  }, py::arg("op"), py::arg("device") = "cpu", py::arg("dtype") = "float32");
  module.def("backend_execution_plan", [device_from_py, devices_from_py](const std::string& op, py::object device, const std::string& dtype, py::object input_devices) {
    const tensorstudio::BackendExecutionPlan info = tensorstudio::backend_execution_plan(
        device_from_py(std::move(device)),
        op,
        tensorstudio::dtype_from_string(dtype),
        devices_from_py(std::move(input_devices)));
    py::dict result;
    result["op"] = info.op;
    result["dtype"] = tensorstudio::dtype_name(info.dtype);
    result["requested"] = info.requested.str();
    result["selected"] = info.selected.str();
    py::list inputs;
    for (const tensorstudio::Device& input : info.input_devices) {
      inputs.append(input.str());
    }
    result["input_devices"] = inputs;
    py::list transfers;
    for (const tensorstudio::DeviceTransferInfo& transfer : info.input_transfers) {
      py::dict transfer_item;
      transfer_item["source"] = transfer.source.str();
      transfer_item["target"] = transfer.target.str();
      transfer_item["supported"] = transfer.supported;
      transfer_item["direct"] = transfer.direct;
      transfer_item["reason"] = transfer.reason;
      transfers.append(transfer_item);
    }
    result["input_transfers"] = transfers;
    result["executable"] = info.executable;
    result["uses_fallback"] = info.uses_fallback;
    result["requires_transfer"] = info.requires_transfer;
    result["stream_required"] = info.stream_required;
    result["graph_compatible"] = info.graph_compatible;
    result["reason"] = info.reason;
    return result;
  }, py::arg("op"), py::arg("device") = "cpu", py::arg("dtype") = "float32", py::arg("input_devices") = py::none());
  module.def("device_transfer_info", [device_from_py](py::object source, py::object target) {
    const tensorstudio::DeviceTransferInfo info = tensorstudio::device_transfer_info(
        device_from_py(std::move(source)),
        device_from_py(std::move(target)));
    py::dict result;
    result["source"] = info.source.str();
    result["target"] = info.target.str();
    result["supported"] = info.supported;
    result["direct"] = info.direct;
    result["reason"] = info.reason;
    return result;
  }, py::arg("source"), py::arg("target"));
  module.def("can_transfer", [device_from_py](py::object source, py::object target) {
    return tensorstudio::can_transfer(
        device_from_py(std::move(source)),
        device_from_py(std::move(target)));
  }, py::arg("source"), py::arg("target"));
  module.def("register_backend", [](const std::string& name, bool compiled, bool available, int64_t device_count, const std::string& reason) {
    tensorstudio::BackendInfo info;
    info.name = name;
    info.compiled = compiled;
    info.available = available;
    info.device_count = device_count;
    info.reason = reason;
    tensorstudio::register_backend(std::move(info));
  }, py::arg("name"), py::arg("compiled") = true, py::arg("available") = false, py::arg("device_count") = 0, py::arg("reason") = "");
  module.def("unregister_backend", &tensorstudio::unregister_backend, py::arg("name"));
  module.def("register_backend_kernel", [dtypes_from_py](
      const std::string& backend,
      const std::string& op,
      const std::vector<std::string>& dtypes,
      bool forward,
      bool backward,
      bool available,
      const std::string& reason,
      const std::string& memory_space,
      const std::string& execution_mode,
      bool deterministic,
      int64_t priority) {
    tensorstudio::BackendKernelInfo info;
    info.backend = backend;
    info.op = op;
    info.dtypes = dtypes_from_py(dtypes);
    info.memory_space = tensorstudio::memory_space_from_string(memory_space);
    info.execution_mode = tensorstudio::kernel_execution_mode_from_string(execution_mode);
    info.forward = forward;
    info.backward = backward;
    info.available = available;
    info.deterministic = deterministic;
    info.priority = priority;
    info.reason = reason;
    tensorstudio::register_backend_kernel(std::move(info));
  }, py::arg("backend"), py::arg("op"), py::arg("dtypes"), py::arg("forward") = true, py::arg("backward") = false, py::arg("available") = true, py::arg("reason") = "", py::arg("memory_space") = "device", py::arg("execution_mode") = "sync", py::arg("deterministic") = true, py::arg("priority") = 50);
  module.def("storage_telemetry", storage_telemetry_dict);
  module.def("reset_storage_telemetry", &tensorstudio::reset_storage_telemetry);

  tensorstudio::bindings::bind_tensor(module);
  tensorstudio::bindings::bind_ops(module);
  tensorstudio::bindings::bind_autograd(module);
  tensorstudio::bindings::bind_nn(module);
  tensorstudio::bindings::bind_optim(module);
}
