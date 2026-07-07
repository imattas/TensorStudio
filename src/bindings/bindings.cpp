#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "tensorstudio/device.hpp"
#include "tensorstudio/errors.hpp"
#include "tensorstudio/perf.hpp"
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

  module.def("device", &tensorstudio::parse_device, py::arg("spec") = "cpu");
  module.def("is_device_available", [](py::object spec) {
    return tensorstudio::is_device_available(tensorstudio::bindings::device_from_py(std::move(spec)));
  }, py::arg("device"));
  module.def("device_count", [](const std::string& type) {
    return tensorstudio::device_count(tensorstudio::parse_device(type).type());
  }, py::arg("type") = "cpu");
  module.def("available_devices", []() {
    py::list result;
    for (const tensorstudio::Device& device : tensorstudio::available_devices()) {
      result.append(device);
    }
    return result;
  });
  module.def("backend_info", []() {
    py::list result;
    for (const tensorstudio::BackendInfo& info : tensorstudio::backend_info()) {
      py::dict item;
      item["name"] = info.name;
      item["compiled"] = info.compiled;
      item["available"] = info.available;
      item["device_count"] = info.device_count;
      item["reason"] = info.reason;
      result.append(item);
    }
    return result;
  });

  module.def("get_num_threads", &tensorstudio::perf::get_num_threads);
  module.def("set_num_threads", &tensorstudio::perf::set_num_threads, py::arg("count"));
  module.def("performance_info", []() {
    const tensorstudio::perf::PerformanceInfo info = tensorstudio::perf::performance_info();
    py::dict result;
    result["num_threads"] = info.num_threads;
    result["threads_enabled"] = info.threads_enabled;
    result["storage_pool_enabled"] = info.storage_pool_enabled;
    result["blas_enabled"] = info.blas_enabled;
    result["simd_enabled"] = info.simd_enabled;
    result["simd_level"] = info.simd_level;
    return result;
  });

  tensorstudio::bindings::bind_tensor(module);
  tensorstudio::bindings::bind_ops(module);
  tensorstudio::bindings::bind_autograd(module);
  tensorstudio::bindings::bind_nn(module);
  tensorstudio::bindings::bind_optim(module);
}
