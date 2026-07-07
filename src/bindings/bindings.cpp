#include "bindings.hpp"

#include <pybind11/pybind11.h>

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
