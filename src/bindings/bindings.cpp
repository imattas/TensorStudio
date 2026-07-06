#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "tensorstudio/errors.hpp"
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

  tensorstudio::bindings::bind_tensor(module);
  tensorstudio::bindings::bind_ops(module);
  tensorstudio::bindings::bind_autograd(module);
  tensorstudio::bindings::bind_nn(module);
  tensorstudio::bindings::bind_optim(module);
}
