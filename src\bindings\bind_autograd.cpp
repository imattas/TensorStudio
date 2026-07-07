#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "tensorstudio/autograd.hpp"

namespace tensorstudio::bindings {

void bind_autograd(py::module_& module) {
  module.def("_grad_enabled", &grad_enabled);
  module.def("_set_grad_enabled", &set_grad_enabled, py::arg("enabled"));
  module.def(
      "backward",
      [](Tensor& output, py::object gradient) {
        if (gradient.is_none()) {
          backward(output);
        } else {
          backward(output, ensure_tensor(gradient));
        }
      },
      py::arg("output"),
      py::arg("gradient") = py::none());
}

}  // namespace tensorstudio::bindings
