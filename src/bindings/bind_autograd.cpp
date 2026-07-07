#include "bindings.hpp"

#include <optional>

#include <pybind11/pybind11.h>

#include "tensorstudio/autograd.hpp"

namespace tensorstudio::bindings {

void bind_autograd(py::module_& module) {
  module.def("_grad_enabled", &grad_enabled);
  module.def("_set_grad_enabled", &set_grad_enabled, py::arg("enabled"));
  module.def(
      "backward",
      [](Tensor& output, py::object gradient, bool retain_graph) {
        if (gradient.is_none()) {
          backward(output, std::nullopt, retain_graph);
        } else {
          backward(output, ensure_tensor(gradient), retain_graph);
        }
      },
      py::arg("output"),
      py::arg("gradient") = py::none(),
      py::arg("retain_graph") = false);
}

}  // namespace tensorstudio::bindings
