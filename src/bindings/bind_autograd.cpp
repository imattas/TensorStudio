#include "bindings.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

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
  module.def(
      "_set_history",
      [](Tensor& output, std::vector<Tensor> parents, py::function backward_fn) {
        set_history(
            output,
            std::move(parents),
            [backward_fn = std::move(backward_fn)](const Tensor& grad) mutable {
              py::gil_scoped_acquire gil;
              py::object result = backward_fn(grad);
              return result.cast<std::vector<Tensor>>();
            });
      },
      py::arg("output"),
      py::arg("parents"),
      py::arg("backward_fn"));
}

}  // namespace tensorstudio::bindings
