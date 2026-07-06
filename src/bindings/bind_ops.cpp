#include "bindings.hpp"

#include <pybind11/pybind11.h>

#include "tensorstudio/ops.hpp"

namespace tensorstudio::bindings {

void bind_ops(py::module_& module) {
  module.def("add", [](const Tensor& left, py::object right) { return add(left, ensure_tensor(right)); });
  module.def("sub", [](const Tensor& left, py::object right) { return sub(left, ensure_tensor(right)); });
  module.def("mul", [](const Tensor& left, py::object right) { return mul(left, ensure_tensor(right)); });
  module.def("div", [](const Tensor& left, py::object right) { return div(left, ensure_tensor(right)); });
  module.def("neg", &neg);
  module.def("pow", &pow, py::arg("input"), py::arg("exponent"));
  module.def("matmul", [](const Tensor& left, py::object right) { return matmul(left, ensure_tensor(right)); });
  module.def("sum", &sum);
  module.def("mean", &mean);
  module.def("relu", &relu);
  module.def("sigmoid", &sigmoid);
  module.def("tanh", &tanh);
  module.def("exp", &exp);
  module.def("log", &log);
  module.def("reshape", [](const Tensor& input, py::object shape) { return reshape(input, shape_from_py(shape)); });
  module.def("flatten", &flatten);
  module.def("transpose", &transpose);
}

}  // namespace tensorstudio::bindings
