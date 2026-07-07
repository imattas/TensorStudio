#pragma once

#include <functional>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio::bindings {

namespace py = pybind11;

Tensor tensor_from_py(py::object data, py::object dtype, bool requires_grad);
Tensor tensor_from_numpy(py::array array, bool requires_grad);
py::array tensor_to_numpy(const Tensor& tensor);
py::object tensor_to_py_scalar(const Tensor& tensor);
py::object tensor_to_py_list(const Tensor& tensor);
Tensor ensure_tensor(py::object object);
Shape shape_from_py(py::object object);
DType dtype_from_py(py::object object, DType fallback = DType::Float32);
Tensor reduce_from_py(
    const Tensor& input,
    py::object axis,
    bool keepdims,
    const std::string& op_name,
    const std::function<Tensor(const Tensor&)>& reduce_all,
    const std::function<Tensor(const Tensor&, int64_t, bool)>& reduce_axis);

void bind_tensor(py::module_& module);
void bind_ops(py::module_& module);
void bind_autograd(py::module_& module);
void bind_nn(py::module_& module);
void bind_optim(py::module_& module);

}  // namespace tensorstudio::bindings
