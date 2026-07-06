#include "bindings.hpp"

#include <algorithm>
#include <functional>
#include <optional>
#include <sstream>
#include <string>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "tensorstudio/autograd.hpp"
#include "tensorstudio/dtype.hpp"
#include "tensorstudio/errors.hpp"
#include "tensorstudio/ops.hpp"
#include "tensorstudio/random.hpp"

namespace tensorstudio::bindings {
namespace {

struct ParsedData {
  Shape shape{};
  std::vector<double> values{};
  bool saw_float{false};
  bool saw_int{false};
  bool saw_bool{false};
};

bool is_sequence_like(py::handle object) {
  if (py::isinstance<py::str>(object) || py::isinstance<py::bytes>(object)) {
    return false;
  }
  return PySequence_Check(object.ptr()) != 0 && !py::isinstance<py::array>(object);
}

void parse_nested(py::handle object, std::size_t depth, ParsedData& parsed) {
  if (is_sequence_like(object)) {
    py::sequence sequence = py::reinterpret_borrow<py::sequence>(object);
    const auto length = static_cast<int64_t>(sequence.size());
    if (parsed.shape.size() <= depth) {
      parsed.shape.push_back(length);
    } else if (parsed.shape[depth] != length) {
      throw ShapeError("ragged nested sequences cannot be converted to a tensor");
    }
    for (py::handle item : sequence) {
      parse_nested(item, depth + 1, parsed);
    }
    return;
  }

  if (py::isinstance<py::bool_>(object)) {
    parsed.saw_bool = true;
    parsed.values.push_back(py::cast<bool>(object) ? 1.0 : 0.0);
    return;
  }
  if (py::isinstance<py::float_>(object)) {
    parsed.saw_float = true;
    parsed.values.push_back(py::cast<double>(object));
    return;
  }
  if (py::isinstance<py::int_>(object)) {
    parsed.saw_int = true;
    parsed.values.push_back(static_cast<double>(py::cast<int64_t>(object)));
    return;
  }
  throw DTypeError("tensor data must contain numeric Python scalars");
}

DType infer_dtype(const ParsedData& parsed) {
  if (parsed.saw_float) {
    return DType::Float32;
  }
  if (parsed.saw_int) {
    return DType::Int64;
  }
  if (parsed.saw_bool) {
    return DType::Bool;
  }
  return DType::Float32;
}

py::object scalar_from_value(double value, DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return py::float_(static_cast<float>(value));
    case DType::Float64:
      return py::float_(value);
    case DType::Int32:
      return py::int_(static_cast<int32_t>(value));
    case DType::Int64:
      return py::int_(static_cast<int64_t>(value));
    case DType::Bool:
      return py::bool_(value != 0.0);
  }
  throw DTypeError("unknown dtype");
}

py::list nested_list_from_tensor(const Tensor& tensor, std::size_t dim, int64_t& linear) {
  py::list result;
  if (dim == tensor.shape().size()) {
    result.append(scalar_from_value(tensor.value_at_logical(linear++), tensor.dtype()));
    return result;
  }
  for (int64_t i = 0; i < tensor.shape()[dim]; ++i) {
    if (dim + 1 == tensor.shape().size()) {
      result.append(scalar_from_value(tensor.value_at_logical(linear++), tensor.dtype()));
    } else {
      result.append(nested_list_from_tensor(tensor, dim + 1, linear));
    }
  }
  return result;
}

std::vector<py::ssize_t> ssize_shape(const Shape& shape) {
  std::vector<py::ssize_t> result;
  result.reserve(shape.size());
  for (const auto dim : shape) {
    result.push_back(static_cast<py::ssize_t>(dim));
  }
  return result;
}

py::dtype numpy_dtype(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return py::dtype::of<float>();
    case DType::Float64:
      return py::dtype::of<double>();
    case DType::Int32:
      return py::dtype::of<int32_t>();
    case DType::Int64:
      return py::dtype::of<int64_t>();
    case DType::Bool:
      return py::dtype::of<bool>();
  }
  throw DTypeError("unknown dtype");
}

std::string dtype_name_from_numpy(const py::array& array) {
  return py::str(array.attr("dtype"));
}

py::object none_if_undefined(const Tensor& tensor) {
  if (!tensor.defined()) {
    return py::none();
  }
  return py::cast(tensor);
}

py::tuple shape_tuple(const Shape& shape) {
  py::tuple result(shape.size());
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result[i] = py::int_(shape[i]);
  }
  return result;
}

Tensor binary_tensor_op(const Tensor& left, py::object right, const std::function<Tensor(const Tensor&, const Tensor&)>& op) {
  return op(left, ensure_tensor(right));
}

Tensor reverse_binary_tensor_op(
    py::object left,
    const Tensor& right,
    const std::function<Tensor(const Tensor&, const Tensor&)>& op) {
  return op(ensure_tensor(left), right);
}

}  // namespace

DType dtype_from_py(py::object object, DType fallback) {
  if (object.is_none()) {
    return fallback;
  }
  return dtype_from_string(py::cast<std::string>(object));
}

Shape shape_from_py(py::object object) {
  if (py::isinstance<py::int_>(object)) {
    return Shape{py::cast<int64_t>(object)};
  }
  if (!is_sequence_like(object)) {
    throw ShapeError("shape must be an int or a sequence of ints");
  }
  Shape shape;
  for (py::handle item : py::reinterpret_borrow<py::sequence>(object)) {
    shape.push_back(py::cast<int64_t>(item));
  }
  validate_shape(shape);
  return shape;
}

Tensor tensor_from_numpy(py::array array, bool requires_grad) {
  py::array contiguous = py::reinterpret_borrow<py::array>(
      py::module_::import("numpy").attr("ascontiguousarray")(array));
  const DType dtype = dtype_from_string(dtype_name_from_numpy(contiguous));
  Shape shape;
  for (py::handle dim : py::reinterpret_borrow<py::tuple>(contiguous.attr("shape"))) {
    shape.push_back(py::cast<int64_t>(dim));
  }
  py::array flat = py::reinterpret_borrow<py::array>(contiguous.attr("ravel")());
  std::vector<double> values;
  values.reserve(static_cast<std::size_t>(flat.size()));
  for (py::ssize_t i = 0; i < flat.size(); ++i) {
    values.push_back(py::cast<double>(flat.attr("__getitem__")(i)));
  }
  return Tensor::from_flat_values(values, shape, dtype, requires_grad);
}

Tensor tensor_from_py(py::object data, py::object dtype_object, bool requires_grad) {
  if (py::isinstance<Tensor>(data)) {
    Tensor tensor = py::cast<Tensor>(data);
    if (!dtype_object.is_none() && dtype_from_py(dtype_object) != tensor.dtype()) {
      Tensor converted(tensor.shape(), dtype_from_py(dtype_object), requires_grad);
      converted.copy_from(tensor);
      return converted;
    }
    tensor.set_requires_grad(requires_grad || tensor.requires_grad());
    return tensor;
  }
  if (py::isinstance<py::array>(data)) {
    Tensor result = tensor_from_numpy(py::cast<py::array>(data), requires_grad);
    if (!dtype_object.is_none() && dtype_from_py(dtype_object) != result.dtype()) {
      Tensor converted(result.shape(), dtype_from_py(dtype_object), requires_grad);
      converted.copy_from(result);
      return converted;
    }
    return result;
  }

  ParsedData parsed;
  parse_nested(data, 0, parsed);
  const DType dtype = dtype_from_py(dtype_object, infer_dtype(parsed));
  return Tensor::from_flat_values(parsed.values, parsed.shape, dtype, requires_grad);
}

Tensor ensure_tensor(py::object object) {
  if (py::isinstance<Tensor>(object)) {
    return py::cast<Tensor>(object);
  }
  return tensor_from_py(std::move(object), py::none(), false);
}

py::array tensor_to_numpy(const Tensor& tensor) {
  py::array array(numpy_dtype(tensor.dtype()), ssize_shape(tensor.shape()));
  py::array flat = py::reinterpret_borrow<py::array>(array.attr("ravel")());
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    flat.attr("__setitem__")(i, scalar_from_value(tensor.value_at_logical(i), tensor.dtype()));
  }
  return array;
}

py::object tensor_to_py_scalar(const Tensor& tensor) {
  if (tensor.numel() != 1) {
    throw ShapeError("item() is only valid for tensors with one element");
  }
  return scalar_from_value(tensor.value_at_logical(0), tensor.dtype());
}

py::object tensor_to_py_list(const Tensor& tensor) {
  if (tensor.shape().empty()) {
    return tensor_to_py_scalar(tensor);
  }
  int64_t linear = 0;
  return nested_list_from_tensor(tensor, 0, linear);
}

void bind_tensor(py::module_& module) {
  py::class_<Tensor>(module, "Tensor")
      .def("__repr__", &Tensor::repr)
      .def_property(
          "requires_grad",
          &Tensor::requires_grad,
          &Tensor::set_requires_grad,
          "Whether this tensor participates in reverse-mode autodiff.")
      .def_property_readonly("grad", [](const Tensor& self) { return none_if_undefined(self.grad()); })
      .def_property_readonly("shape", [](const Tensor& self) { return shape_tuple(self.shape()); })
      .def_property_readonly("dtype", [](const Tensor& self) { return dtype_name(self.dtype()); })
      .def_property_readonly("ndim", &Tensor::ndim)
      .def_property_readonly("size", &Tensor::size)
      .def_property_readonly("T", [](const Tensor& self) { return transpose(self); })
      .def("numpy", &tensor_to_numpy)
      .def("tolist", &tensor_to_py_list)
      .def("item", &tensor_to_py_scalar)
      .def("reshape", [](const Tensor& self, py::args args) {
        if (args.size() == 1 && !py::isinstance<py::int_>(args[0])) {
          return reshape(self, shape_from_py(py::reinterpret_borrow<py::object>(args[0])));
        }
        Shape shape;
        for (py::handle arg : args) {
          shape.push_back(py::cast<int64_t>(arg));
        }
        return reshape(self, shape);
      })
      .def("flatten", &flatten)
      .def("transpose", &transpose)
      .def("sum", &sum)
      .def("mean", &mean)
      .def("relu", &relu)
      .def("sigmoid", &sigmoid)
      .def("tanh", &tanh)
      .def("exp", &exp)
      .def("log", &log)
      .def("backward", [](Tensor& self, py::object gradient) {
        if (gradient.is_none()) {
          backward(self);
          return;
        }
        backward(self, ensure_tensor(gradient));
      }, py::arg("gradient") = py::none())
      .def("zero_grad", &Tensor::zero_grad)
      .def("_assign", &Tensor::copy_from)
      .def("_add_scaled_", &Tensor::add_scaled_)
      .def("__add__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, add); }, py::is_operator())
      .def("__radd__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, add); }, py::is_operator())
      .def("__sub__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, sub); }, py::is_operator())
      .def("__rsub__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, sub); }, py::is_operator())
      .def("__mul__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, mul); }, py::is_operator())
      .def("__rmul__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, mul); }, py::is_operator())
      .def("__truediv__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, div); }, py::is_operator())
      .def("__rtruediv__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, div); }, py::is_operator())
      .def("__neg__", &neg, py::is_operator())
      .def("__pow__", [](const Tensor& self, double exponent) { return pow(self, exponent); }, py::is_operator())
      .def("__matmul__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, matmul); }, py::is_operator())
      .def("__rmatmul__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, matmul); }, py::is_operator())
      .def(py::pickle(
          [](const Tensor& self) {
            return py::make_tuple(self.shape(), dtype_name(self.dtype()), self.to_flat_vector(), self.requires_grad());
          },
          [](py::tuple state) {
            if (state.size() != 4) {
              throw TensorStudioError("invalid Tensor pickle state");
            }
            return Tensor::from_flat_values(
                state[2].cast<std::vector<double>>(),
                state[0].cast<Shape>(),
                dtype_from_string(state[1].cast<std::string>()),
                state[3].cast<bool>());
          }));

  module.def(
      "tensor",
      [](py::object data, py::object dtype, bool requires_grad) {
        return tensor_from_py(std::move(data), std::move(dtype), requires_grad);
      },
      py::arg("data"),
      py::arg("dtype") = py::none(),
      py::arg("requires_grad") = false);
  module.def(
      "from_numpy",
      [](py::array array, bool requires_grad) { return tensor_from_numpy(array, requires_grad); },
      py::arg("array"),
      py::arg("requires_grad") = false);
  module.def("zeros", [](py::object shape, py::object dtype) { return zeros(shape_from_py(shape), dtype_from_py(dtype)); }, py::arg("shape"), py::arg("dtype") = "float32");
  module.def("ones", [](py::object shape, py::object dtype) { return ones(shape_from_py(shape), dtype_from_py(dtype)); }, py::arg("shape"), py::arg("dtype") = "float32");
  module.def("full", [](py::object shape, double fill_value, py::object dtype) { return full(shape_from_py(shape), fill_value, dtype_from_py(dtype)); }, py::arg("shape"), py::arg("fill_value"), py::arg("dtype") = "float32");
  module.def(
      "randn",
      [](py::object shape, py::object dtype, py::object seed) {
        std::optional<uint64_t> cpp_seed;
        if (!seed.is_none()) {
          cpp_seed = py::cast<uint64_t>(seed);
        }
        return randn(shape_from_py(shape), dtype_from_py(dtype), cpp_seed);
      },
      py::arg("shape"),
      py::arg("dtype") = "float32",
      py::arg("seed") = py::none());
  module.def(
      "arange",
      [](double start, py::object stop, double step, py::object dtype) {
        if (stop.is_none()) {
          return arange(0.0, start, step, dtype_from_py(dtype));
        }
        return arange(start, py::cast<double>(stop), step, dtype_from_py(dtype));
      },
      py::arg("start"),
      py::arg("stop") = py::none(),
      py::arg("step") = 1.0,
      py::arg("dtype") = "float32");
}

}  // namespace tensorstudio::bindings
