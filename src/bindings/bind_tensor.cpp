#include "bindings.hpp"

#include <algorithm>
#include <cstring>
#include <functional>
#include <optional>
#include <sstream>
#include <set>
#include <string>

#include <pybind11/operators.h>
#include <pybind11/stl.h>

#include "tensorstudio/autograd.hpp"
#include "tensorstudio/dtype.hpp"
#include "tensorstudio/errors.hpp"
#include "tensorstudio/ops.hpp"
#include "tensorstudio/random.hpp"
#include "tensorstudio/shape.hpp"

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

int64_t normalize_reduction_axis(const Tensor& input, int64_t axis, const std::string& op_name) {
  const int64_t ndim = input.ndim();
  const int64_t original_axis = axis;
  if (axis < 0) {
    axis += ndim;
  }
  if (axis < 0 || axis >= ndim) {
    throw ShapeError(
        op_name + " axis " + std::to_string(original_axis) + " is out of range for tensor with shape " +
        shape_to_string(input.shape()));
  }
  return axis;
}

std::vector<int64_t> reduction_axes_from_py(const Tensor& input, py::object axis, const std::string& op_name) {
  if (py::isinstance<py::int_>(axis)) {
    return {normalize_reduction_axis(input, py::cast<int64_t>(axis), op_name)};
  }
  if (!is_sequence_like(axis)) {
    throw ShapeError(op_name + " axis must be None, an int, or a sequence of ints");
  }

  std::vector<int64_t> axes;
  std::set<int64_t> seen;
  for (py::handle item : py::reinterpret_borrow<py::sequence>(axis)) {
    if (!py::isinstance<py::int_>(item)) {
      throw ShapeError(op_name + " axis entries must be integers");
    }
    const int64_t normalized = normalize_reduction_axis(input, py::cast<int64_t>(item), op_name);
    if (!seen.insert(normalized).second) {
      throw ShapeError(op_name + " received duplicate reduction axis " + std::to_string(normalized));
    }
    axes.push_back(normalized);
  }
  std::sort(axes.begin(), axes.end(), std::greater<int64_t>());
  return axes;
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
  Tensor tensor(shape, dtype, requires_grad);
  const auto bytes = static_cast<std::size_t>(tensor.numel()) * dtype_size(dtype);
  if (static_cast<std::size_t>(contiguous.nbytes()) != bytes) {
    throw TensorStudioError("NumPy array byte size does not match TensorStudio dtype and shape");
  }
  std::memcpy(tensor.impl()->storage->data(), contiguous.data(), bytes);
  return tensor;
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

Tensor reduce_from_py(
    const Tensor& input,
    py::object axis,
    bool keepdims,
    const std::string& op_name,
    const std::function<Tensor(const Tensor&)>& reduce_all,
    const std::function<Tensor(const Tensor&, int64_t, bool)>& reduce_axis) {
  if (axis.is_none()) {
    return reduce_all(input);
  }

  const std::vector<int64_t> axes = reduction_axes_from_py(input, std::move(axis), op_name);
  if (axes.empty()) {
    return input;
  }

  Tensor result = input;
  for (const int64_t normalized_axis : axes) {
    result = reduce_axis(result, normalized_axis, keepdims);
  }
  return result;
}

Tensor arg_reduce_from_py(
    const Tensor& input,
    py::object axis,
    bool keepdims,
    const std::string& op_name,
    const std::function<Tensor(const Tensor&, bool)>& reduce_all,
    const std::function<Tensor(const Tensor&, int64_t, bool)>& reduce_axis) {
  if (axis.is_none()) {
    return reduce_all(input, keepdims);
  }
  if (!py::isinstance<py::int_>(axis)) {
    throw ShapeError(op_name + " axis must be None or an int");
  }
  return reduce_axis(input, normalize_reduction_axis(input, py::cast<int64_t>(axis), op_name), keepdims);
}

Shape shape_from_varargs(py::args args, const std::string& op_name) {
  if (args.size() == 1 && !py::isinstance<py::int_>(args[0])) {
    return shape_from_py(py::reinterpret_borrow<py::object>(args[0]));
  }
  Shape shape;
  shape.reserve(args.size());
  for (py::handle arg : args) {
    if (!py::isinstance<py::int_>(arg) || py::isinstance<py::bool_>(arg)) {
      throw ShapeError(op_name + " axes must be integers");
    }
    shape.push_back(py::cast<int64_t>(arg));
  }
  return shape;
}

Tensor transpose_from_args(const Tensor& self, py::args args) {
  if (args.empty()) {
    return transpose(self);
  }
  if (args.size() == 2) {
    return transpose(self, py::cast<int64_t>(args[0]), py::cast<int64_t>(args[1]));
  }
  throw ShapeError("transpose expects no axes or exactly two axes");
}

bool is_ellipsis(py::handle object) {
  return object.ptr() == Py_Ellipsis;
}

bool is_slice(py::handle object) {
  return PySlice_Check(object.ptr()) != 0;
}

std::vector<TensorIndex> indices_from_py(const Tensor& input, py::object key) {
  std::vector<py::object> raw_items;
  if (py::isinstance<py::tuple>(key)) {
    py::tuple tuple = py::reinterpret_borrow<py::tuple>(key);
    raw_items.reserve(tuple.size());
    for (py::handle item : tuple) {
      raw_items.push_back(py::reinterpret_borrow<py::object>(item));
    }
  } else {
    raw_items.push_back(std::move(key));
  }

  int64_t ellipsis_count = 0;
  int64_t consuming_count = 0;
  for (const py::object& item : raw_items) {
    if (is_ellipsis(item)) {
      ++ellipsis_count;
      continue;
    }
    if (item.is_none()) {
      continue;
    }
    if (py::isinstance<py::bool_>(item)) {
      throw ShapeError("boolean tensor indexing is not supported; use comparison masks with where");
    }
    if (py::isinstance<py::int_>(item) || is_slice(item)) {
      ++consuming_count;
      continue;
    }
    throw ShapeError("unsupported tensor index type; use integers, slices, ellipsis, or None/newaxis");
  }

  if (ellipsis_count > 1) {
    throw ShapeError("an index can contain at most one ellipsis");
  }
  if (consuming_count > input.ndim()) {
    throw ShapeError(
        "too many indices for tensor with shape " + shape_to_string(input.shape()));
  }

  std::vector<TensorIndex> indices;
  int64_t input_dim = 0;
  for (const py::object& item : raw_items) {
    if (is_ellipsis(item)) {
      const int64_t missing = input.ndim() - consuming_count;
      for (int64_t i = 0; i < missing; ++i) {
        const int64_t dim_size = input.shape()[static_cast<std::size_t>(input_dim)];
        indices.push_back(TensorIndex::slice(0, dim_size, 1));
        ++input_dim;
      }
      continue;
    }
    if (item.is_none()) {
      indices.push_back(TensorIndex::new_axis());
      continue;
    }
    if (py::isinstance<py::int_>(item) && !py::isinstance<py::bool_>(item)) {
      if (input_dim >= input.ndim()) {
        throw ShapeError("too many indices for tensor with shape " + shape_to_string(input.shape()));
      }
      indices.push_back(TensorIndex::at(py::cast<int64_t>(item)));
      ++input_dim;
      continue;
    }
    if (is_slice(item)) {
      if (input_dim >= input.ndim()) {
        throw ShapeError("too many indices for tensor with shape " + shape_to_string(input.shape()));
      }
      const auto dim_size = static_cast<Py_ssize_t>(input.shape()[static_cast<std::size_t>(input_dim)]);
      Py_ssize_t start = 0;
      Py_ssize_t stop = 0;
      Py_ssize_t step = 1;
      Py_ssize_t length = 0;
      if (PySlice_GetIndicesEx(item.ptr(), dim_size, &start, &stop, &step, &length) != 0) {
        PyErr_Clear();
        throw ShapeError("invalid slice for tensor axis " + std::to_string(input_dim));
      }
      indices.push_back(TensorIndex::slice(
          static_cast<int64_t>(start),
          static_cast<int64_t>(length),
          static_cast<int64_t>(step)));
      ++input_dim;
      continue;
    }
  }
  return indices;
}

py::array tensor_to_numpy(const Tensor& tensor) {
  py::array array(numpy_dtype(tensor.dtype()), ssize_shape(tensor.shape()));
  if (tensor.is_contiguous()) {
    const auto bytes = static_cast<std::size_t>(tensor.numel()) * dtype_size(tensor.dtype());
    const auto* source =
        tensor.impl()->storage->data() + static_cast<std::size_t>(tensor.offset()) * dtype_size(tensor.dtype());
    std::memcpy(array.mutable_data(), source, bytes);
    return array;
  }
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
      .def_property_readonly("strides", [](const Tensor& self) { return shape_tuple(self.strides()); })
      .def_property_readonly("dtype", [](const Tensor& self) { return dtype_name(self.dtype()); })
      .def_property_readonly("device", [](const Tensor& self) { return self.device().str(); })
      .def_property_readonly("ndim", &Tensor::ndim)
      .def_property_readonly("size", &Tensor::size)
      .def_property_readonly("is_contiguous", &Tensor::is_contiguous)
      .def_property_readonly("T", [](const Tensor& self) { return transpose(self); })
      .def("__getitem__", [](const Tensor& self, py::object key) {
        return index(self, indices_from_py(self, std::move(key)));
      }, py::is_operator())
      .def("numpy", &tensor_to_numpy)
      .def("tolist", &tensor_to_py_list)
      .def("item", &tensor_to_py_scalar)
      .def("clone", &Tensor::clone)
      .def("detach", &Tensor::detach)
      .def("reshape", [](const Tensor& self, py::args args) {
        return reshape(self, shape_from_varargs(args, "reshape"));
      })
      .def("flatten", &flatten)
      .def("transpose", &transpose_from_args)
      .def("permute", [](const Tensor& self, py::args args) {
        return permute(self, shape_from_varargs(args, "permute"));
      })
      .def("squeeze", [](const Tensor& self, py::object axis) {
        if (axis.is_none()) {
          return squeeze(self);
        }
        return squeeze(self, py::cast<int64_t>(axis));
      }, py::arg("axis") = py::none())
      .def("unsqueeze", &unsqueeze, py::arg("axis"))
      .def("sum", [](const Tensor& self, py::object axis, bool keepdims) {
        return reduce_from_py(self, std::move(axis), keepdims, "sum", [](const Tensor& input) {
          return sum(input);
        }, [](const Tensor& input, int64_t normalized_axis, bool keep) {
          return sum(input, normalized_axis, keep);
        });
      }, py::arg("axis") = py::none(), py::arg("keepdims") = false)
      .def("mean", [](const Tensor& self, py::object axis, bool keepdims) {
        return reduce_from_py(self, std::move(axis), keepdims, "mean", [](const Tensor& input) {
          return mean(input);
        }, [](const Tensor& input, int64_t normalized_axis, bool keep) {
          return mean(input, normalized_axis, keep);
        });
      }, py::arg("axis") = py::none(), py::arg("keepdims") = false)
      .def("max", [](const Tensor& self, py::object axis, bool keepdims) {
        return reduce_from_py(self, std::move(axis), keepdims, "max", [](const Tensor& input) {
          return max(input);
        }, [](const Tensor& input, int64_t normalized_axis, bool keep) {
          return max(input, normalized_axis, keep);
        });
      }, py::arg("axis") = py::none(), py::arg("keepdims") = false)
      .def("min", [](const Tensor& self, py::object axis, bool keepdims) {
        return reduce_from_py(self, std::move(axis), keepdims, "min", [](const Tensor& input) {
          return min(input);
        }, [](const Tensor& input, int64_t normalized_axis, bool keep) {
          return min(input, normalized_axis, keep);
        });
      }, py::arg("axis") = py::none(), py::arg("keepdims") = false)
      .def("argmax", [](const Tensor& self, py::object axis, bool keepdims) {
        return arg_reduce_from_py(self, std::move(axis), keepdims, "argmax", [](const Tensor& input, bool keep) {
          return argmax(input, keep);
        }, [](const Tensor& input, int64_t normalized_axis, bool keep) {
          return argmax(input, normalized_axis, keep);
        });
      }, py::arg("axis") = py::none(), py::arg("keepdims") = false)
      .def("argmin", [](const Tensor& self, py::object axis, bool keepdims) {
        return arg_reduce_from_py(self, std::move(axis), keepdims, "argmin", [](const Tensor& input, bool keep) {
          return argmin(input, keep);
        }, [](const Tensor& input, int64_t normalized_axis, bool keep) {
          return argmin(input, normalized_axis, keep);
        });
      }, py::arg("axis") = py::none(), py::arg("keepdims") = false)
      .def("relu", &relu)
      .def("sigmoid", &sigmoid)
      .def("tanh", &tanh)
      .def("exp", &exp)
      .def("log", &log)
      .def("log1p", &log1p)
      .def("sqrt", &sqrt)
      .def("rsqrt", &rsqrt)
      .def("sin", &sin)
      .def("cos", &cos)
      .def("tan", &tan)
      .def("asin", &asin)
      .def("acos", &acos)
      .def("atan", &atan)
      .def("abs", &abs)
      .def("clamp", &clamp, py::arg("min_value"), py::arg("max_value"))
      .def("clip", &clamp, py::arg("min_value"), py::arg("max_value"))
      .def("astype", [](const Tensor& self, py::object dtype) {
        return astype(self, dtype_from_py(dtype, self.dtype()));
      }, py::arg("dtype"))
      .def("to", [](const Tensor& self, py::object dtype) {
        return astype(self, dtype_from_py(dtype, self.dtype()));
      }, py::arg("dtype"))
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
      .def("equal", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, equal); }, py::arg("other"))
      .def("not_equal", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, not_equal); }, py::arg("other"))
      .def("less", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, less); }, py::arg("other"))
      .def("less_equal", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, less_equal); }, py::arg("other"))
      .def("greater", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, greater); }, py::arg("other"))
      .def("greater_equal", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, greater_equal); }, py::arg("other"))
      .def("maximum", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, maximum); }, py::arg("other"))
      .def("minimum", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, minimum); }, py::arg("other"))
      .def("where", [](const Tensor& self, py::object true_value, py::object false_value) {
        return where(self, ensure_tensor(std::move(true_value)), ensure_tensor(std::move(false_value)));
      }, py::arg("true_value"), py::arg("false_value"))
      .def("__add__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, add); }, py::is_operator())
      .def("__radd__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, add); }, py::is_operator())
      .def("__sub__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, sub); }, py::is_operator())
      .def("__rsub__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, sub); }, py::is_operator())
      .def("__mul__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, mul); }, py::is_operator())
      .def("__rmul__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, mul); }, py::is_operator())
      .def("__truediv__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, div); }, py::is_operator())
      .def("__rtruediv__", [](const Tensor& self, py::object other) { return reverse_binary_tensor_op(other, self, div); }, py::is_operator())
      .def("__eq__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, equal); }, py::is_operator())
      .def("__ne__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, not_equal); }, py::is_operator())
      .def("__lt__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, less); }, py::is_operator())
      .def("__le__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, less_equal); }, py::is_operator())
      .def("__gt__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, greater); }, py::is_operator())
      .def("__ge__", [](const Tensor& self, py::object other) { return binary_tensor_op(self, other, greater_equal); }, py::is_operator())
      .def("__iadd__", [](Tensor&, py::object) -> Tensor {
        throw TensorStudioError("in-place += is not supported in TensorStudio 1.0.0rc1; use x = x + y");
      }, py::is_operator())
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
  module.def("empty", [](py::object shape, py::object dtype) { return empty(shape_from_py(shape), dtype_from_py(dtype)); }, py::arg("shape"), py::arg("dtype") = "float32");
  module.def("full", [](py::object shape, double fill_value, py::object dtype) { return full(shape_from_py(shape), fill_value, dtype_from_py(dtype)); }, py::arg("shape"), py::arg("fill_value"), py::arg("dtype") = "float32");
  module.def(
      "rand",
      [](py::object shape, py::object dtype, py::object seed) {
        std::optional<uint64_t> cpp_seed;
        if (!seed.is_none()) {
          cpp_seed = py::cast<uint64_t>(seed);
        }
        return rand(shape_from_py(shape), dtype_from_py(dtype), cpp_seed);
      },
      py::arg("shape"),
      py::arg("dtype") = "float32",
      py::arg("seed") = py::none());
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
  module.def("eye", [](int64_t n, py::object m, py::object dtype) {
    return eye(n, m.is_none() ? -1 : py::cast<int64_t>(m), dtype_from_py(dtype));
  }, py::arg("n"), py::arg("m") = py::none(), py::arg("dtype") = "float32");
  module.def("linspace", [](double start, double stop, int64_t steps, py::object dtype) {
    return linspace(start, stop, steps, dtype_from_py(dtype));
  }, py::arg("start"), py::arg("stop"), py::arg("steps"), py::arg("dtype") = "float32");
  module.def("manual_seed", &manual_seed, py::arg("seed"));
  module.def(
      "promote_types",
      [](py::object left, py::object right) {
        return dtype_name(promote_types(dtype_from_py(left), dtype_from_py(right)));
      },
      py::arg("left"),
      py::arg("right"));
}

}  // namespace tensorstudio::bindings
