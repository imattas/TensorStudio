#include "bindings.hpp"

#include <utility>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "tensorstudio/ops.hpp"

namespace tensorstudio::bindings {

void bind_ops(py::module_& module) {
  module.def("add", [](const Tensor& left, py::object right) { return add(left, ensure_tensor(right)); });
  module.def("sub", [](const Tensor& left, py::object right) { return sub(left, ensure_tensor(right)); });
  module.def("mul", [](const Tensor& left, py::object right) { return mul(left, ensure_tensor(right)); });
  module.def("div", [](const Tensor& left, py::object right) { return div(left, ensure_tensor(right)); });
  module.def("equal", [](const Tensor& left, py::object right) { return equal(left, ensure_tensor(right)); });
  module.def("not_equal", [](const Tensor& left, py::object right) { return not_equal(left, ensure_tensor(right)); });
  module.def("less", [](const Tensor& left, py::object right) { return less(left, ensure_tensor(right)); });
  module.def("less_equal", [](const Tensor& left, py::object right) { return less_equal(left, ensure_tensor(right)); });
  module.def("greater", [](const Tensor& left, py::object right) { return greater(left, ensure_tensor(right)); });
  module.def("greater_equal", [](const Tensor& left, py::object right) { return greater_equal(left, ensure_tensor(right)); });
  module.def("neg", &neg);
  module.def("pow", &pow, py::arg("input"), py::arg("exponent"));
  module.def("matmul", [](const Tensor& left, py::object right) { return matmul(left, ensure_tensor(right)); });
  module.def(
      "conv2d",
      [](const Tensor& input,
         const Tensor& weight,
         py::object bias,
         int64_t stride_h,
         int64_t stride_w,
         int64_t padding_h,
         int64_t padding_w,
         int64_t dilation_h,
         int64_t dilation_w) {
        std::optional<Tensor> maybe_bias;
        if (!bias.is_none()) {
          maybe_bias = ensure_tensor(bias);
        }
        return conv2d(
            input, weight, maybe_bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
      },
      py::arg("input"),
      py::arg("weight"),
      py::arg("bias") = py::none(),
      py::arg("stride_h") = 1,
      py::arg("stride_w") = 1,
      py::arg("padding_h") = 0,
      py::arg("padding_w") = 0,
      py::arg("dilation_h") = 1,
      py::arg("dilation_w") = 1);
  module.def(
      "max_pool2d",
      &max_pool2d,
      py::arg("input"),
      py::arg("kernel_h"),
      py::arg("kernel_w"),
      py::arg("stride_h") = 1,
      py::arg("stride_w") = 1,
      py::arg("padding_h") = 0,
      py::arg("padding_w") = 0,
      py::arg("dilation_h") = 1,
      py::arg("dilation_w") = 1);
  module.def(
      "avg_pool2d",
      &avg_pool2d,
      py::arg("input"),
      py::arg("kernel_h"),
      py::arg("kernel_w"),
      py::arg("stride_h") = 1,
      py::arg("stride_w") = 1,
      py::arg("padding_h") = 0,
      py::arg("padding_w") = 0,
      py::arg("count_include_pad") = false);
  module.def("sum", [](const Tensor& input, py::object axis, bool keepdims) {
    return reduce_from_py(input, std::move(axis), keepdims, "sum", [](const Tensor& tensor) {
      return sum(tensor);
    }, [](const Tensor& tensor, int64_t normalized_axis, bool keep) {
      return sum(tensor, normalized_axis, keep);
    });
  }, py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
  module.def("mean", [](const Tensor& input, py::object axis, bool keepdims) {
    return reduce_from_py(input, std::move(axis), keepdims, "mean", [](const Tensor& tensor) {
      return mean(tensor);
    }, [](const Tensor& tensor, int64_t normalized_axis, bool keep) {
      return mean(tensor, normalized_axis, keep);
    });
  }, py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
  module.def("max", [](const Tensor& input, py::object axis, bool keepdims) {
    return reduce_from_py(input, std::move(axis), keepdims, "max", [](const Tensor& tensor) {
      return max(tensor);
    }, [](const Tensor& tensor, int64_t normalized_axis, bool keep) {
      return max(tensor, normalized_axis, keep);
    });
  }, py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
  module.def("min", [](const Tensor& input, py::object axis, bool keepdims) {
    return reduce_from_py(input, std::move(axis), keepdims, "min", [](const Tensor& tensor) {
      return min(tensor);
    }, [](const Tensor& tensor, int64_t normalized_axis, bool keep) {
      return min(tensor, normalized_axis, keep);
    });
  }, py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
  module.def("argmax", [](const Tensor& input, py::object axis, bool keepdims) {
    return arg_reduce_from_py(input, std::move(axis), keepdims, "argmax", [](const Tensor& tensor, bool keep) {
      return argmax(tensor, keep);
    }, [](const Tensor& tensor, int64_t normalized_axis, bool keep) {
      return argmax(tensor, normalized_axis, keep);
    });
  }, py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
  module.def("argmin", [](const Tensor& input, py::object axis, bool keepdims) {
    return arg_reduce_from_py(input, std::move(axis), keepdims, "argmin", [](const Tensor& tensor, bool keep) {
      return argmin(tensor, keep);
    }, [](const Tensor& tensor, int64_t normalized_axis, bool keep) {
      return argmin(tensor, normalized_axis, keep);
    });
  }, py::arg("input"), py::arg("axis") = py::none(), py::arg("keepdims") = false);
  module.def("relu", &relu);
  module.def("sigmoid", &sigmoid);
  module.def("tanh", &tanh);
  module.def("exp", &exp);
  module.def("log", &log);
  module.def("log1p", &log1p);
  module.def("sqrt", &sqrt);
  module.def("rsqrt", &rsqrt);
  module.def("sin", &sin);
  module.def("cos", &cos);
  module.def("tan", &tan);
  module.def("asin", &asin);
  module.def("acos", &acos);
  module.def("atan", &atan);
  module.def("abs", &abs);
  module.def("clamp", &clamp, py::arg("input"), py::arg("min_value"), py::arg("max_value"));
  module.def(
      "astype",
      [](const Tensor& input, py::object dtype) { return astype(input, dtype_from_py(dtype, input.dtype())); },
      py::arg("input"),
      py::arg("dtype"));
  module.def("concat", &concat, py::arg("tensors"), py::arg("axis") = 0);
  module.def("stack", &stack, py::arg("tensors"), py::arg("axis") = 0);
  module.def("reshape", [](const Tensor& input, py::object shape) { return reshape(input, shape_from_py(shape)); });
  module.def("flatten", &flatten);
  module.def("transpose", &transpose);
}

}  // namespace tensorstudio::bindings
