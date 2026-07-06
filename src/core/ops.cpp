#include "tensorstudio/ops.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>

#include "tensorstudio/autograd.hpp"
#include "tensorstudio/errors.hpp"

namespace tensorstudio {
namespace {

using BinaryOp = std::function<double(double, double)>;
using UnaryOp = std::function<double(double)>;

Tensor full_like(const Shape& shape, double value, DType dtype) {
  return full(shape, value, dtype);
}

Tensor elementwise_binary_impl(
    const Tensor& left,
    const Tensor& right,
    DType dtype,
    const BinaryOp& op,
    bool track_grad) {
  const Shape out_shape = broadcast_shapes(left.shape(), right.shape());
  Tensor out(out_shape, dtype, false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    const auto left_storage =
        logical_to_storage_offset(i, out_shape, left.shape(), left.strides(), left.offset());
    const auto right_storage =
        logical_to_storage_offset(i, out_shape, right.shape(), right.strides(), right.offset());
    out.set_value_at_logical(i, op(left.value_at_storage(left_storage), right.value_at_storage(right_storage)));
  }
  (void)track_grad;
  return out;
}

Tensor elementwise_unary_impl(const Tensor& input, DType dtype, const UnaryOp& op) {
  Tensor out(input.shape(), dtype, false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    out.set_value_at_logical(i, op(input.value_at_logical(i)));
  }
  return out;
}

Tensor add_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor sub_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor mul_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor div_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor neg_impl(const Tensor& input, bool track_grad);
Tensor pow_impl(const Tensor& input, double exponent, bool track_grad);
Tensor matmul_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor sum_impl(const Tensor& input, bool track_grad);
Tensor mean_impl(const Tensor& input, bool track_grad);
Tensor relu_impl(const Tensor& input, bool track_grad);
Tensor sigmoid_impl(const Tensor& input, bool track_grad);
Tensor tanh_impl(const Tensor& input, bool track_grad);
Tensor exp_impl(const Tensor& input, bool track_grad);
Tensor log_impl(const Tensor& input, bool track_grad);
Tensor reshape_impl(const Tensor& input, const Shape& shape, bool track_grad);
Tensor transpose_impl(const Tensor& input, bool track_grad);

Tensor add_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left, right, promote_types(left.dtype(), right.dtype()), [](double a, double b) { return a + b; }, track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left_shape = left.shape(), right_shape = right.shape()](const Tensor& grad) {
      return std::vector<Tensor>{unbroadcast(grad, left_shape), unbroadcast(grad, right_shape)};
    });
  }
  return out;
}

Tensor sub_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left, right, promote_types(left.dtype(), right.dtype()), [](double a, double b) { return a - b; }, track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left_shape = left.shape(), right_shape = right.shape()](const Tensor& grad) {
      return std::vector<Tensor>{unbroadcast(grad, left_shape), unbroadcast(neg_impl(grad, false), right_shape)};
    });
  }
  return out;
}

Tensor mul_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left, right, promote_types(left.dtype(), right.dtype()), [](double a, double b) { return a * b; }, track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left, right](const Tensor& grad) {
      Tensor left_grad = unbroadcast(mul_impl(grad, right, false), left.shape());
      Tensor right_grad = unbroadcast(mul_impl(grad, left, false), right.shape());
      return std::vector<Tensor>{left_grad, right_grad};
    });
  }
  return out;
}

Tensor div_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  const DType dtype =
      left.dtype() == DType::Float64 || right.dtype() == DType::Float64 ? DType::Float64 : DType::Float32;
  Tensor out =
      elementwise_binary_impl(left, right, dtype, [](double a, double b) { return a / b; }, track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left, right](const Tensor& grad) {
      Tensor left_grad = unbroadcast(div_impl(grad, right, false), left.shape());
      Tensor denom = mul_impl(right, right, false);
      Tensor right_grad = unbroadcast(neg_impl(div_impl(mul_impl(grad, left, false), denom, false), false), right.shape());
      return std::vector<Tensor>{left_grad, right_grad};
    });
  }
  return out;
}

Tensor neg_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, input.dtype(), [](double value) { return -value; });
  if (track_grad) {
    set_history(out, {input}, [](const Tensor& grad) { return std::vector<Tensor>{neg_impl(grad, false)}; });
  }
  return out;
}

Tensor pow_impl(const Tensor& input, double exponent, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [exponent](double value) {
    return std::pow(value, exponent);
  });
  if (track_grad) {
    set_history(out, {input}, [input, exponent](const Tensor& grad) {
      Tensor coeff = scalar_tensor(exponent, grad.dtype());
      Tensor base = pow_impl(input, exponent - 1.0, false);
      return std::vector<Tensor>{mul_impl(mul_impl(grad, coeff, false), base, false)};
    });
  }
  return out;
}

Tensor matmul_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  if (left.ndim() != 2 || right.ndim() != 2) {
    throw ShapeError(
        "matmul expects two 2D tensors, got " + shape_to_string(left.shape()) + " and " +
        shape_to_string(right.shape()));
  }
  if (left.shape()[1] != right.shape()[0]) {
    throw ShapeError(
        "matmul shape mismatch: " + shape_to_string(left.shape()) + " @ " + shape_to_string(right.shape()));
  }
  const int64_t rows = left.shape()[0];
  const int64_t inner = left.shape()[1];
  const int64_t cols = right.shape()[1];
  Tensor out({rows, cols}, promote_types(left.dtype(), right.dtype()), false);
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      double acc = 0.0;
      for (int64_t k = 0; k < inner; ++k) {
        const int64_t left_storage = left.offset() + r * left.strides()[0] + k * left.strides()[1];
        const int64_t right_storage = right.offset() + k * right.strides()[0] + c * right.strides()[1];
        acc += left.value_at_storage(left_storage) * right.value_at_storage(right_storage);
      }
      out.set_value_at_logical(r * cols + c, acc);
    }
  }
  if (track_grad) {
    set_history(out, {left, right}, [left, right](const Tensor& grad) {
      Tensor left_grad = matmul_impl(grad, transpose_impl(right, false), false);
      Tensor right_grad = matmul_impl(transpose_impl(left, false), grad, false);
      return std::vector<Tensor>{left_grad, right_grad};
    });
  }
  return out;
}

Tensor sum_impl(const Tensor& input, bool track_grad) {
  Tensor out({}, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float64, false);
  double acc = 0.0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    acc += input.value_at_logical(i);
  }
  out.set_value_at_logical(0, acc);
  if (track_grad) {
    set_history(out, {input}, [shape = input.shape(), dtype = out.dtype()](const Tensor& grad) {
      return std::vector<Tensor>{full_like(shape, grad.value_at_logical(0), dtype)};
    });
  }
  return out;
}

Tensor mean_impl(const Tensor& input, bool track_grad) {
  if (input.numel() == 0) {
    throw ShapeError("mean is undefined for empty tensors");
  }
  Tensor out({}, DType::Float64, false);
  double acc = 0.0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    acc += input.value_at_logical(i);
  }
  out.set_value_at_logical(0, acc / static_cast<double>(input.numel()));
  if (track_grad) {
    set_history(out, {input}, [shape = input.shape(), n = input.numel()](const Tensor& grad) {
      return std::vector<Tensor>{full_like(shape, grad.value_at_logical(0) / static_cast<double>(n), grad.dtype())};
    });
  }
  return out;
}

Tensor relu_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, input.dtype(), [](double value) { return std::max(0.0, value); });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor mask(input.shape(), grad.dtype(), false);
      for (int64_t i = 0; i < input.numel(); ++i) {
        mask.set_value_at_logical(i, input.value_at_logical(i) > 0.0 ? 1.0 : 0.0);
      }
      return std::vector<Tensor>{mul_impl(grad, mask, false)};
    });
  }
  return out;
}

Tensor sigmoid_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
    return 1.0 / (1.0 + std::exp(-value));
  });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor y = sigmoid_impl(input, false);
      Tensor one_minus_y = sub_impl(full_like(y.shape(), 1.0, y.dtype()), y, false);
      return std::vector<Tensor>{mul_impl(mul_impl(grad, y, false), one_minus_y, false)};
    });
  }
  return out;
}

Tensor tanh_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
    return std::tanh(value);
  });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor y = tanh_impl(input, false);
      Tensor one_minus_yy = sub_impl(full_like(y.shape(), 1.0, y.dtype()), mul_impl(y, y, false), false);
      return std::vector<Tensor>{mul_impl(grad, one_minus_yy, false)};
    });
  }
  return out;
}

Tensor exp_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
    return std::exp(value);
  });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      return std::vector<Tensor>{mul_impl(grad, exp_impl(input, false), false)};
    });
  }
  return out;
}

Tensor log_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
    return std::log(value);
  });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      return std::vector<Tensor>{div_impl(grad, input, false)};
    });
  }
  return out;
}

Tensor reshape_impl(const Tensor& input, const Shape& shape, bool track_grad) {
  if (!input.is_contiguous()) {
    throw ShapeError("reshape currently requires a contiguous tensor");
  }
  Shape normalized = normalize_shape(shape, input.numel());
  Tensor out(input.impl()->storage, input.dtype(), normalized, contiguous_strides(normalized), input.offset(), false);
  if (track_grad) {
    set_history(out, {input}, [original_shape = input.shape()](const Tensor& grad) {
      return std::vector<Tensor>{reshape_impl(grad, original_shape, false)};
    });
  }
  return out;
}

Tensor transpose_impl(const Tensor& input, bool track_grad) {
  if (input.ndim() != 2) {
    throw ShapeError("transpose expects a 2D tensor, got " + shape_to_string(input.shape()));
  }
  Tensor out(
      input.impl()->storage,
      input.dtype(),
      Shape{input.shape()[1], input.shape()[0]},
      Shape{input.strides()[1], input.strides()[0]},
      input.offset(),
      false);
  if (track_grad) {
    set_history(out, {input}, [](const Tensor& grad) { return std::vector<Tensor>{transpose_impl(grad, false)}; });
  }
  return out;
}

}  // namespace

Tensor zeros(const Shape& shape, DType dtype) {
  return full(shape, 0.0, dtype);
}

Tensor ones(const Shape& shape, DType dtype) {
  return full(shape, 1.0, dtype);
}

Tensor full(const Shape& shape, double fill_value, DType dtype) {
  Tensor out(shape, dtype, false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    out.set_value_at_logical(i, fill_value);
  }
  return out;
}

Tensor arange(double start, double stop, double step, DType dtype) {
  if (step == 0.0) {
    throw ShapeError("arange step must not be zero");
  }
  const double span = (stop - start) / step;
  const auto count = static_cast<int64_t>(std::max(0.0, std::ceil(span)));
  Tensor out({count}, dtype, false);
  double value = start;
  for (int64_t i = 0; i < count; ++i) {
    out.set_value_at_logical(i, value);
    value += step;
  }
  return out;
}

Tensor add(const Tensor& left, const Tensor& right) {
  return add_impl(left, right, true);
}

Tensor sub(const Tensor& left, const Tensor& right) {
  return sub_impl(left, right, true);
}

Tensor mul(const Tensor& left, const Tensor& right) {
  return mul_impl(left, right, true);
}

Tensor div(const Tensor& left, const Tensor& right) {
  return div_impl(left, right, true);
}

Tensor neg(const Tensor& input) {
  return neg_impl(input, true);
}

Tensor pow(const Tensor& input, double exponent) {
  return pow_impl(input, exponent, true);
}

Tensor matmul(const Tensor& left, const Tensor& right) {
  return matmul_impl(left, right, true);
}

Tensor sum(const Tensor& input) {
  return sum_impl(input, true);
}

Tensor mean(const Tensor& input) {
  return mean_impl(input, true);
}

Tensor relu(const Tensor& input) {
  return relu_impl(input, true);
}

Tensor sigmoid(const Tensor& input) {
  return sigmoid_impl(input, true);
}

Tensor tanh(const Tensor& input) {
  return tanh_impl(input, true);
}

Tensor exp(const Tensor& input) {
  return exp_impl(input, true);
}

Tensor log(const Tensor& input) {
  return log_impl(input, true);
}

Tensor reshape(const Tensor& input, const Shape& shape) {
  return reshape_impl(input, shape, true);
}

Tensor flatten(const Tensor& input) {
  return reshape(input, Shape{input.numel()});
}

Tensor transpose(const Tensor& input) {
  return transpose_impl(input, true);
}

Tensor scalar_tensor(double value, DType dtype) {
  Tensor out({}, dtype, false);
  out.set_value_at_logical(0, value);
  return out;
}

Tensor unbroadcast(const Tensor& grad, const Shape& target_shape) {
  Tensor out(target_shape, grad.dtype(), false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    out.set_value_at_logical(i, 0.0);
  }

  const Shape& grad_shape = grad.shape();
  const auto grad_rank = grad_shape.size();
  const auto target_rank = target_shape.size();

  for (int64_t linear = 0; linear < grad.numel(); ++linear) {
    std::vector<int64_t> grad_coords(grad_rank, 0);
    int64_t remaining = linear;
    for (std::size_t dim = 0; dim < grad_rank; ++dim) {
      int64_t suffix = 1;
      for (std::size_t j = dim + 1; j < grad_rank; ++j) {
        suffix *= grad_shape[j];
      }
      grad_coords[dim] = suffix == 0 ? 0 : remaining / suffix;
      remaining = suffix == 0 ? 0 : remaining % suffix;
    }

    int64_t target_linear = 0;
    int64_t target_stride = 1;
    for (int64_t dim = static_cast<int64_t>(target_rank) - 1; dim >= 0; --dim) {
      const auto grad_dim = static_cast<int64_t>(grad_rank) - static_cast<int64_t>(target_rank) + dim;
      int64_t coord = 0;
      if (grad_dim >= 0 && target_shape[static_cast<std::size_t>(dim)] != 1) {
        coord = grad_coords[static_cast<std::size_t>(grad_dim)];
      }
      target_linear += coord * target_stride;
      target_stride *= target_shape[static_cast<std::size_t>(dim)];
    }
    out.set_value_at_logical(target_linear, out.value_at_logical(target_linear) + grad.value_at_logical(linear));
  }
  return out;
}

}  // namespace tensorstudio
