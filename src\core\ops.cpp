#include "tensorstudio/ops.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
#include <vector>

#include "tensorstudio/autograd.hpp"
#include "tensorstudio/errors.hpp"

namespace tensorstudio {
namespace {

Tensor full_like(const Shape& shape, double value, DType dtype) {
  return full(shape, value, dtype);
}

template <typename T>
T* raw_data(Tensor& tensor) {
  return reinterpret_cast<T*>(tensor.impl()->storage->data());
}

template <typename T>
const T* raw_data(const Tensor& tensor) {
  return reinterpret_cast<const T*>(tensor.impl()->storage->data());
}

template <typename T>
double to_double(T value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value ? 1.0 : 0.0;
  } else {
    return static_cast<double>(value);
  }
}

template <typename T>
T from_double(double value) {
  if constexpr (std::is_same_v<T, bool>) {
    return value != 0.0;
  } else {
    return static_cast<T>(value);
  }
}

double value_at_storage_unchecked(const Tensor& tensor, int64_t storage_index) {
  const auto index = static_cast<std::size_t>(storage_index);
  switch (tensor.dtype()) {
    case DType::Float32:
      return to_double(raw_data<float>(tensor)[index]);
    case DType::Float64:
      return to_double(raw_data<double>(tensor)[index]);
    case DType::Int32:
      return to_double(raw_data<int32_t>(tensor)[index]);
    case DType::Int64:
      return to_double(raw_data<int64_t>(tensor)[index]);
    case DType::Bool:
      return to_double(raw_data<bool>(tensor)[index]);
  }
  throw DTypeError("unknown dtype");
}

void set_value_at_storage_unchecked(Tensor& tensor, int64_t storage_index, double value) {
  const auto index = static_cast<std::size_t>(storage_index);
  switch (tensor.dtype()) {
    case DType::Float32:
      raw_data<float>(tensor)[index] = from_double<float>(value);
      return;
    case DType::Float64:
      raw_data<double>(tensor)[index] = from_double<double>(value);
      return;
    case DType::Int32:
      raw_data<int32_t>(tensor)[index] = from_double<int32_t>(value);
      return;
    case DType::Int64:
      raw_data<int64_t>(tensor)[index] = from_double<int64_t>(value);
      return;
    case DType::Bool:
      raw_data<bool>(tensor)[index] = from_double<bool>(value);
      return;
  }
  throw DTypeError("unknown dtype");
}

void fill_storage_unchecked(Tensor& tensor, double value) {
  const auto count = static_cast<std::size_t>(tensor.numel());
  const auto offset = static_cast<std::size_t>(tensor.offset());
  switch (tensor.dtype()) {
    case DType::Float32:
      std::fill(raw_data<float>(tensor) + offset, raw_data<float>(tensor) + offset + count, from_double<float>(value));
      return;
    case DType::Float64:
      std::fill(
          raw_data<double>(tensor) + offset, raw_data<double>(tensor) + offset + count, from_double<double>(value));
      return;
    case DType::Int32:
      std::fill(
          raw_data<int32_t>(tensor) + offset, raw_data<int32_t>(tensor) + offset + count, from_double<int32_t>(value));
      return;
    case DType::Int64:
      std::fill(
          raw_data<int64_t>(tensor) + offset, raw_data<int64_t>(tensor) + offset + count, from_double<int64_t>(value));
      return;
    case DType::Bool:
      std::fill(raw_data<bool>(tensor) + offset, raw_data<bool>(tensor) + offset + count, from_double<bool>(value));
      return;
  }
  throw DTypeError("unknown dtype");
}

void write_contiguous_from_accumulator(Tensor& tensor, const std::vector<double>& values) {
  const auto offset = static_cast<std::size_t>(tensor.offset());
  switch (tensor.dtype()) {
    case DType::Float32: {
      auto* out = raw_data<float>(tensor) + offset;
      for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = from_double<float>(values[i]);
      }
      return;
    }
    case DType::Float64: {
      auto* out = raw_data<double>(tensor) + offset;
      for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = from_double<double>(values[i]);
      }
      return;
    }
    case DType::Int32: {
      auto* out = raw_data<int32_t>(tensor) + offset;
      for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = from_double<int32_t>(values[i]);
      }
      return;
    }
    case DType::Int64: {
      auto* out = raw_data<int64_t>(tensor) + offset;
      for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = from_double<int64_t>(values[i]);
      }
      return;
    }
    case DType::Bool: {
      auto* out = raw_data<bool>(tensor) + offset;
      for (std::size_t i = 0; i < values.size(); ++i) {
        out[i] = from_double<bool>(values[i]);
      }
      return;
    }
  }
  throw DTypeError("unknown dtype");
}

enum class BinaryFastPath {
  Direct,
  LeftScalar,
  RightScalar,
};

template <typename OutT, typename LeftT, typename RightT, typename Op>
void binary_direct_kernel(Tensor& out, const Tensor& left, const Tensor& right, Op op) {
  const auto count = out.numel();
  auto* out_data = raw_data<OutT>(out) + out.offset();
  const auto* left_data = raw_data<LeftT>(left) + left.offset();
  const auto* right_data = raw_data<RightT>(right) + right.offset();
  for (int64_t i = 0; i < count; ++i) {
    out_data[i] = from_double<OutT>(op(to_double(left_data[i]), to_double(right_data[i])));
  }
}

template <typename OutT, typename LeftT, typename RightT, typename Op>
void binary_left_scalar_kernel(Tensor& out, const Tensor& left, const Tensor& right, Op op) {
  const auto count = out.numel();
  auto* out_data = raw_data<OutT>(out) + out.offset();
  const double left_value = to_double(raw_data<LeftT>(left)[static_cast<std::size_t>(left.offset())]);
  const auto* right_data = raw_data<RightT>(right) + right.offset();
  for (int64_t i = 0; i < count; ++i) {
    out_data[i] = from_double<OutT>(op(left_value, to_double(right_data[i])));
  }
}

template <typename OutT, typename LeftT, typename RightT, typename Op>
void binary_right_scalar_kernel(Tensor& out, const Tensor& left, const Tensor& right, Op op) {
  const auto count = out.numel();
  auto* out_data = raw_data<OutT>(out) + out.offset();
  const auto* left_data = raw_data<LeftT>(left) + left.offset();
  const double right_value = to_double(raw_data<RightT>(right)[static_cast<std::size_t>(right.offset())]);
  for (int64_t i = 0; i < count; ++i) {
    out_data[i] = from_double<OutT>(op(to_double(left_data[i]), right_value));
  }
}

template <typename OutT, typename LeftT, typename RightT, typename Op>
void binary_kernel_by_mode(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    Op op,
    BinaryFastPath mode) {
  switch (mode) {
    case BinaryFastPath::Direct:
      binary_direct_kernel<OutT, LeftT, RightT>(out, left, right, op);
      return;
    case BinaryFastPath::LeftScalar:
      binary_left_scalar_kernel<OutT, LeftT, RightT>(out, left, right, op);
      return;
    case BinaryFastPath::RightScalar:
      binary_right_scalar_kernel<OutT, LeftT, RightT>(out, left, right, op);
      return;
  }
  throw TensorStudioError("unknown binary fast path");
}

template <typename OutT, typename LeftT, typename Op>
void dispatch_binary_right(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    Op op,
    BinaryFastPath mode) {
  switch (right.dtype()) {
    case DType::Float32:
      binary_kernel_by_mode<OutT, LeftT, float>(out, left, right, op, mode);
      return;
    case DType::Float64:
      binary_kernel_by_mode<OutT, LeftT, double>(out, left, right, op, mode);
      return;
    case DType::Int32:
      binary_kernel_by_mode<OutT, LeftT, int32_t>(out, left, right, op, mode);
      return;
    case DType::Int64:
      binary_kernel_by_mode<OutT, LeftT, int64_t>(out, left, right, op, mode);
      return;
    case DType::Bool:
      binary_kernel_by_mode<OutT, LeftT, bool>(out, left, right, op, mode);
      return;
  }
  throw DTypeError("unknown dtype");
}

template <typename OutT, typename Op>
void dispatch_binary_left(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    Op op,
    BinaryFastPath mode) {
  switch (left.dtype()) {
    case DType::Float32:
      dispatch_binary_right<OutT, float>(out, left, right, op, mode);
      return;
    case DType::Float64:
      dispatch_binary_right<OutT, double>(out, left, right, op, mode);
      return;
    case DType::Int32:
      dispatch_binary_right<OutT, int32_t>(out, left, right, op, mode);
      return;
    case DType::Int64:
      dispatch_binary_right<OutT, int64_t>(out, left, right, op, mode);
      return;
    case DType::Bool:
      dispatch_binary_right<OutT, bool>(out, left, right, op, mode);
      return;
  }
  throw DTypeError("unknown dtype");
}

template <typename Op>
void elementwise_binary_fast_path(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    Op op,
    BinaryFastPath mode) {
  switch (out.dtype()) {
    case DType::Float32:
      dispatch_binary_left<float>(out, left, right, op, mode);
      return;
    case DType::Float64:
      dispatch_binary_left<double>(out, left, right, op, mode);
      return;
    case DType::Int32:
      dispatch_binary_left<int32_t>(out, left, right, op, mode);
      return;
    case DType::Int64:
      dispatch_binary_left<int64_t>(out, left, right, op, mode);
      return;
    case DType::Bool:
      dispatch_binary_left<bool>(out, left, right, op, mode);
      return;
  }
  throw DTypeError("unknown dtype");
}

template <typename OutT, typename InT, typename Op>
void unary_contiguous_kernel(Tensor& out, const Tensor& input, Op op) {
  const auto count = out.numel();
  auto* out_data = raw_data<OutT>(out) + out.offset();
  const auto* input_data = raw_data<InT>(input) + input.offset();
  for (int64_t i = 0; i < count; ++i) {
    out_data[i] = from_double<OutT>(op(to_double(input_data[i])));
  }
}

template <typename OutT, typename Op>
void dispatch_unary_input(Tensor& out, const Tensor& input, Op op) {
  switch (input.dtype()) {
    case DType::Float32:
      unary_contiguous_kernel<OutT, float>(out, input, op);
      return;
    case DType::Float64:
      unary_contiguous_kernel<OutT, double>(out, input, op);
      return;
    case DType::Int32:
      unary_contiguous_kernel<OutT, int32_t>(out, input, op);
      return;
    case DType::Int64:
      unary_contiguous_kernel<OutT, int64_t>(out, input, op);
      return;
    case DType::Bool:
      unary_contiguous_kernel<OutT, bool>(out, input, op);
      return;
  }
  throw DTypeError("unknown dtype");
}

template <typename Op>
void elementwise_unary_fast_path(Tensor& out, const Tensor& input, Op op) {
  switch (out.dtype()) {
    case DType::Float32:
      dispatch_unary_input<float>(out, input, op);
      return;
    case DType::Float64:
      dispatch_unary_input<double>(out, input, op);
      return;
    case DType::Int32:
      dispatch_unary_input<int32_t>(out, input, op);
      return;
    case DType::Int64:
      dispatch_unary_input<int64_t>(out, input, op);
      return;
    case DType::Bool:
      dispatch_unary_input<bool>(out, input, op);
      return;
  }
  throw DTypeError("unknown dtype");
}

template <typename T>
double sum_contiguous_kernel(const Tensor& input) {
  const auto* data = raw_data<T>(input) + input.offset();
  double acc = 0.0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    acc += to_double(data[i]);
  }
  return acc;
}

double sum_contiguous_fast_path(const Tensor& input) {
  switch (input.dtype()) {
    case DType::Float32:
      return sum_contiguous_kernel<float>(input);
    case DType::Float64:
      return sum_contiguous_kernel<double>(input);
    case DType::Int32:
      return sum_contiguous_kernel<int32_t>(input);
    case DType::Int64:
      return sum_contiguous_kernel<int64_t>(input);
    case DType::Bool:
      return sum_contiguous_kernel<bool>(input);
  }
  throw DTypeError("unknown dtype");
}

template <typename Op>
Tensor elementwise_binary_impl(
    const Tensor& left,
    const Tensor& right,
    DType dtype,
    Op op,
    bool track_grad) {
  const Shape out_shape = broadcast_shapes(left.shape(), right.shape());
  Tensor out(out_shape, dtype, false);
  const bool left_direct = left.shape() == out_shape && left.is_contiguous();
  const bool right_direct = right.shape() == out_shape && right.is_contiguous();
  const bool left_scalar = left.numel() == 1;
  const bool right_scalar = right.numel() == 1;

  if (left_direct && right_direct) {
    elementwise_binary_fast_path(out, left, right, op, BinaryFastPath::Direct);
    (void)track_grad;
    return out;
  }

  if (left_scalar && right_direct) {
    elementwise_binary_fast_path(out, left, right, op, BinaryFastPath::LeftScalar);
    (void)track_grad;
    return out;
  }

  if (right_scalar && left_direct) {
    elementwise_binary_fast_path(out, left, right, op, BinaryFastPath::RightScalar);
    (void)track_grad;
    return out;
  }

  for (int64_t i = 0; i < out.numel(); ++i) {
    const auto left_storage =
        logical_to_storage_offset(i, out_shape, left.shape(), left.strides(), left.offset());
    const auto right_storage =
        logical_to_storage_offset(i, out_shape, right.shape(), right.strides(), right.offset());
    set_value_at_storage_unchecked(
        out,
        i,
        op(value_at_storage_unchecked(left, left_storage), value_at_storage_unchecked(right, right_storage)));
  }
  (void)track_grad;
  return out;
}

template <typename Op>
Tensor elementwise_unary_impl(const Tensor& input, DType dtype, Op op) {
  Tensor out(input.shape(), dtype, false);
  if (input.is_contiguous()) {
    elementwise_unary_fast_path(out, input, op);
    return out;
  }
  for (int64_t i = 0; i < out.numel(); ++i) {
    const auto storage = logical_to_storage_offset(i, input.shape(), input.shape(), input.strides(), input.offset());
    set_value_at_storage_unchecked(out, i, op(value_at_storage_unchecked(input, storage)));
  }
  return out;
}

template <typename LeftT, typename RightT>
void matmul_contiguous_kernel(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    int64_t rows,
    int64_t inner,
    int64_t cols) {
  const auto* left_data = raw_data<LeftT>(left) + left.offset();
  const auto* right_data = raw_data<RightT>(right) + right.offset();

  if constexpr (std::is_same_v<LeftT, float> && std::is_same_v<RightT, float>) {
    if (out.dtype() == DType::Float32) {
      std::vector<float> accumulator(static_cast<std::size_t>(rows * cols), 0.0F);
      for (int64_t r = 0; r < rows; ++r) {
        auto* out_row = accumulator.data() + static_cast<std::size_t>(r * cols);
        const auto* left_row = left_data + r * inner;
        for (int64_t k = 0; k < inner; ++k) {
          const float left_value = left_row[k];
          const auto* right_row = right_data + k * cols;
          for (int64_t c = 0; c < cols; ++c) {
            out_row[c] += left_value * right_row[c];
          }
        }
      }
      auto* out_data = raw_data<float>(out) + out.offset();
      std::copy(accumulator.begin(), accumulator.end(), out_data);
      return;
    }
  }

  std::vector<double> accumulator(static_cast<std::size_t>(rows * cols), 0.0);

  for (int64_t r = 0; r < rows; ++r) {
    auto* out_row = accumulator.data() + static_cast<std::size_t>(r * cols);
    const auto* left_row = left_data + r * inner;
    for (int64_t k = 0; k < inner; ++k) {
      const double left_value = to_double(left_row[k]);
      const auto* right_row = right_data + k * cols;
      for (int64_t c = 0; c < cols; ++c) {
        out_row[c] += left_value * to_double(right_row[c]);
      }
    }
  }

  write_contiguous_from_accumulator(out, accumulator);
}

template <typename LeftT>
void dispatch_matmul_right(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    int64_t rows,
    int64_t inner,
    int64_t cols) {
  switch (right.dtype()) {
    case DType::Float32:
      matmul_contiguous_kernel<LeftT, float>(out, left, right, rows, inner, cols);
      return;
    case DType::Float64:
      matmul_contiguous_kernel<LeftT, double>(out, left, right, rows, inner, cols);
      return;
    case DType::Int32:
      matmul_contiguous_kernel<LeftT, int32_t>(out, left, right, rows, inner, cols);
      return;
    case DType::Int64:
      matmul_contiguous_kernel<LeftT, int64_t>(out, left, right, rows, inner, cols);
      return;
    case DType::Bool:
      matmul_contiguous_kernel<LeftT, bool>(out, left, right, rows, inner, cols);
      return;
  }
  throw DTypeError("unknown dtype");
}

void matmul_contiguous_fast_path(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    int64_t rows,
    int64_t inner,
    int64_t cols) {
  switch (left.dtype()) {
    case DType::Float32:
      dispatch_matmul_right<float>(out, left, right, rows, inner, cols);
      return;
    case DType::Float64:
      dispatch_matmul_right<double>(out, left, right, rows, inner, cols);
      return;
    case DType::Int32:
      dispatch_matmul_right<int32_t>(out, left, right, rows, inner, cols);
      return;
    case DType::Int64:
      dispatch_matmul_right<int64_t>(out, left, right, rows, inner, cols);
      return;
    case DType::Bool:
      dispatch_matmul_right<bool>(out, left, right, rows, inner, cols);
      return;
  }
  throw DTypeError("unknown dtype");
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
Tensor max_impl(const Tensor& input, bool track_grad);
Tensor min_impl(const Tensor& input, bool track_grad);
Tensor relu_impl(const Tensor& input, bool track_grad);
Tensor sigmoid_impl(const Tensor& input, bool track_grad);
Tensor tanh_impl(const Tensor& input, bool track_grad);
Tensor exp_impl(const Tensor& input, bool track_grad);
Tensor log_impl(const Tensor& input, bool track_grad);
Tensor sqrt_impl(const Tensor& input, bool track_grad);
Tensor abs_impl(const Tensor& input, bool track_grad);
Tensor clamp_impl(const Tensor& input, double min_value, double max_value, bool track_grad);
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

  if (left.is_contiguous() && right.is_contiguous()) {
    matmul_contiguous_fast_path(out, left, right, rows, inner, cols);
  } else {
    for (int64_t r = 0; r < rows; ++r) {
      for (int64_t c = 0; c < cols; ++c) {
        double acc = 0.0;
        for (int64_t k = 0; k < inner; ++k) {
          const int64_t left_storage = left.offset() + r * left.strides()[0] + k * left.strides()[1];
          const int64_t right_storage = right.offset() + k * right.strides()[0] + c * right.strides()[1];
          acc += value_at_storage_unchecked(left, left_storage) * value_at_storage_unchecked(right, right_storage);
        }
        set_value_at_storage_unchecked(out, r * cols + c, acc);
      }
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
  if (input.is_contiguous()) {
    acc = sum_contiguous_fast_path(input);
  } else {
    for (int64_t i = 0; i < input.numel(); ++i) {
      acc += input.value_at_logical(i);
    }
  }
  set_value_at_storage_unchecked(out, 0, acc);
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
  if (input.is_contiguous()) {
    acc = sum_contiguous_fast_path(input);
  } else {
    for (int64_t i = 0; i < input.numel(); ++i) {
      acc += input.value_at_logical(i);
    }
  }
  set_value_at_storage_unchecked(out, 0, acc / static_cast<double>(input.numel()));
  if (track_grad) {
    set_history(out, {input}, [shape = input.shape(), n = input.numel()](const Tensor& grad) {
      return std::vector<Tensor>{full_like(shape, grad.value_at_logical(0) / static_cast<double>(n), grad.dtype())};
    });
  }
  return out;
}

Tensor extreme_grad(const Tensor& input, const Tensor& grad, double extreme_value) {
  int64_t count = 0;
  for (int64_t i = 0; i < input.numel(); ++i) {
    if (input.value_at_logical(i) == extreme_value) {
      ++count;
    }
  }
  Tensor out(input.shape(), grad.dtype(), false);
  const double scale = count == 0 ? 0.0 : grad.value_at_logical(0) / static_cast<double>(count);
  for (int64_t i = 0; i < input.numel(); ++i) {
    out.set_value_at_logical(i, input.value_at_logical(i) == extreme_value ? scale : 0.0);
  }
  return out;
}

Tensor max_impl(const Tensor& input, bool track_grad) {
  if (input.numel() == 0) {
    throw ShapeError("max is undefined for empty tensors");
  }
  Tensor out({}, input.dtype(), false);
  double value = input.value_at_logical(0);
  for (int64_t i = 1; i < input.numel(); ++i) {
    value = std::max(value, input.value_at_logical(i));
  }
  out.set_value_at_logical(0, value);
  if (track_grad) {
    set_history(out, {input}, [input, value](const Tensor& grad) {
      return std::vector<Tensor>{extreme_grad(input, grad, value)};
    });
  }
  return out;
}

Tensor min_impl(const Tensor& input, bool track_grad) {
  if (input.numel() == 0) {
    throw ShapeError("min is undefined for empty tensors");
  }
  Tensor out({}, input.dtype(), false);
  double value = input.value_at_logical(0);
  for (int64_t i = 1; i < input.numel(); ++i) {
    value = std::min(value, input.value_at_logical(i));
  }
  out.set_value_at_logical(0, value);
  if (track_grad) {
    set_history(out, {input}, [input, value](const Tensor& grad) {
      return std::vector<Tensor>{extreme_grad(input, grad, value)};
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

Tensor sqrt_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(
      input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::sqrt(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor denom = mul_impl(scalar_tensor(2.0, grad.dtype()), sqrt_impl(input, false), false);
      return std::vector<Tensor>{div_impl(grad, denom, false)};
    });
  }
  return out;
}

Tensor abs_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, input.dtype(), [](double value) { return std::abs(value); });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor sign(input.shape(), grad.dtype(), false);
      for (int64_t i = 0; i < input.numel(); ++i) {
        const double value = input.value_at_logical(i);
        sign.set_value_at_logical(i, value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0));
      }
      return std::vector<Tensor>{mul_impl(grad, sign, false)};
    });
  }
  return out;
}

Tensor clamp_impl(const Tensor& input, double min_value, double max_value, bool track_grad) {
  if (min_value > max_value) {
    throw ShapeError("clamp min_value must be <= max_value");
  }
  Tensor out = elementwise_unary_impl(input, input.dtype(), [min_value, max_value](double value) {
    return std::clamp(value, min_value, max_value);
  });
  if (track_grad) {
    set_history(out, {input}, [input, min_value, max_value](const Tensor& grad) {
      Tensor mask(input.shape(), grad.dtype(), false);
      for (int64_t i = 0; i < input.numel(); ++i) {
        const double value = input.value_at_logical(i);
        mask.set_value_at_logical(i, value >= min_value && value <= max_value ? 1.0 : 0.0);
      }
      return std::vector<Tensor>{mul_impl(grad, mask, false)};
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
  fill_storage_unchecked(out, fill_value);
  return out;
}

Tensor empty(const Shape& shape, DType dtype) {
  return Tensor::empty(shape, dtype, false);
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

Tensor eye(int64_t n, int64_t m, DType dtype) {
  if (n < 0 || m < -1) {
    throw ShapeError("eye dimensions must be non-negative");
  }
  const int64_t cols = m == -1 ? n : m;
  Tensor out({n, cols}, dtype, false);
  fill_storage_unchecked(out, 0.0);
  const int64_t diagonal = std::min(n, cols);
  for (int64_t i = 0; i < diagonal; ++i) {
    set_value_at_storage_unchecked(out, i * cols + i, 1.0);
  }
  return out;
}

Tensor linspace(double start, double stop, int64_t steps, DType dtype) {
  if (steps < 0) {
    throw ShapeError("linspace steps must be non-negative");
  }
  Tensor out({steps}, dtype, false);
  if (steps == 0) {
    return out;
  }
  if (steps == 1) {
    out.set_value_at_logical(0, start);
    return out;
  }
  const double step = (stop - start) / static_cast<double>(steps - 1);
  for (int64_t i = 0; i < steps; ++i) {
    out.set_value_at_logical(i, start + static_cast<double>(i) * step);
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

Tensor equal(const Tensor& left, const Tensor& right) {
  return elementwise_binary_impl(left, right, DType::Bool, [](double a, double b) { return a == b ? 1.0 : 0.0; }, false);
}

Tensor not_equal(const Tensor& left, const Tensor& right) {
  return elementwise_binary_impl(left, right, DType::Bool, [](double a, double b) { return a != b ? 1.0 : 0.0; }, false);
}

Tensor less(const Tensor& left, const Tensor& right) {
  return elementwise_binary_impl(left, right, DType::Bool, [](double a, double b) { return a < b ? 1.0 : 0.0; }, false);
}

Tensor less_equal(const Tensor& left, const Tensor& right) {
  return elementwise_binary_impl(left, right, DType::Bool, [](double a, double b) { return a <= b ? 1.0 : 0.0; }, false);
}

Tensor greater(const Tensor& left, const Tensor& right) {
  return elementwise_binary_impl(left, right, DType::Bool, [](double a, double b) { return a > b ? 1.0 : 0.0; }, false);
}

Tensor greater_equal(const Tensor& left, const Tensor& right) {
  return elementwise_binary_impl(left, right, DType::Bool, [](double a, double b) { return a >= b ? 1.0 : 0.0; }, false);
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

Tensor max(const Tensor& input) {
  return max_impl(input, true);
}

Tensor min(const Tensor& input) {
  return min_impl(input, true);
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

Tensor sqrt(const Tensor& input) {
  return sqrt_impl(input, true);
}

Tensor abs(const Tensor& input) {
  return abs_impl(input, true);
}

Tensor clamp(const Tensor& input, double min_value, double max_value) {
  return clamp_impl(input, min_value, max_value, true);
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
