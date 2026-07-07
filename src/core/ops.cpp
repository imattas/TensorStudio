#include "tensorstudio/ops.hpp"

#if defined(TENSORSTUDIO_HAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#elif defined(TENSORSTUDIO_HAS_CBLAS)
#include <cblas.h>
#endif

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <mutex>
#include <numeric>
#include <type_traits>
#include <vector>

#include "tensorstudio/autograd.hpp"
#include "tensorstudio/errors.hpp"
#include "tensorstudio/perf.hpp"

namespace tensorstudio {

TensorIndex TensorIndex::at(int64_t index_value) {
  TensorIndex item;
  item.kind = TensorIndexKind::Index;
  item.index = index_value;
  return item;
}

TensorIndex TensorIndex::slice(int64_t start_value, int64_t length_value, int64_t step_value) {
  if (step_value == 0) {
    throw ShapeError("slice step cannot be zero");
  }
  if (length_value < 0) {
    throw ShapeError("slice length must be non-negative");
  }
  TensorIndex item;
  item.kind = TensorIndexKind::Slice;
  item.start = start_value;
  item.length = length_value;
  item.step = step_value;
  return item;
}

TensorIndex TensorIndex::new_axis() {
  TensorIndex item;
  item.kind = TensorIndexKind::NewAxis;
  return item;
}

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

struct AddOp {
  double operator()(double a, double b) const {
    return a + b;
  }
  template <typename T>
  T typed(T a, T b) const {
    return a + b;
  }
};

struct SubOp {
  double operator()(double a, double b) const {
    return a - b;
  }
  template <typename T>
  T typed(T a, T b) const {
    return a - b;
  }
};

struct MulOp {
  double operator()(double a, double b) const {
    return a * b;
  }
  template <typename T>
  T typed(T a, T b) const {
    return a * b;
  }
};

struct DivOp {
  double operator()(double a, double b) const {
    return a / b;
  }
  template <typename T>
  T typed(T a, T b) const {
    return a / b;
  }
};

struct MaximumOp {
  double operator()(double a, double b) const {
    return std::max(a, b);
  }
  template <typename T>
  T typed(T a, T b) const {
    return std::max(a, b);
  }
};

struct MinimumOp {
  double operator()(double a, double b) const {
    return std::min(a, b);
  }
  template <typename T>
  T typed(T a, T b) const {
    return std::min(a, b);
  }
};

struct NegOp {
  double operator()(double value) const {
    return -value;
  }
  template <typename T>
  T typed(T value) const {
    return -value;
  }
};

struct ReluOp {
  double operator()(double value) const {
    return std::max(0.0, value);
  }
  template <typename T>
  T typed(T value) const {
    return std::max<T>(static_cast<T>(0), value);
  }
};

struct SigmoidOp {
  double operator()(double value) const {
    return 1.0 / (1.0 + std::exp(-value));
  }
  template <typename T>
  T typed(T value) const {
    return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-value));
  }
};

struct TanhOp {
  double operator()(double value) const {
    return std::tanh(value);
  }
  template <typename T>
  T typed(T value) const {
    return std::tanh(value);
  }
};

struct ExpOp {
  double operator()(double value) const {
    return std::exp(value);
  }
  template <typename T>
  T typed(T value) const {
    return std::exp(value);
  }
};

struct LogOp {
  double operator()(double value) const {
    return std::log(value);
  }
  template <typename T>
  T typed(T value) const {
    return std::log(value);
  }
};

template <typename Op, typename T>
auto typed_binary_apply(const Op& op, T a, T b, int) -> decltype(op.template typed<T>(a, b)) {
  return op.template typed<T>(a, b);
}

template <typename Op, typename T>
T typed_binary_apply(const Op& op, T a, T b, long) {
  return from_double<T>(op(to_double(a), to_double(b)));
}

template <typename Op, typename T>
auto typed_unary_apply(const Op& op, T value, int) -> decltype(op.template typed<T>(value)) {
  return op.template typed<T>(value);
}

template <typename Op, typename T>
T typed_unary_apply(const Op& op, T value, long) {
  return from_double<T>(op(to_double(value)));
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
  perf::parallel_for(0, count, 16 * 1024, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      if constexpr (
          std::is_same_v<OutT, LeftT> && std::is_same_v<OutT, RightT> &&
          (std::is_same_v<OutT, float> || std::is_same_v<OutT, double>)) {
        out_data[i] = typed_binary_apply(op, left_data[i], right_data[i], 0);
      } else {
        out_data[i] = from_double<OutT>(op(to_double(left_data[i]), to_double(right_data[i])));
      }
    }
  });
}

template <typename OutT, typename LeftT, typename RightT, typename Op>
void binary_left_scalar_kernel(Tensor& out, const Tensor& left, const Tensor& right, Op op) {
  const auto count = out.numel();
  auto* out_data = raw_data<OutT>(out) + out.offset();
  const double left_value = to_double(raw_data<LeftT>(left)[static_cast<std::size_t>(left.offset())]);
  const LeftT typed_left_value = raw_data<LeftT>(left)[static_cast<std::size_t>(left.offset())];
  const auto* right_data = raw_data<RightT>(right) + right.offset();
  perf::parallel_for(0, count, 16 * 1024, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      if constexpr (
          std::is_same_v<OutT, LeftT> && std::is_same_v<OutT, RightT> &&
          (std::is_same_v<OutT, float> || std::is_same_v<OutT, double>)) {
        out_data[i] = typed_binary_apply(op, typed_left_value, right_data[i], 0);
      } else {
        out_data[i] = from_double<OutT>(op(left_value, to_double(right_data[i])));
      }
    }
  });
}

template <typename OutT, typename LeftT, typename RightT, typename Op>
void binary_right_scalar_kernel(Tensor& out, const Tensor& left, const Tensor& right, Op op) {
  const auto count = out.numel();
  auto* out_data = raw_data<OutT>(out) + out.offset();
  const auto* left_data = raw_data<LeftT>(left) + left.offset();
  const double right_value = to_double(raw_data<RightT>(right)[static_cast<std::size_t>(right.offset())]);
  const RightT typed_right_value = raw_data<RightT>(right)[static_cast<std::size_t>(right.offset())];
  perf::parallel_for(0, count, 16 * 1024, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      if constexpr (
          std::is_same_v<OutT, LeftT> && std::is_same_v<OutT, RightT> &&
          (std::is_same_v<OutT, float> || std::is_same_v<OutT, double>)) {
        out_data[i] = typed_binary_apply(op, left_data[i], typed_right_value, 0);
      } else {
        out_data[i] = from_double<OutT>(op(to_double(left_data[i]), right_value));
      }
    }
  });
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
  perf::parallel_for(0, count, 16 * 1024, [&](int64_t begin, int64_t end) {
    for (int64_t i = begin; i < end; ++i) {
      if constexpr (
          std::is_same_v<OutT, InT> && (std::is_same_v<OutT, float> || std::is_same_v<OutT, double>)) {
        out_data[i] = typed_unary_apply(op, input_data[i], 0);
      } else {
        out_data[i] = from_double<OutT>(op(to_double(input_data[i])));
      }
    }
  });
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
  const int64_t count = input.numel();
  if (perf::threads_enabled() && count > 32 * 1024) {
    std::mutex mutex;
    perf::parallel_for(0, count, 32 * 1024, [&](int64_t begin, int64_t end) {
      double local = 0.0;
      for (int64_t i = begin; i < end; ++i) {
        local += to_double(data[i]);
      }
      std::lock_guard<std::mutex> lock(mutex);
      acc += local;
    });
  } else {
    for (int64_t i = 0; i < count; ++i) {
      acc += to_double(data[i]);
    }
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

template <typename T>
void sum_axis2d_contiguous_kernel(std::vector<double>& values, const Tensor& input, int64_t axis) {
  const auto* data = raw_data<T>(input) + input.offset();
  const int64_t rows = input.shape()[0];
  const int64_t cols = input.shape()[1];
  if (axis == 1) {
    perf::parallel_for(0, rows, 128, [&](int64_t row_begin, int64_t row_end) {
      for (int64_t row = row_begin; row < row_end; ++row) {
        double acc = 0.0;
        const int64_t row_offset = row * cols;
        for (int64_t col = 0; col < cols; ++col) {
          acc += to_double(data[row_offset + col]);
        }
        values[static_cast<std::size_t>(row)] = acc;
      }
    });
    return;
  }

  for (int64_t row = 0; row < rows; ++row) {
    const int64_t row_offset = row * cols;
    for (int64_t col = 0; col < cols; ++col) {
      values[static_cast<std::size_t>(col)] += to_double(data[row_offset + col]);
    }
  }
}

void sum_axis2d_contiguous_values(std::vector<double>& values, const Tensor& input, int64_t axis) {
  switch (input.dtype()) {
    case DType::Float32:
      sum_axis2d_contiguous_kernel<float>(values, input, axis);
      return;
    case DType::Float64:
      sum_axis2d_contiguous_kernel<double>(values, input, axis);
      return;
    case DType::Int32:
      sum_axis2d_contiguous_kernel<int32_t>(values, input, axis);
      return;
    case DType::Int64:
      sum_axis2d_contiguous_kernel<int64_t>(values, input, axis);
      return;
    case DType::Bool:
      sum_axis2d_contiguous_kernel<bool>(values, input, axis);
      return;
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

bool blas_dimensions_supported(int64_t rows, int64_t inner, int64_t cols) {
  constexpr int64_t max_blas_int = static_cast<int64_t>(std::numeric_limits<int>::max());
  return rows <= max_blas_int && inner <= max_blas_int && cols <= max_blas_int;
}

bool try_blas_matmul(
    Tensor& out,
    const Tensor& left,
    const Tensor& right,
    int64_t rows,
    int64_t inner,
    int64_t cols) {
#if defined(TENSORSTUDIO_HAS_CBLAS) || defined(TENSORSTUDIO_HAS_ACCELERATE)
  if (!blas_dimensions_supported(rows, inner, cols)) {
    return false;
  }
  if (left.dtype() == DType::Float32 && right.dtype() == DType::Float32 && out.dtype() == DType::Float32) {
    const auto m = static_cast<int>(rows);
    const auto n = static_cast<int>(cols);
    const auto k = static_cast<int>(inner);
    cblas_sgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m,
        n,
        k,
        1.0F,
        raw_data<float>(left) + left.offset(),
        k,
        raw_data<float>(right) + right.offset(),
        n,
        0.0F,
        raw_data<float>(out) + out.offset(),
        n);
    return true;
  }
  if (left.dtype() == DType::Float64 && right.dtype() == DType::Float64 && out.dtype() == DType::Float64) {
    const auto m = static_cast<int>(rows);
    const auto n = static_cast<int>(cols);
    const auto k = static_cast<int>(inner);
    cblas_dgemm(
        CblasRowMajor,
        CblasNoTrans,
        CblasNoTrans,
        m,
        n,
        k,
        1.0,
        raw_data<double>(left) + left.offset(),
        k,
        raw_data<double>(right) + right.offset(),
        n,
        0.0,
        raw_data<double>(out) + out.offset(),
        n);
    return true;
  }
#else
  (void)out;
  (void)left;
  (void)right;
  (void)rows;
  (void)inner;
  (void)cols;
#endif
  return false;
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
      perf::parallel_for(0, rows, 8, [&](int64_t row_begin, int64_t row_end) {
        for (int64_t r = row_begin; r < row_end; ++r) {
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
      });
      auto* out_data = raw_data<float>(out) + out.offset();
      std::copy(accumulator.begin(), accumulator.end(), out_data);
      return;
    }
  }

  std::vector<double> accumulator(static_cast<std::size_t>(rows * cols), 0.0);

  perf::parallel_for(0, rows, 8, [&](int64_t row_begin, int64_t row_end) {
    for (int64_t r = row_begin; r < row_end; ++r) {
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
  });

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
Tensor maximum_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor minimum_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor where_impl(const Tensor& condition, const Tensor& true_value, const Tensor& false_value, bool track_grad);
Tensor neg_impl(const Tensor& input, bool track_grad);
Tensor pow_impl(const Tensor& input, double exponent, bool track_grad);
Tensor matmul_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor bmm_impl(const Tensor& left, const Tensor& right, bool track_grad);
Tensor conv2d_impl(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    bool track_grad);
Tensor max_pool2d_impl(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    bool track_grad);
Tensor avg_pool2d_impl(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool count_include_pad,
    bool track_grad);
Tensor sum_impl(const Tensor& input, bool track_grad);
Tensor sum_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad);
Tensor mean_impl(const Tensor& input, bool track_grad);
Tensor mean_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad);
Tensor variance_impl(const Tensor& input, int64_t correction, bool track_grad);
Tensor variance_axis_impl(const Tensor& input, int64_t axis, bool keepdims, int64_t correction, bool track_grad);
Tensor stddev_impl(const Tensor& input, int64_t correction, bool track_grad);
Tensor stddev_axis_impl(const Tensor& input, int64_t axis, bool keepdims, int64_t correction, bool track_grad);
Tensor max_impl(const Tensor& input, bool track_grad);
Tensor max_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad);
Tensor min_impl(const Tensor& input, bool track_grad);
Tensor min_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad);
Tensor all_impl(const Tensor& input, bool keepdims);
Tensor all_axis_impl(const Tensor& input, int64_t axis, bool keepdims);
Tensor any_impl(const Tensor& input, bool keepdims);
Tensor any_axis_impl(const Tensor& input, int64_t axis, bool keepdims);
Tensor relu_impl(const Tensor& input, bool track_grad);
Tensor sigmoid_impl(const Tensor& input, bool track_grad);
Tensor tanh_impl(const Tensor& input, bool track_grad);
Tensor exp_impl(const Tensor& input, bool track_grad);
Tensor log_impl(const Tensor& input, bool track_grad);
Tensor logsumexp_impl(const Tensor& input, bool track_grad);
Tensor logsumexp_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad);
Tensor softmax_impl(const Tensor& input, int64_t axis, bool track_grad);
Tensor log_softmax_impl(const Tensor& input, int64_t axis, bool track_grad);
Tensor log1p_impl(const Tensor& input, bool track_grad);
Tensor sqrt_impl(const Tensor& input, bool track_grad);
Tensor rsqrt_impl(const Tensor& input, bool track_grad);
Tensor sin_impl(const Tensor& input, bool track_grad);
Tensor cos_impl(const Tensor& input, bool track_grad);
Tensor tan_impl(const Tensor& input, bool track_grad);
Tensor asin_impl(const Tensor& input, bool track_grad);
Tensor acos_impl(const Tensor& input, bool track_grad);
Tensor atan_impl(const Tensor& input, bool track_grad);
Tensor abs_impl(const Tensor& input, bool track_grad);
Tensor clamp_impl(const Tensor& input, double min_value, double max_value, bool track_grad);
Tensor astype_impl(const Tensor& input, DType dtype, bool track_grad);
Tensor concat_impl(const std::vector<Tensor>& tensors, int64_t axis, bool track_grad);
Tensor stack_impl(const std::vector<Tensor>& tensors, int64_t axis, bool track_grad);
Tensor reshape_impl(const Tensor& input, const Shape& shape, bool track_grad);
Tensor transpose_impl(const Tensor& input, bool track_grad);
Tensor transpose_impl(const Tensor& input, int64_t axis0, int64_t axis1, bool track_grad);
Tensor permute_impl(const Tensor& input, const Shape& axes, bool track_grad);
Tensor squeeze_impl(const Tensor& input, std::optional<int64_t> axis, bool track_grad);
Tensor unsqueeze_impl(const Tensor& input, int64_t axis, bool track_grad);
Tensor index_impl(const Tensor& input, const std::vector<TensorIndex>& indices, bool track_grad);

Tensor add_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left, right, promote_types(left.dtype(), right.dtype()), AddOp{}, track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left_shape = left.shape(), right_shape = right.shape()](const Tensor& grad) {
      return std::vector<Tensor>{unbroadcast(grad, left_shape), unbroadcast(grad, right_shape)};
    });
  }
  return out;
}

Tensor sub_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left, right, promote_types(left.dtype(), right.dtype()), SubOp{}, track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left_shape = left.shape(), right_shape = right.shape()](const Tensor& grad) {
      return std::vector<Tensor>{unbroadcast(grad, left_shape), unbroadcast(neg_impl(grad, false), right_shape)};
    });
  }
  return out;
}

Tensor mul_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left, right, promote_types(left.dtype(), right.dtype()), MulOp{}, track_grad);
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
  Tensor out = elementwise_binary_impl(left, right, dtype, DivOp{}, track_grad);
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

Tensor comparison_mask_impl(
    const Tensor& left,
    const Tensor& right,
    DType dtype,
    bool is_max,
    bool for_left) {
  const Shape out_shape = broadcast_shapes(left.shape(), right.shape());
  Tensor out(out_shape, dtype, false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    const auto left_storage =
        logical_to_storage_offset(i, out_shape, left.shape(), left.strides(), left.offset());
    const auto right_storage =
        logical_to_storage_offset(i, out_shape, right.shape(), right.strides(), right.offset());
    const double a = value_at_storage_unchecked(left, left_storage);
    const double b = value_at_storage_unchecked(right, right_storage);
    double value = 0.0;
    if (a == b) {
      value = 0.5;
    } else if (is_max) {
      value = for_left ? (a > b ? 1.0 : 0.0) : (b > a ? 1.0 : 0.0);
    } else {
      value = for_left ? (a < b ? 1.0 : 0.0) : (b < a ? 1.0 : 0.0);
    }
    out.set_value_at_logical(i, value);
  }
  return out;
}

Tensor maximum_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left,
      right,
      promote_types(left.dtype(), right.dtype()),
      MaximumOp{},
      track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left, right](const Tensor& grad) {
      Tensor left_mask = comparison_mask_impl(left, right, grad.dtype(), true, true);
      Tensor right_mask = comparison_mask_impl(left, right, grad.dtype(), true, false);
      Tensor left_grad = unbroadcast(mul_impl(grad, left_mask, false), left.shape());
      Tensor right_grad = unbroadcast(mul_impl(grad, right_mask, false), right.shape());
      return std::vector<Tensor>{left_grad, right_grad};
    });
  }
  return out;
}

Tensor minimum_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  Tensor out = elementwise_binary_impl(
      left,
      right,
      promote_types(left.dtype(), right.dtype()),
      MinimumOp{},
      track_grad);
  if (track_grad) {
    set_history(out, {left, right}, [left, right](const Tensor& grad) {
      Tensor left_mask = comparison_mask_impl(left, right, grad.dtype(), false, true);
      Tensor right_mask = comparison_mask_impl(left, right, grad.dtype(), false, false);
      Tensor left_grad = unbroadcast(mul_impl(grad, left_mask, false), left.shape());
      Tensor right_grad = unbroadcast(mul_impl(grad, right_mask, false), right.shape());
      return std::vector<Tensor>{left_grad, right_grad};
    });
  }
  return out;
}

Tensor condition_mask_impl(
    const Tensor& condition,
    const Shape& out_shape,
    bool selected_when_true,
    DType dtype) {
  Tensor out(out_shape, dtype, false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    const auto condition_storage =
        logical_to_storage_offset(
            i,
            out_shape,
            condition.shape(),
            condition.strides(),
            condition.offset());
    const bool selected = value_at_storage_unchecked(condition, condition_storage) != 0.0;
    out.set_value_at_logical(i, selected == selected_when_true ? 1.0 : 0.0);
  }
  return out;
}

Tensor where_impl(
    const Tensor& condition,
    const Tensor& true_value,
    const Tensor& false_value,
    bool track_grad) {
  if (condition.dtype() != DType::Bool) {
    throw DTypeError("where condition must have dtype bool, got " + dtype_name(condition.dtype()));
  }
  const Shape branch_shape = broadcast_shapes(true_value.shape(), false_value.shape());
  const Shape out_shape = broadcast_shapes(condition.shape(), branch_shape);
  Tensor out(out_shape, promote_types(true_value.dtype(), false_value.dtype()), false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    const auto condition_storage =
        logical_to_storage_offset(
            i,
            out_shape,
            condition.shape(),
            condition.strides(),
            condition.offset());
    const auto true_storage =
        logical_to_storage_offset(i, out_shape, true_value.shape(), true_value.strides(), true_value.offset());
    const auto false_storage =
        logical_to_storage_offset(i, out_shape, false_value.shape(), false_value.strides(), false_value.offset());
    const bool selected = value_at_storage_unchecked(condition, condition_storage) != 0.0;
    out.set_value_at_logical(
        i,
        selected ? value_at_storage_unchecked(true_value, true_storage)
                 : value_at_storage_unchecked(false_value, false_storage));
  }
  if (track_grad) {
    set_history(
        out,
        {true_value, false_value},
        [condition, true_shape = true_value.shape(), false_shape = false_value.shape()](
            const Tensor& grad) {
          Tensor true_mask = condition_mask_impl(condition, grad.shape(), true, grad.dtype());
          Tensor false_mask = condition_mask_impl(condition, grad.shape(), false, grad.dtype());
          Tensor true_grad = unbroadcast(mul_impl(grad, true_mask, false), true_shape);
          Tensor false_grad = unbroadcast(mul_impl(grad, false_mask, false), false_shape);
          return std::vector<Tensor>{true_grad, false_grad};
        });
  }
  return out;
}

Tensor neg_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, input.dtype(), NegOp{});
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
  if (left.ndim() == 3 && right.ndim() == 3) {
    return bmm_impl(left, right, track_grad);
  }
  if (left.ndim() != 2 || right.ndim() != 2) {
    throw ShapeError(
        "matmul expects two 2D tensors or two 3D batched tensors, got " + shape_to_string(left.shape()) + " and " +
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
    if (!try_blas_matmul(out, left, right, rows, inner, cols)) {
      matmul_contiguous_fast_path(out, left, right, rows, inner, cols);
    }
  } else {
    perf::parallel_for(0, rows, 8, [&](int64_t row_begin, int64_t row_end) {
      for (int64_t r = row_begin; r < row_end; ++r) {
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
    });
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

Tensor transpose_last_two_3d(const Tensor& input) {
  Tensor out({input.shape()[0], input.shape()[2], input.shape()[1]}, input.dtype(), false);
  for (int64_t b = 0; b < input.shape()[0]; ++b) {
    for (int64_t row = 0; row < input.shape()[1]; ++row) {
      for (int64_t col = 0; col < input.shape()[2]; ++col) {
        const int64_t input_linear = b * input.shape()[1] * input.shape()[2] + row * input.shape()[2] + col;
        const int64_t out_linear = b * out.shape()[1] * out.shape()[2] + col * out.shape()[2] + row;
        out.set_value_at_logical(out_linear, input.value_at_logical(input_linear));
      }
    }
  }
  return out;
}

Tensor bmm_impl(const Tensor& left, const Tensor& right, bool track_grad) {
  if (left.ndim() != 3 || right.ndim() != 3) {
    throw ShapeError(
        "bmm expects two 3D tensors, got " + shape_to_string(left.shape()) + " and " +
        shape_to_string(right.shape()));
  }
  if (left.shape()[0] != right.shape()[0] || left.shape()[2] != right.shape()[1]) {
    throw ShapeError(
        "bmm shape mismatch: " + shape_to_string(left.shape()) + " @ " + shape_to_string(right.shape()));
  }

  const int64_t batch = left.shape()[0];
  const int64_t rows = left.shape()[1];
  const int64_t inner = left.shape()[2];
  const int64_t cols = right.shape()[2];
  Tensor out({batch, rows, cols}, promote_types(left.dtype(), right.dtype()), false);

  perf::parallel_for(0, batch * rows, 4, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t b = task / rows;
      const int64_t r = task % rows;
      for (int64_t c = 0; c < cols; ++c) {
        double acc = 0.0;
        for (int64_t k = 0; k < inner; ++k) {
          const int64_t left_storage =
              left.offset() + b * left.strides()[0] + r * left.strides()[1] + k * left.strides()[2];
          const int64_t right_storage =
              right.offset() + b * right.strides()[0] + k * right.strides()[1] + c * right.strides()[2];
          acc += value_at_storage_unchecked(left, left_storage) * value_at_storage_unchecked(right, right_storage);
        }
        out.set_value_at_logical(b * rows * cols + r * cols + c, acc);
      }
    }
  });

  if (track_grad) {
    set_history(out, {left, right}, [left, right](const Tensor& grad) {
      Tensor left_grad = bmm_impl(grad, transpose_last_two_3d(right), false);
      Tensor right_grad = bmm_impl(transpose_last_two_3d(left), grad, false);
      return std::vector<Tensor>{left_grad, right_grad};
    });
  }
  return out;
}

int64_t storage_offset_1d(const Tensor& tensor, int64_t i0) {
  return tensor.offset() + i0 * tensor.strides()[0];
}

int64_t storage_offset_4d(
    const Tensor& tensor,
    int64_t i0,
    int64_t i1,
    int64_t i2,
    int64_t i3) {
  return tensor.offset() + i0 * tensor.strides()[0] + i1 * tensor.strides()[1] +
         i2 * tensor.strides()[2] + i3 * tensor.strides()[3];
}

void validate_conv2d_inputs(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  if (input.ndim() != 4) {
    throw ShapeError("conv2d input must be NCHW with rank 4, got " + shape_to_string(input.shape()));
  }
  if (weight.ndim() != 4) {
    throw ShapeError("conv2d weight must have shape (out_channels, in_channels, kernel_h, kernel_w), got " +
                     shape_to_string(weight.shape()));
  }
  if (stride_h <= 0 || stride_w <= 0) {
    throw ShapeError("conv2d stride values must be positive");
  }
  if (padding_h < 0 || padding_w < 0) {
    throw ShapeError("conv2d padding values must be non-negative");
  }
  if (dilation_h <= 0 || dilation_w <= 0) {
    throw ShapeError("conv2d dilation values must be positive");
  }
  if (input.shape()[1] != weight.shape()[1]) {
    throw ShapeError(
        "conv2d channel mismatch: input shape " + shape_to_string(input.shape()) +
        " and weight shape " + shape_to_string(weight.shape()));
  }
  if (weight.shape()[2] <= 0 || weight.shape()[3] <= 0) {
    throw ShapeError("conv2d kernel dimensions must be positive");
  }
  if (bias.has_value()) {
    const Tensor& bias_tensor = *bias;
    if (bias_tensor.ndim() != 1 || bias_tensor.shape()[0] != weight.shape()[0]) {
      throw ShapeError(
          "conv2d bias must have shape (" + std::to_string(weight.shape()[0]) +
          ",), got " + shape_to_string(bias_tensor.shape()));
    }
  }
}

Shape conv2d_output_shape(
    const Tensor& input,
    const Tensor& weight,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  const int64_t batch = input.shape()[0];
  const int64_t out_channels = weight.shape()[0];
  const int64_t input_h = input.shape()[2];
  const int64_t input_w = input.shape()[3];
  const int64_t kernel_h = weight.shape()[2];
  const int64_t kernel_w = weight.shape()[3];
  const int64_t effective_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t effective_w = dilation_w * (kernel_w - 1) + 1;
  const int64_t out_h = (input_h + 2 * padding_h - effective_h) / stride_h + 1;
  const int64_t out_w = (input_w + 2 * padding_w - effective_w) / stride_w + 1;
  if (out_h <= 0 || out_w <= 0) {
    throw ShapeError(
        "conv2d calculated non-positive output shape from input " + shape_to_string(input.shape()) +
        ", weight " + shape_to_string(weight.shape()) + ", stride=(" + std::to_string(stride_h) +
        ", " + std::to_string(stride_w) + "), padding=(" + std::to_string(padding_h) +
        ", " + std::to_string(padding_w) + "), dilation=(" + std::to_string(dilation_h) +
        ", " + std::to_string(dilation_w) + ")");
  }
  return Shape{batch, out_channels, out_h, out_w};
}

void conv2d_forward_kernel(
    Tensor& out,
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  const int64_t batch = input.shape()[0];
  const int64_t in_channels = input.shape()[1];
  const int64_t input_h = input.shape()[2];
  const int64_t input_w = input.shape()[3];
  const int64_t out_channels = weight.shape()[0];
  const int64_t kernel_h = weight.shape()[2];
  const int64_t kernel_w = weight.shape()[3];
  const int64_t out_h = out.shape()[2];
  const int64_t out_w = out.shape()[3];

  perf::parallel_for(0, batch * out_channels, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t n = task / out_channels;
      const int64_t oc = task % out_channels;
      const double bias_value =
          bias.has_value() ? value_at_storage_unchecked(*bias, storage_offset_1d(*bias, oc)) : 0.0;
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          double acc = bias_value;
          for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
              const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
              if (ih < 0 || ih >= input_h) {
                continue;
              }
              for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
                if (iw < 0 || iw >= input_w) {
                  continue;
                }
                const double input_value =
                    value_at_storage_unchecked(input, storage_offset_4d(input, n, ic, ih, iw));
                const double weight_value =
                    value_at_storage_unchecked(weight, storage_offset_4d(weight, oc, ic, kh, kw));
                acc += input_value * weight_value;
              }
            }
          }
          set_value_at_storage_unchecked(out, storage_offset_4d(out, n, oc, oh, ow), acc);
        }
      }
    }
  });
}

Tensor conv2d_input_grad(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  Tensor input_grad(input.shape(), grad.dtype(), false);
  fill_storage_unchecked(input_grad, 0.0);

  const int64_t batch = input.shape()[0];
  const int64_t in_channels = input.shape()[1];
  const int64_t input_h = input.shape()[2];
  const int64_t input_w = input.shape()[3];
  const int64_t out_channels = weight.shape()[0];
  const int64_t kernel_h = weight.shape()[2];
  const int64_t kernel_w = weight.shape()[3];
  const int64_t out_h = grad.shape()[2];
  const int64_t out_w = grad.shape()[3];

  for (int64_t n = 0; n < batch; ++n) {
    for (int64_t oc = 0; oc < out_channels; ++oc) {
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          const double grad_value = value_at_storage_unchecked(grad, storage_offset_4d(grad, n, oc, oh, ow));
          for (int64_t ic = 0; ic < in_channels; ++ic) {
            for (int64_t kh = 0; kh < kernel_h; ++kh) {
              const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
              if (ih < 0 || ih >= input_h) {
                continue;
              }
              for (int64_t kw = 0; kw < kernel_w; ++kw) {
                const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
                if (iw < 0 || iw >= input_w) {
                  continue;
                }
                const int64_t input_grad_offset = storage_offset_4d(input_grad, n, ic, ih, iw);
                const double current = value_at_storage_unchecked(input_grad, input_grad_offset);
                const double weight_value =
                    value_at_storage_unchecked(weight, storage_offset_4d(weight, oc, ic, kh, kw));
                set_value_at_storage_unchecked(input_grad, input_grad_offset, current + grad_value * weight_value);
              }
            }
          }
        }
      }
    }
  }
  return input_grad;
}

Tensor conv2d_weight_grad(
    const Tensor& grad,
    const Tensor& input,
    const Tensor& weight,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  Tensor weight_grad(weight.shape(), grad.dtype(), false);

  const int64_t batch = input.shape()[0];
  const int64_t in_channels = input.shape()[1];
  const int64_t input_h = input.shape()[2];
  const int64_t input_w = input.shape()[3];
  const int64_t out_channels = weight.shape()[0];
  const int64_t kernel_h = weight.shape()[2];
  const int64_t kernel_w = weight.shape()[3];
  const int64_t out_h = grad.shape()[2];
  const int64_t out_w = grad.shape()[3];

  for (int64_t oc = 0; oc < out_channels; ++oc) {
    for (int64_t ic = 0; ic < in_channels; ++ic) {
      for (int64_t kh = 0; kh < kernel_h; ++kh) {
        for (int64_t kw = 0; kw < kernel_w; ++kw) {
          double acc = 0.0;
          for (int64_t n = 0; n < batch; ++n) {
            for (int64_t oh = 0; oh < out_h; ++oh) {
              const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
              if (ih < 0 || ih >= input_h) {
                continue;
              }
              for (int64_t ow = 0; ow < out_w; ++ow) {
                const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
                if (iw < 0 || iw >= input_w) {
                  continue;
                }
                const double input_value =
                    value_at_storage_unchecked(input, storage_offset_4d(input, n, ic, ih, iw));
                const double grad_value = value_at_storage_unchecked(grad, storage_offset_4d(grad, n, oc, oh, ow));
                acc += input_value * grad_value;
              }
            }
          }
          set_value_at_storage_unchecked(weight_grad, storage_offset_4d(weight_grad, oc, ic, kh, kw), acc);
        }
      }
    }
  }
  return weight_grad;
}

Tensor conv2d_bias_grad(const Tensor& grad) {
  Tensor bias_grad({grad.shape()[1]}, grad.dtype(), false);
  for (int64_t oc = 0; oc < grad.shape()[1]; ++oc) {
    double acc = 0.0;
    for (int64_t n = 0; n < grad.shape()[0]; ++n) {
      for (int64_t oh = 0; oh < grad.shape()[2]; ++oh) {
        for (int64_t ow = 0; ow < grad.shape()[3]; ++ow) {
          acc += value_at_storage_unchecked(grad, storage_offset_4d(grad, n, oc, oh, ow));
        }
      }
    }
    set_value_at_storage_unchecked(bias_grad, storage_offset_1d(bias_grad, oc), acc);
  }
  return bias_grad;
}

Tensor conv2d_impl(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    bool track_grad) {
  validate_conv2d_inputs(input, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
  DType out_dtype = promote_types(input.dtype(), weight.dtype());
  if (bias.has_value()) {
    out_dtype = promote_types(out_dtype, bias->dtype());
  }
  Tensor out(conv2d_output_shape(input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w),
             out_dtype,
             false);
  conv2d_forward_kernel(out, input, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);

  if (track_grad) {
    std::vector<Tensor> parents{input, weight};
    if (bias.has_value()) {
      parents.push_back(*bias);
    }
    set_history(
        out,
        parents,
        [input, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w](
            const Tensor& grad) {
          std::vector<Tensor> gradients{
              conv2d_input_grad(grad, input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w),
              conv2d_weight_grad(grad, input, weight, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w),
          };
          if (bias.has_value()) {
            gradients.push_back(conv2d_bias_grad(grad));
          }
          return gradients;
        });
  }
  return out;
}

void validate_pool2d_inputs(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    const std::string& op_name) {
  if (input.ndim() != 4) {
    throw ShapeError(op_name + " input must be NCHW with rank 4, got " + shape_to_string(input.shape()));
  }
  if (kernel_h <= 0 || kernel_w <= 0) {
    throw ShapeError(op_name + " kernel_size values must be positive");
  }
  if (stride_h <= 0 || stride_w <= 0) {
    throw ShapeError(op_name + " stride values must be positive");
  }
  if (padding_h < 0 || padding_w < 0) {
    throw ShapeError(op_name + " padding values must be non-negative");
  }
  if (dilation_h <= 0 || dilation_w <= 0) {
    throw ShapeError(op_name + " dilation values must be positive");
  }
  const int64_t effective_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t effective_w = dilation_w * (kernel_w - 1) + 1;
  if (padding_h >= effective_h || padding_w >= effective_w) {
    throw ShapeError(op_name + " padding must be smaller than the effective kernel size");
  }
}

Shape pool2d_output_shape(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    const std::string& op_name) {
  const int64_t effective_h = dilation_h * (kernel_h - 1) + 1;
  const int64_t effective_w = dilation_w * (kernel_w - 1) + 1;
  const int64_t out_h = (input.shape()[2] + 2 * padding_h - effective_h) / stride_h + 1;
  const int64_t out_w = (input.shape()[3] + 2 * padding_w - effective_w) / stride_w + 1;
  if (out_h <= 0 || out_w <= 0) {
    throw ShapeError(
        op_name + " calculated non-positive output shape from input " + shape_to_string(input.shape()) +
        ", kernel_size=(" + std::to_string(kernel_h) + ", " + std::to_string(kernel_w) +
        "), stride=(" + std::to_string(stride_h) + ", " + std::to_string(stride_w) +
        "), padding=(" + std::to_string(padding_h) + ", " + std::to_string(padding_w) +
        "), dilation=(" + std::to_string(dilation_h) + ", " + std::to_string(dilation_w) + ")");
  }
  return Shape{input.shape()[0], input.shape()[1], out_h, out_w};
}

void max_pool2d_forward_kernel(
    Tensor& out,
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  const int64_t batch = input.shape()[0];
  const int64_t channels = input.shape()[1];
  const int64_t input_h = input.shape()[2];
  const int64_t input_w = input.shape()[3];
  const int64_t out_h = out.shape()[2];
  const int64_t out_w = out.shape()[3];

  perf::parallel_for(0, batch * channels, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t n = task / channels;
      const int64_t c = task % channels;
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          double max_value = -std::numeric_limits<double>::infinity();
          bool found = false;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
            if (ih < 0 || ih >= input_h) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
              if (iw < 0 || iw >= input_w) {
                continue;
              }
              const double value = value_at_storage_unchecked(input, storage_offset_4d(input, n, c, ih, iw));
              max_value = found ? std::max(max_value, value) : value;
              found = true;
            }
          }
          if (!found) {
            throw ShapeError("max_pool2d window did not overlap input");
          }
          set_value_at_storage_unchecked(out, storage_offset_4d(out, n, c, oh, ow), max_value);
        }
      }
    }
  });
}

void avg_pool2d_forward_kernel(
    Tensor& out,
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool count_include_pad) {
  const int64_t batch = input.shape()[0];
  const int64_t channels = input.shape()[1];
  const int64_t input_h = input.shape()[2];
  const int64_t input_w = input.shape()[3];
  const int64_t out_h = out.shape()[2];
  const int64_t out_w = out.shape()[3];
  const double padded_window_count = static_cast<double>(kernel_h * kernel_w);

  perf::parallel_for(0, batch * channels, 1, [&](int64_t begin, int64_t end) {
    for (int64_t task = begin; task < end; ++task) {
      const int64_t n = task / channels;
      const int64_t c = task % channels;
      for (int64_t oh = 0; oh < out_h; ++oh) {
        for (int64_t ow = 0; ow < out_w; ++ow) {
          double acc = 0.0;
          int64_t valid_count = 0;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - padding_h + kh;
            if (ih < 0 || ih >= input_h) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - padding_w + kw;
              if (iw < 0 || iw >= input_w) {
                continue;
              }
              acc += value_at_storage_unchecked(input, storage_offset_4d(input, n, c, ih, iw));
              ++valid_count;
            }
          }
          if (valid_count == 0) {
            throw ShapeError("avg_pool2d window did not overlap input");
          }
          const double denom = count_include_pad ? padded_window_count : static_cast<double>(valid_count);
          set_value_at_storage_unchecked(out, storage_offset_4d(out, n, c, oh, ow), acc / denom);
        }
      }
    }
  });
}

Tensor max_pool2d_input_grad(
    const Tensor& grad,
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  Tensor input_grad(input.shape(), grad.dtype(), false);
  fill_storage_unchecked(input_grad, 0.0);

  for (int64_t n = 0; n < input.shape()[0]; ++n) {
    for (int64_t c = 0; c < input.shape()[1]; ++c) {
      for (int64_t oh = 0; oh < grad.shape()[2]; ++oh) {
        for (int64_t ow = 0; ow < grad.shape()[3]; ++ow) {
          double max_value = -std::numeric_limits<double>::infinity();
          int64_t tie_count = 0;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
            if (ih < 0 || ih >= input.shape()[2]) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
              if (iw < 0 || iw >= input.shape()[3]) {
                continue;
              }
              const double value = value_at_storage_unchecked(input, storage_offset_4d(input, n, c, ih, iw));
              if (tie_count == 0 || value > max_value) {
                max_value = value;
                tie_count = 1;
              } else if (value == max_value) {
                ++tie_count;
              }
            }
          }
          if (tie_count == 0) {
            continue;
          }
          const double grad_share =
              value_at_storage_unchecked(grad, storage_offset_4d(grad, n, c, oh, ow)) /
              static_cast<double>(tie_count);
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - padding_h + kh * dilation_h;
            if (ih < 0 || ih >= input.shape()[2]) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - padding_w + kw * dilation_w;
              if (iw < 0 || iw >= input.shape()[3]) {
                continue;
              }
              const int64_t input_offset = storage_offset_4d(input, n, c, ih, iw);
              if (value_at_storage_unchecked(input, input_offset) != max_value) {
                continue;
              }
              const int64_t grad_offset = storage_offset_4d(input_grad, n, c, ih, iw);
              const double current = value_at_storage_unchecked(input_grad, grad_offset);
              set_value_at_storage_unchecked(input_grad, grad_offset, current + grad_share);
            }
          }
        }
      }
    }
  }
  return input_grad;
}

Tensor avg_pool2d_input_grad(
    const Tensor& grad,
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool count_include_pad) {
  Tensor input_grad(input.shape(), grad.dtype(), false);
  fill_storage_unchecked(input_grad, 0.0);
  const double padded_window_count = static_cast<double>(kernel_h * kernel_w);

  for (int64_t n = 0; n < input.shape()[0]; ++n) {
    for (int64_t c = 0; c < input.shape()[1]; ++c) {
      for (int64_t oh = 0; oh < grad.shape()[2]; ++oh) {
        for (int64_t ow = 0; ow < grad.shape()[3]; ++ow) {
          int64_t valid_count = 0;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - padding_h + kh;
            if (ih < 0 || ih >= input.shape()[2]) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - padding_w + kw;
              if (iw >= 0 && iw < input.shape()[3]) {
                ++valid_count;
              }
            }
          }
          if (valid_count == 0) {
            continue;
          }
          const double denom = count_include_pad ? padded_window_count : static_cast<double>(valid_count);
          const double grad_share =
              value_at_storage_unchecked(grad, storage_offset_4d(grad, n, c, oh, ow)) / denom;
          for (int64_t kh = 0; kh < kernel_h; ++kh) {
            const int64_t ih = oh * stride_h - padding_h + kh;
            if (ih < 0 || ih >= input.shape()[2]) {
              continue;
            }
            for (int64_t kw = 0; kw < kernel_w; ++kw) {
              const int64_t iw = ow * stride_w - padding_w + kw;
              if (iw < 0 || iw >= input.shape()[3]) {
                continue;
              }
              const int64_t grad_offset = storage_offset_4d(input_grad, n, c, ih, iw);
              const double current = value_at_storage_unchecked(input_grad, grad_offset);
              set_value_at_storage_unchecked(input_grad, grad_offset, current + grad_share);
            }
          }
        }
      }
    }
  }
  return input_grad;
}

Tensor max_pool2d_impl(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w,
    bool track_grad) {
  validate_pool2d_inputs(
      input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, "max_pool2d");
  Tensor out(pool2d_output_shape(
                 input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, "max_pool2d"),
             input.dtype(),
             false);
  max_pool2d_forward_kernel(out, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w);
  if (track_grad) {
    set_history(out, {input}, [input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w](
                              const Tensor& grad) {
      return std::vector<Tensor>{max_pool2d_input_grad(
          grad, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w)};
    });
  }
  return out;
}

Tensor avg_pool2d_impl(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool count_include_pad,
    bool track_grad) {
  validate_pool2d_inputs(
      input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, 1, 1, "avg_pool2d");
  Tensor out(pool2d_output_shape(input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, 1, 1, "avg_pool2d"),
             dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32,
             false);
  avg_pool2d_forward_kernel(out, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, count_include_pad);
  if (track_grad) {
    set_history(out, {input}, [input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, count_include_pad](
                              const Tensor& grad) {
      return std::vector<Tensor>{avg_pool2d_input_grad(
          grad, input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, count_include_pad)};
    });
  }
  return out;
}

int64_t normalize_axis(int64_t axis, const Shape& shape, const std::string& op_name) {
  const auto rank = static_cast<int64_t>(shape.size());
  if (rank == 0) {
    throw ShapeError(op_name + " axis is out of range for a scalar tensor");
  }
  int64_t normalized = axis < 0 ? axis + rank : axis;
  if (normalized < 0 || normalized >= rank) {
    throw ShapeError(
        op_name + " axis " + std::to_string(axis) + " is out of range for shape " +
        shape_to_string(shape));
  }
  return normalized;
}

Shape reduction_output_shape(const Shape& input_shape, int64_t axis, bool keepdims) {
  Shape output_shape;
  output_shape.reserve(input_shape.size());
  for (std::size_t i = 0; i < input_shape.size(); ++i) {
    if (static_cast<int64_t>(i) == axis) {
      if (keepdims) {
        output_shape.push_back(1);
      }
      continue;
    }
    output_shape.push_back(input_shape[i]);
  }
  return output_shape;
}

int64_t linear_from_coordinates(const Shape& coords, const Shape& shape) {
  if (shape.empty()) {
    return 0;
  }
  int64_t linear = 0;
  int64_t stride = 1;
  for (int64_t dim = static_cast<int64_t>(shape.size()) - 1; dim >= 0; --dim) {
    const auto index = static_cast<std::size_t>(dim);
    linear += coords[index] * stride;
    stride *= shape[index];
  }
  return linear;
}

struct AppliedIndex {
  TensorIndexKind kind{TensorIndexKind::Slice};
  int64_t input_dim{-1};
  int64_t index{0};
  int64_t start{0};
  int64_t length{0};
  int64_t step{1};
};

int64_t normalize_index_value(int64_t index_value, int64_t dim_size, int64_t dim) {
  int64_t normalized = index_value < 0 ? index_value + dim_size : index_value;
  if (normalized < 0 || normalized >= dim_size) {
    throw ShapeError(
        "index " + std::to_string(index_value) + " is out of range for axis " +
        std::to_string(dim) + " with size " + std::to_string(dim_size));
  }
  return normalized;
}

std::vector<AppliedIndex> normalize_indices(const Shape& input_shape, const std::vector<TensorIndex>& indices) {
  std::vector<AppliedIndex> result;
  result.reserve(indices.size() + input_shape.size());
  int64_t input_dim = 0;
  const int64_t input_rank = static_cast<int64_t>(input_shape.size());

  for (const TensorIndex& item : indices) {
    if (item.kind == TensorIndexKind::NewAxis) {
      result.push_back(AppliedIndex{TensorIndexKind::NewAxis, -1, 0, 0, 1, 1});
      continue;
    }
    if (input_dim >= input_rank) {
      throw ShapeError("too many indices for tensor with shape " + shape_to_string(input_shape));
    }

    const int64_t dim_size = input_shape[static_cast<std::size_t>(input_dim)];
    if (item.kind == TensorIndexKind::Index) {
      const int64_t normalized = normalize_index_value(item.index, dim_size, input_dim);
      result.push_back(AppliedIndex{TensorIndexKind::Index, input_dim, normalized, 0, 0, 1});
      ++input_dim;
      continue;
    }

    if (item.kind == TensorIndexKind::Slice) {
      if (item.length < 0) {
        throw ShapeError("slice length must be non-negative");
      }
      if (item.step == 0) {
        throw ShapeError("slice step cannot be zero");
      }
      if (item.length > 0) {
        const int64_t last_index = item.start + (item.length - 1) * item.step;
        if (item.start < 0 || item.start >= dim_size || last_index < 0 || last_index >= dim_size) {
          throw ShapeError(
              "slice is out of range for axis " + std::to_string(input_dim) +
              " with size " + std::to_string(dim_size));
        }
      }
      result.push_back(AppliedIndex{TensorIndexKind::Slice, input_dim, 0, item.start, item.length, item.step});
      ++input_dim;
      continue;
    }
  }

  while (input_dim < input_rank) {
    const int64_t dim_size = input_shape[static_cast<std::size_t>(input_dim)];
    result.push_back(AppliedIndex{TensorIndexKind::Slice, input_dim, 0, 0, dim_size, 1});
    ++input_dim;
  }

  return result;
}

Tensor index_backward(
    const Tensor& grad,
    const Shape& input_shape,
    const std::vector<AppliedIndex>& applied) {
  Tensor out(input_shape, grad.dtype(), false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    out.set_value_at_logical(i, 0.0);
  }

  for (int64_t linear = 0; linear < grad.numel(); ++linear) {
    const Shape grad_coords = coordinates_from_linear(linear, grad.shape());
    Shape input_coords(input_shape.size(), 0);
    std::size_t grad_dim = 0;

    for (const AppliedIndex& item : applied) {
      if (item.kind == TensorIndexKind::NewAxis) {
        ++grad_dim;
        continue;
      }
      const auto input_axis = static_cast<std::size_t>(item.input_dim);
      if (item.kind == TensorIndexKind::Index) {
        input_coords[input_axis] = item.index;
        continue;
      }
      input_coords[input_axis] = item.start + grad_coords[grad_dim] * item.step;
      ++grad_dim;
    }

    const int64_t target_linear = linear_from_coordinates(input_coords, input_shape);
    out.set_value_at_logical(target_linear, out.value_at_logical(target_linear) + grad.value_at_logical(linear));
  }
  return out;
}

Tensor index_impl(const Tensor& input, const std::vector<TensorIndex>& indices, bool track_grad) {
  const std::vector<AppliedIndex> applied = normalize_indices(input.shape(), indices);
  Shape output_shape;
  Shape output_strides;
  int64_t output_offset = input.offset();

  for (const AppliedIndex& item : applied) {
    if (item.kind == TensorIndexKind::NewAxis) {
      output_shape.push_back(1);
      output_strides.push_back(0);
      continue;
    }
    const auto input_axis = static_cast<std::size_t>(item.input_dim);
    if (item.kind == TensorIndexKind::Index) {
      output_offset += item.index * input.strides()[input_axis];
      continue;
    }
    output_offset += item.start * input.strides()[input_axis];
    output_shape.push_back(item.length);
    output_strides.push_back(input.strides()[input_axis] * item.step);
  }

  Tensor out(input.impl()->storage, input.dtype(), output_shape, output_strides, output_offset, false);
  if (track_grad) {
    set_history(out, {input}, [input_shape = input.shape(), applied](const Tensor& grad) {
      return std::vector<Tensor>{index_backward(grad, input_shape, applied)};
    });
  }
  return out;
}

int64_t reduction_output_linear(
    const Shape& input_coords,
    const Shape& input_shape,
    const Shape& output_shape,
    int64_t axis,
    bool keepdims) {
  if (output_shape.empty()) {
    return 0;
  }
  Shape output_coords;
  output_coords.reserve(output_shape.size());
  for (std::size_t i = 0; i < input_shape.size(); ++i) {
    if (static_cast<int64_t>(i) == axis) {
      if (keepdims) {
        output_coords.push_back(0);
      }
      continue;
    }
    output_coords.push_back(input_coords[i]);
  }
  return linear_from_coordinates(output_coords, output_shape);
}

Tensor broadcast_reduction_grad(
    const Tensor& grad,
    const Shape& input_shape,
    int64_t axis,
    bool keepdims,
    double scale) {
  Tensor out(input_shape, grad.dtype(), false);
  for (int64_t i = 0; i < out.numel(); ++i) {
    const Shape input_coords = coordinates_from_linear(i, input_shape);
    Shape grad_coords;
    grad_coords.reserve(grad.shape().size());
    for (std::size_t dim = 0; dim < input_shape.size(); ++dim) {
      if (static_cast<int64_t>(dim) == axis) {
        if (keepdims) {
          grad_coords.push_back(0);
        }
        continue;
      }
      grad_coords.push_back(input_coords[dim]);
    }
    const int64_t grad_linear = linear_from_coordinates(grad_coords, grad.shape());
    out.set_value_at_logical(i, grad.value_at_logical(grad_linear) * scale);
  }
  return out;
}

Tensor astype_impl(const Tensor& input, DType dtype, bool track_grad) {
  Tensor out(input.shape(), dtype, false);
  out.copy_from(input);
  if (track_grad && dtype_is_floating(dtype)) {
    set_history(out, {input}, [input_dtype = input.dtype()](const Tensor& grad) {
      return std::vector<Tensor>{astype_impl(grad, input_dtype, false)};
    });
  }
  return out;
}

int64_t normalize_insert_axis(int64_t axis, int64_t output_rank, const std::string& op_name) {
  int64_t normalized = axis < 0 ? axis + output_rank : axis;
  if (normalized < 0 || normalized >= output_rank) {
    throw ShapeError(op_name + " axis " + std::to_string(axis) + " is out of range");
  }
  return normalized;
}

void validate_concat_inputs(const std::vector<Tensor>& tensors, int64_t axis) {
  if (tensors.empty()) {
    throw ShapeError("concat requires at least one tensor");
  }
  const Shape& base_shape = tensors.front().shape();
  if (base_shape.empty()) {
    throw ShapeError("concat does not support scalar tensors; use stack instead");
  }
  for (std::size_t i = 1; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    if (tensor.dtype() != tensors.front().dtype()) {
      throw DTypeError(
          "concat requires all tensors to have the same dtype: tensor 0 has dtype " +
          dtype_name(tensors.front().dtype()) + ", tensor " + std::to_string(i) +
          " has dtype " + dtype_name(tensor.dtype()));
    }
    if (tensor.shape().size() != base_shape.size()) {
      throw ShapeError(
          "concat requires tensors with the same rank: tensor 0 has shape " +
          shape_to_string(base_shape) + ", tensor " + std::to_string(i) +
          " has shape " + shape_to_string(tensor.shape()));
    }
    for (std::size_t dim = 0; dim < base_shape.size(); ++dim) {
      if (static_cast<int64_t>(dim) == axis) {
        continue;
      }
      if (tensor.shape()[dim] != base_shape[dim]) {
        throw ShapeError(
            "concat shape mismatch: expected non-concat dimensions compatible with " +
            shape_to_string(base_shape) + ", got " + shape_to_string(tensor.shape()));
      }
    }
  }
}

Tensor concat_grad_slice(const Tensor& grad, const Shape& shape, int64_t axis, int64_t axis_offset) {
  Tensor out(shape, grad.dtype(), false);
  for (int64_t linear = 0; linear < out.numel(); ++linear) {
    Shape coords = coordinates_from_linear(linear, shape);
    coords[static_cast<std::size_t>(axis)] += axis_offset;
    const int64_t grad_linear = linear_from_coordinates(coords, grad.shape());
    out.set_value_at_logical(linear, grad.value_at_logical(grad_linear));
  }
  return out;
}

Tensor concat_impl(const std::vector<Tensor>& tensors, int64_t axis, bool track_grad) {
  if (tensors.empty()) {
    throw ShapeError("concat requires at least one tensor");
  }
  const int64_t normalized_axis = normalize_axis(axis, tensors.front().shape(), "concat");
  validate_concat_inputs(tensors, normalized_axis);

  Shape out_shape = tensors.front().shape();
  out_shape[static_cast<std::size_t>(normalized_axis)] = 0;
  for (const Tensor& tensor : tensors) {
    out_shape[static_cast<std::size_t>(normalized_axis)] +=
        tensor.shape()[static_cast<std::size_t>(normalized_axis)];
  }

  Tensor out(out_shape, tensors.front().dtype(), false);
  int64_t axis_offset = 0;
  for (const Tensor& tensor : tensors) {
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      Shape coords = coordinates_from_linear(linear, tensor.shape());
      coords[static_cast<std::size_t>(normalized_axis)] += axis_offset;
      const int64_t out_linear = linear_from_coordinates(coords, out_shape);
      out.set_value_at_logical(out_linear, tensor.value_at_logical(linear));
    }
    axis_offset += tensor.shape()[static_cast<std::size_t>(normalized_axis)];
  }

  if (track_grad) {
    std::vector<Shape> shapes;
    shapes.reserve(tensors.size());
    for (const Tensor& tensor : tensors) {
      shapes.push_back(tensor.shape());
    }
    set_history(out, tensors, [shapes, normalized_axis](const Tensor& grad) {
      std::vector<Tensor> grads;
      grads.reserve(shapes.size());
      int64_t offset = 0;
      for (const Shape& shape : shapes) {
        grads.push_back(concat_grad_slice(grad, shape, normalized_axis, offset));
        offset += shape[static_cast<std::size_t>(normalized_axis)];
      }
      return grads;
    });
  }
  return out;
}

void validate_stack_inputs(const std::vector<Tensor>& tensors) {
  if (tensors.empty()) {
    throw ShapeError("stack requires at least one tensor");
  }
  const Shape& base_shape = tensors.front().shape();
  for (std::size_t i = 1; i < tensors.size(); ++i) {
    const Tensor& tensor = tensors[i];
    if (tensor.dtype() != tensors.front().dtype()) {
      throw DTypeError(
          "stack requires all tensors to have the same dtype: tensor 0 has dtype " +
          dtype_name(tensors.front().dtype()) + ", tensor " + std::to_string(i) +
          " has dtype " + dtype_name(tensor.dtype()));
    }
    if (tensor.shape() != base_shape) {
      throw ShapeError(
          "stack requires tensors with identical shapes, got " + shape_to_string(base_shape) +
          " and " + shape_to_string(tensor.shape()));
    }
  }
}

Tensor stack_grad_slice(const Tensor& grad, const Shape& input_shape, int64_t axis, int64_t stack_index) {
  Tensor out(input_shape, grad.dtype(), false);
  for (int64_t linear = 0; linear < out.numel(); ++linear) {
    const Shape input_coords = coordinates_from_linear(linear, input_shape);
    Shape grad_coords;
    grad_coords.reserve(input_shape.size() + 1);
    std::size_t input_dim = 0;
    for (std::size_t dim = 0; dim < input_shape.size() + 1; ++dim) {
      if (static_cast<int64_t>(dim) == axis) {
        grad_coords.push_back(stack_index);
      } else {
        grad_coords.push_back(input_coords[input_dim++]);
      }
    }
    const int64_t grad_linear = linear_from_coordinates(grad_coords, grad.shape());
    out.set_value_at_logical(linear, grad.value_at_logical(grad_linear));
  }
  return out;
}

Tensor stack_impl(const std::vector<Tensor>& tensors, int64_t axis, bool track_grad) {
  validate_stack_inputs(tensors);
  const Shape& input_shape = tensors.front().shape();
  const int64_t output_rank = static_cast<int64_t>(input_shape.size()) + 1;
  const int64_t normalized_axis = normalize_insert_axis(axis, output_rank, "stack");

  Shape out_shape;
  out_shape.reserve(input_shape.size() + 1);
  std::size_t input_dim = 0;
  for (int64_t dim = 0; dim < output_rank; ++dim) {
    if (dim == normalized_axis) {
      out_shape.push_back(static_cast<int64_t>(tensors.size()));
    } else {
      out_shape.push_back(input_shape[input_dim++]);
    }
  }

  Tensor out(out_shape, tensors.front().dtype(), false);
  for (std::size_t tensor_index = 0; tensor_index < tensors.size(); ++tensor_index) {
    const Tensor& tensor = tensors[tensor_index];
    for (int64_t linear = 0; linear < tensor.numel(); ++linear) {
      const Shape input_coords = coordinates_from_linear(linear, input_shape);
      Shape out_coords;
      out_coords.reserve(out_shape.size());
      std::size_t current_input_dim = 0;
      for (int64_t dim = 0; dim < output_rank; ++dim) {
        if (dim == normalized_axis) {
          out_coords.push_back(static_cast<int64_t>(tensor_index));
        } else {
          out_coords.push_back(input_coords[current_input_dim++]);
        }
      }
      const int64_t out_linear = linear_from_coordinates(out_coords, out_shape);
      out.set_value_at_logical(out_linear, tensor.value_at_logical(linear));
    }
  }

  if (track_grad) {
    set_history(out, tensors, [input_shape, normalized_axis, count = tensors.size()](const Tensor& grad) {
      std::vector<Tensor> grads;
      grads.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        grads.push_back(stack_grad_slice(grad, input_shape, normalized_axis, static_cast<int64_t>(index)));
      }
      return grads;
    });
  }
  return out;
}

Tensor sum_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), "sum");
  const Shape out_shape = reduction_output_shape(input.shape(), normalized_axis, keepdims);
  const DType out_dtype = dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float64;
  Tensor out(out_shape, out_dtype, false);
  std::vector<double> values(static_cast<std::size_t>(out.numel()), 0.0);
  if (input.is_contiguous() && input.ndim() == 2) {
    sum_axis2d_contiguous_values(values, input, normalized_axis);
  } else {
    for (int64_t i = 0; i < input.numel(); ++i) {
      const Shape input_coords = coordinates_from_linear(i, input.shape());
      const int64_t out_linear =
          reduction_output_linear(input_coords, input.shape(), out_shape, normalized_axis, keepdims);
      values[static_cast<std::size_t>(out_linear)] += input.value_at_logical(i);
    }
  }
  write_contiguous_from_accumulator(out, values);
  if (track_grad) {
    set_history(out, {input}, [shape = input.shape(), normalized_axis, keepdims](const Tensor& grad) {
      return std::vector<Tensor>{broadcast_reduction_grad(grad, shape, normalized_axis, keepdims, 1.0)};
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

Tensor mean_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), "mean");
  const int64_t reduction_size = input.shape()[static_cast<std::size_t>(normalized_axis)];
  if (reduction_size == 0) {
    throw ShapeError("mean is undefined for empty reduction axes");
  }
  const Shape out_shape = reduction_output_shape(input.shape(), normalized_axis, keepdims);
  Tensor out(out_shape, DType::Float64, false);
  std::vector<double> values(static_cast<std::size_t>(out.numel()), 0.0);
  if (input.is_contiguous() && input.ndim() == 2) {
    sum_axis2d_contiguous_values(values, input, normalized_axis);
  } else {
    for (int64_t i = 0; i < input.numel(); ++i) {
      const Shape input_coords = coordinates_from_linear(i, input.shape());
      const int64_t out_linear =
          reduction_output_linear(input_coords, input.shape(), out_shape, normalized_axis, keepdims);
      values[static_cast<std::size_t>(out_linear)] += input.value_at_logical(i);
    }
  }
  for (double& value : values) {
    value /= static_cast<double>(reduction_size);
  }
  write_contiguous_from_accumulator(out, values);
  if (track_grad) {
    set_history(out, {input}, [shape = input.shape(), normalized_axis, keepdims, reduction_size](const Tensor& grad) {
      return std::vector<Tensor>{broadcast_reduction_grad(
          grad, shape, normalized_axis, keepdims, 1.0 / static_cast<double>(reduction_size))};
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

Tensor variance_impl(const Tensor& input, int64_t correction, bool track_grad) {
  if (correction < 0) {
    throw ShapeError("variance correction must be non-negative");
  }
  if (input.numel() - correction <= 0) {
    throw ShapeError("variance denominator must be positive");
  }
  Tensor mean_value = mean_impl(input, track_grad);
  Tensor centered = sub_impl(input, mean_value, track_grad);
  Tensor squared = mul_impl(centered, centered, track_grad);
  Tensor total = sum_impl(squared, track_grad);
  return div_impl(total, scalar_tensor(static_cast<double>(input.numel() - correction), total.dtype()), track_grad);
}

Tensor variance_axis_impl(const Tensor& input, int64_t axis, bool keepdims, int64_t correction, bool track_grad) {
  if (correction < 0) {
    throw ShapeError("variance correction must be non-negative");
  }
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), "variance");
  const int64_t reduction_size = input.shape()[static_cast<std::size_t>(normalized_axis)];
  if (reduction_size - correction <= 0) {
    throw ShapeError("variance denominator must be positive");
  }
  Tensor mean_value = mean_axis_impl(input, normalized_axis, true, track_grad);
  Tensor centered = sub_impl(input, mean_value, track_grad);
  Tensor squared = mul_impl(centered, centered, track_grad);
  Tensor total = sum_axis_impl(squared, normalized_axis, keepdims, track_grad);
  return div_impl(total, scalar_tensor(static_cast<double>(reduction_size - correction), total.dtype()), track_grad);
}

Tensor stddev_impl(const Tensor& input, int64_t correction, bool track_grad) {
  return sqrt_impl(variance_impl(input, correction, track_grad), track_grad);
}

Tensor stddev_axis_impl(const Tensor& input, int64_t axis, bool keepdims, int64_t correction, bool track_grad) {
  return sqrt_impl(variance_axis_impl(input, axis, keepdims, correction, track_grad), track_grad);
}

Tensor truth_reduction_impl(const Tensor& input, bool keepdims, bool want_all) {
  Shape out_shape;
  if (keepdims) {
    out_shape.assign(input.shape().size(), 1);
  }
  Tensor out(out_shape, DType::Bool, false);
  bool result = want_all;
  if (input.numel() == 0) {
    result = want_all;
  } else {
    for (int64_t i = 0; i < input.numel(); ++i) {
      const bool value = input.value_at_logical(i) != 0.0;
      if (want_all) {
        result = result && value;
      } else {
        result = result || value;
      }
    }
  }
  out.set_value_at_logical(0, result ? 1.0 : 0.0);
  return out;
}

Tensor truth_reduction_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool want_all, const std::string& op_name) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), op_name);
  const Shape out_shape = reduction_output_shape(input.shape(), normalized_axis, keepdims);
  Tensor out(out_shape, DType::Bool, false);
  std::vector<bool> values(static_cast<std::size_t>(out.numel()), want_all);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const Shape input_coords = coordinates_from_linear(i, input.shape());
    const int64_t out_linear =
        reduction_output_linear(input_coords, input.shape(), out_shape, normalized_axis, keepdims);
    const auto index = static_cast<std::size_t>(out_linear);
    const bool value = input.value_at_logical(i) != 0.0;
    values[index] = want_all ? (values[index] && value) : (values[index] || value);
  }
  for (int64_t i = 0; i < out.numel(); ++i) {
    out.set_value_at_logical(i, values[static_cast<std::size_t>(i)] ? 1.0 : 0.0);
  }
  return out;
}

Tensor all_impl(const Tensor& input, bool keepdims) {
  return truth_reduction_impl(input, keepdims, true);
}

Tensor all_axis_impl(const Tensor& input, int64_t axis, bool keepdims) {
  return truth_reduction_axis_impl(input, axis, keepdims, true, "all");
}

Tensor any_impl(const Tensor& input, bool keepdims) {
  return truth_reduction_impl(input, keepdims, false);
}

Tensor any_axis_impl(const Tensor& input, int64_t axis, bool keepdims) {
  return truth_reduction_axis_impl(input, axis, keepdims, false, "any");
}

Tensor logsumexp_impl(const Tensor& input, bool track_grad) {
  Tensor max_value = max_impl(input, false);
  Tensor shifted = sub_impl(input, max_value, track_grad);
  Tensor exp_values = exp_impl(shifted, track_grad);
  Tensor total = sum_impl(exp_values, track_grad);
  return add_impl(log_impl(total, track_grad), max_value, track_grad);
}

Tensor logsumexp_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), "logsumexp");
  Tensor max_values = max_axis_impl(input, normalized_axis, true, false);
  Tensor shifted = sub_impl(input, max_values, track_grad);
  Tensor exp_values = exp_impl(shifted, track_grad);
  Tensor totals = sum_axis_impl(exp_values, normalized_axis, true, track_grad);
  Tensor result = add_impl(log_impl(totals, track_grad), max_values, track_grad);
  if (keepdims) {
    return result;
  }
  return squeeze_impl(result, normalized_axis, track_grad);
}

Tensor softmax_impl(const Tensor& input, int64_t axis, bool track_grad) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), "softmax");
  Tensor max_values = max_axis_impl(input, normalized_axis, true, false);
  Tensor shifted = sub_impl(input, max_values, track_grad);
  Tensor exp_values = exp_impl(shifted, track_grad);
  Tensor totals = sum_axis_impl(exp_values, normalized_axis, true, track_grad);
  return div_impl(exp_values, totals, track_grad);
}

Tensor log_softmax_impl(const Tensor& input, int64_t axis, bool track_grad) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), "log_softmax");
  Tensor max_values = max_axis_impl(input, normalized_axis, true, false);
  Tensor shifted = sub_impl(input, max_values, track_grad);
  Tensor exp_values = exp_impl(shifted, track_grad);
  Tensor totals = sum_axis_impl(exp_values, normalized_axis, true, track_grad);
  return sub_impl(shifted, log_impl(totals, track_grad), track_grad);
}

Tensor extreme_axis_grad(
    const Tensor& input,
    const Tensor& grad,
    int64_t axis,
    bool keepdims,
    bool is_max) {
  const Tensor extremes =
      is_max ? max_axis_impl(input, axis, keepdims, false) : min_axis_impl(input, axis, keepdims, false);
  std::vector<int64_t> counts(static_cast<std::size_t>(extremes.numel()), 0);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const Shape input_coords = coordinates_from_linear(i, input.shape());
    const int64_t out_linear = reduction_output_linear(input_coords, input.shape(), extremes.shape(), axis, keepdims);
    if (input.value_at_logical(i) == extremes.value_at_logical(out_linear)) {
      ++counts[static_cast<std::size_t>(out_linear)];
    }
  }

  Tensor out(input.shape(), grad.dtype(), false);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const Shape input_coords = coordinates_from_linear(i, input.shape());
    const int64_t out_linear = reduction_output_linear(input_coords, input.shape(), extremes.shape(), axis, keepdims);
    double value = 0.0;
    const auto count = counts[static_cast<std::size_t>(out_linear)];
    if (count > 0 && input.value_at_logical(i) == extremes.value_at_logical(out_linear)) {
      value = grad.value_at_logical(out_linear) / static_cast<double>(count);
    }
    out.set_value_at_logical(i, value);
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

Tensor extreme_axis_impl(
    const Tensor& input,
    int64_t axis,
    bool keepdims,
    bool track_grad,
    bool is_max,
    const std::string& op_name) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), op_name);
  if (input.shape()[static_cast<std::size_t>(normalized_axis)] == 0) {
    throw ShapeError(op_name + " is undefined for empty reduction axes");
  }
  const Shape out_shape = reduction_output_shape(input.shape(), normalized_axis, keepdims);
  Tensor out(out_shape, input.dtype(), false);
  std::vector<double> values(static_cast<std::size_t>(out.numel()), 0.0);
  std::vector<bool> initialized(static_cast<std::size_t>(out.numel()), false);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const Shape input_coords = coordinates_from_linear(i, input.shape());
    const int64_t out_linear =
        reduction_output_linear(input_coords, input.shape(), out_shape, normalized_axis, keepdims);
    const auto index = static_cast<std::size_t>(out_linear);
    const double value = input.value_at_logical(i);
    if (!initialized[index] || (is_max ? value > values[index] : value < values[index])) {
      values[index] = value;
      initialized[index] = true;
    }
  }
  write_contiguous_from_accumulator(out, values);
  if (track_grad) {
    set_history(out, {input}, [input, normalized_axis, keepdims, is_max](const Tensor& grad) {
      return std::vector<Tensor>{extreme_axis_grad(input, grad, normalized_axis, keepdims, is_max)};
    });
  }
  return out;
}

Tensor max_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad) {
  return extreme_axis_impl(input, axis, keepdims, track_grad, true, "max");
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

Tensor min_axis_impl(const Tensor& input, int64_t axis, bool keepdims, bool track_grad) {
  return extreme_axis_impl(input, axis, keepdims, track_grad, false, "min");
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

Tensor arg_extreme_impl(const Tensor& input, bool keepdims, bool is_max, const std::string& op_name) {
  if (input.numel() == 0) {
    throw ShapeError(op_name + " is undefined for empty tensors");
  }
  Shape out_shape;
  if (keepdims) {
    out_shape.assign(input.shape().size(), 1);
  }
  Tensor out(out_shape, DType::Int64, false);
  int64_t best_index = 0;
  double best_value = input.value_at_logical(0);
  for (int64_t i = 1; i < input.numel(); ++i) {
    const double value = input.value_at_logical(i);
    if (is_max ? value > best_value : value < best_value) {
      best_value = value;
      best_index = i;
    }
  }
  out.set_value_at_logical(0, static_cast<double>(best_index));
  return out;
}

Tensor arg_extreme_axis_impl(
    const Tensor& input,
    int64_t axis,
    bool keepdims,
    bool is_max,
    const std::string& op_name) {
  const int64_t normalized_axis = normalize_axis(axis, input.shape(), op_name);
  if (input.shape()[static_cast<std::size_t>(normalized_axis)] == 0) {
    throw ShapeError(op_name + " is undefined for empty reduction axes");
  }
  const Shape out_shape = reduction_output_shape(input.shape(), normalized_axis, keepdims);
  Tensor out(out_shape, DType::Int64, false);
  std::vector<int64_t> best_indices(static_cast<std::size_t>(out.numel()), -1);
  std::vector<double> best_values(static_cast<std::size_t>(out.numel()), 0.0);
  for (int64_t i = 0; i < input.numel(); ++i) {
    const Shape input_coords = coordinates_from_linear(i, input.shape());
    const int64_t out_linear =
        reduction_output_linear(input_coords, input.shape(), out_shape, normalized_axis, keepdims);
    const auto out_index = static_cast<std::size_t>(out_linear);
    const double value = input.value_at_logical(i);
    if (best_indices[out_index] == -1 || (is_max ? value > best_values[out_index] : value < best_values[out_index])) {
      best_values[out_index] = value;
      best_indices[out_index] = input_coords[static_cast<std::size_t>(normalized_axis)];
    }
  }
  for (int64_t i = 0; i < out.numel(); ++i) {
    out.set_value_at_logical(i, static_cast<double>(best_indices[static_cast<std::size_t>(i)]));
  }
  return out;
}

Tensor relu_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, input.dtype(), ReluOp{});
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
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, SigmoidOp{});
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
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, TanhOp{});
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
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, ExpOp{});
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      return std::vector<Tensor>{mul_impl(grad, exp_impl(input, false), false)};
    });
  }
  return out;
}

Tensor log_impl(const Tensor& input, bool track_grad) {
  Tensor out = elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, LogOp{});
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      return std::vector<Tensor>{div_impl(grad, input, false)};
    });
  }
  return out;
}

Tensor log1p_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::log1p(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor denom = add_impl(input, scalar_tensor(1.0, input.dtype()), false);
      return std::vector<Tensor>{div_impl(grad, denom, false)};
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

Tensor rsqrt_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return 1.0 / std::sqrt(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor coeff = scalar_tensor(-0.5, grad.dtype());
      Tensor base = pow_impl(input, -1.5, false);
      return std::vector<Tensor>{mul_impl(mul_impl(grad, coeff, false), base, false)};
    });
  }
  return out;
}

Tensor sin_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::sin(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      return std::vector<Tensor>{mul_impl(grad, cos_impl(input, false), false)};
    });
  }
  return out;
}

Tensor cos_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::cos(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      return std::vector<Tensor>{neg_impl(mul_impl(grad, sin_impl(input, false), false), false)};
    });
  }
  return out;
}

Tensor tan_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::tan(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor y = tan_impl(input, false);
      Tensor sec_sq = add_impl(full_like(y.shape(), 1.0, y.dtype()), mul_impl(y, y, false), false);
      return std::vector<Tensor>{mul_impl(grad, sec_sq, false)};
    });
  }
  return out;
}

Tensor asin_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::asin(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor one_minus_square =
          sub_impl(full_like(input.shape(), 1.0, input.dtype()), mul_impl(input, input, false), false);
      Tensor denom = sqrt_impl(one_minus_square, false);
      return std::vector<Tensor>{div_impl(grad, denom, false)};
    });
  }
  return out;
}

Tensor acos_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::acos(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor one_minus_square =
          sub_impl(full_like(input.shape(), 1.0, input.dtype()), mul_impl(input, input, false), false);
      Tensor denom = sqrt_impl(one_minus_square, false);
      return std::vector<Tensor>{neg_impl(div_impl(grad, denom, false), false)};
    });
  }
  return out;
}

Tensor atan_impl(const Tensor& input, bool track_grad) {
  Tensor out =
      elementwise_unary_impl(input, dtype_is_floating(input.dtype()) ? input.dtype() : DType::Float32, [](double value) {
        return std::atan(value);
      });
  if (track_grad) {
    set_history(out, {input}, [input](const Tensor& grad) {
      Tensor one_plus_square =
          add_impl(full_like(input.shape(), 1.0, input.dtype()), mul_impl(input, input, false), false);
      return std::vector<Tensor>{div_impl(grad, one_plus_square, false)};
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
      Tensor usable_grad = grad.is_contiguous() ? grad : grad.clone();
      return std::vector<Tensor>{reshape_impl(usable_grad, original_shape, false)};
    });
  }
  return out;
}

int64_t normalize_unsqueeze_axis(int64_t axis, int64_t rank, const std::string& op_name) {
  const int64_t output_rank = rank + 1;
  const int64_t normalized = axis < 0 ? axis + output_rank : axis;
  if (normalized < 0 || normalized > rank) {
    throw ShapeError(
        op_name + " axis " + std::to_string(axis) + " is out of range for rank " +
        std::to_string(rank));
  }
  return normalized;
}

Shape normalize_permutation(const Shape& axes, int64_t rank, const std::string& op_name) {
  if (static_cast<int64_t>(axes.size()) != rank) {
    throw ShapeError(
        op_name + " expected " + std::to_string(rank) + " axes, got " +
        shape_to_string(axes));
  }
  Shape normalized;
  normalized.reserve(axes.size());
  std::vector<bool> seen(static_cast<std::size_t>(rank), false);
  for (const int64_t axis : axes) {
    const int64_t value = axis < 0 ? axis + rank : axis;
    if (value < 0 || value >= rank) {
      throw ShapeError(op_name + " axis " + std::to_string(axis) + " is out of range");
    }
    const auto index = static_cast<std::size_t>(value);
    if (seen[index]) {
      throw ShapeError(op_name + " axes must be unique, got " + shape_to_string(axes));
    }
    seen[index] = true;
    normalized.push_back(value);
  }
  return normalized;
}

Shape inverse_permutation(const Shape& axes) {
  Shape inverse(axes.size(), 0);
  for (std::size_t i = 0; i < axes.size(); ++i) {
    inverse[static_cast<std::size_t>(axes[i])] = static_cast<int64_t>(i);
  }
  return inverse;
}

Tensor permute_impl(const Tensor& input, const Shape& axes, bool track_grad) {
  const Shape normalized_axes = normalize_permutation(axes, input.ndim(), "permute");
  Shape output_shape;
  Shape output_strides;
  output_shape.reserve(normalized_axes.size());
  output_strides.reserve(normalized_axes.size());
  for (const int64_t axis : normalized_axes) {
    const auto index = static_cast<std::size_t>(axis);
    output_shape.push_back(input.shape()[index]);
    output_strides.push_back(input.strides()[index]);
  }
  Tensor out(
      input.impl()->storage,
      input.dtype(),
      output_shape,
      output_strides,
      input.offset(),
      false);
  if (track_grad) {
    set_history(out, {input}, [inverse = inverse_permutation(normalized_axes)](const Tensor& grad) {
      return std::vector<Tensor>{permute_impl(grad, inverse, false)};
    });
  }
  return out;
}

Tensor transpose_impl(const Tensor& input, bool track_grad) {
  Shape axes(static_cast<std::size_t>(input.ndim()), 0);
  std::iota(axes.begin(), axes.end(), int64_t{0});
  std::reverse(axes.begin(), axes.end());
  return permute_impl(input, axes, track_grad);
}

Tensor transpose_impl(const Tensor& input, int64_t axis0, int64_t axis1, bool track_grad) {
  Shape axes(static_cast<std::size_t>(input.ndim()), 0);
  std::iota(axes.begin(), axes.end(), int64_t{0});
  const int64_t dim0 = normalize_axis(axis0, input.shape(), "transpose");
  const int64_t dim1 = normalize_axis(axis1, input.shape(), "transpose");
  std::swap(axes[static_cast<std::size_t>(dim0)], axes[static_cast<std::size_t>(dim1)]);
  return permute_impl(input, axes, track_grad);
}

Tensor unsqueeze_impl(const Tensor& input, int64_t axis, bool track_grad) {
  const int64_t normalized_axis = normalize_unsqueeze_axis(axis, input.ndim(), "unsqueeze");
  Shape output_shape = input.shape();
  Shape output_strides = input.strides();
  int64_t inserted_stride = 1;
  if (!input.shape().empty() && normalized_axis < input.ndim()) {
    const auto index = static_cast<std::size_t>(normalized_axis);
    inserted_stride = input.strides()[index] * std::max<int64_t>(input.shape()[index], 1);
  }
  output_shape.insert(output_shape.begin() + normalized_axis, 1);
  output_strides.insert(output_strides.begin() + normalized_axis, inserted_stride);
  Tensor out(input.impl()->storage, input.dtype(), output_shape, output_strides, input.offset(), false);
  if (track_grad) {
    set_history(out, {input}, [normalized_axis](const Tensor& grad) {
      return std::vector<Tensor>{squeeze_impl(grad, normalized_axis, false)};
    });
  }
  return out;
}

Tensor squeeze_impl(const Tensor& input, std::optional<int64_t> axis, bool track_grad) {
  Shape output_shape;
  Shape output_strides;
  std::vector<int64_t> removed_axes;

  std::optional<int64_t> normalized_axis;
  if (axis.has_value()) {
    normalized_axis = normalize_axis(*axis, input.shape(), "squeeze");
    if (input.shape()[static_cast<std::size_t>(*normalized_axis)] != 1) {
      throw ShapeError(
          "cannot squeeze axis " + std::to_string(*axis) + " with size " +
          std::to_string(input.shape()[static_cast<std::size_t>(*normalized_axis)]) +
          " for shape " + shape_to_string(input.shape()));
    }
  }

  for (int64_t dim = 0; dim < input.ndim(); ++dim) {
    const bool remove = axis.has_value()
        ? dim == *normalized_axis
        : input.shape()[static_cast<std::size_t>(dim)] == 1;
    if (remove) {
      removed_axes.push_back(dim);
      continue;
    }
    output_shape.push_back(input.shape()[static_cast<std::size_t>(dim)]);
    output_strides.push_back(input.strides()[static_cast<std::size_t>(dim)]);
  }

  Tensor out(input.impl()->storage, input.dtype(), output_shape, output_strides, input.offset(), false);
  if (track_grad) {
    set_history(out, {input}, [removed_axes](const Tensor& grad) {
      Tensor result = grad;
      for (const int64_t removed_axis : removed_axes) {
        result = unsqueeze_impl(result, removed_axis, false);
      }
      return std::vector<Tensor>{result};
    });
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

Tensor maximum(const Tensor& left, const Tensor& right) {
  return maximum_impl(left, right, true);
}

Tensor minimum(const Tensor& left, const Tensor& right) {
  return minimum_impl(left, right, true);
}

Tensor where(const Tensor& condition, const Tensor& true_value, const Tensor& false_value) {
  return where_impl(condition, true_value, false_value, true);
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

Tensor bmm(const Tensor& left, const Tensor& right) {
  return bmm_impl(left, right, true);
}

Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  return conv2d_impl(
      input, weight, bias, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, true);
}

Tensor max_pool2d(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    int64_t dilation_h,
    int64_t dilation_w) {
  return max_pool2d_impl(
      input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, dilation_h, dilation_w, true);
}

Tensor avg_pool2d(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h,
    int64_t stride_w,
    int64_t padding_h,
    int64_t padding_w,
    bool count_include_pad) {
  return avg_pool2d_impl(
      input, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, count_include_pad, true);
}

Tensor sum(const Tensor& input) {
  return sum_impl(input, true);
}

Tensor sum(const Tensor& input, int64_t axis, bool keepdims) {
  return sum_axis_impl(input, axis, keepdims, true);
}

Tensor mean(const Tensor& input) {
  return mean_impl(input, true);
}

Tensor mean(const Tensor& input, int64_t axis, bool keepdims) {
  return mean_axis_impl(input, axis, keepdims, true);
}

Tensor variance(const Tensor& input, int64_t correction) {
  return variance_impl(input, correction, true);
}

Tensor variance(const Tensor& input, int64_t axis, bool keepdims, int64_t correction) {
  return variance_axis_impl(input, axis, keepdims, correction, true);
}

Tensor stddev(const Tensor& input, int64_t correction) {
  return stddev_impl(input, correction, true);
}

Tensor stddev(const Tensor& input, int64_t axis, bool keepdims, int64_t correction) {
  return stddev_axis_impl(input, axis, keepdims, correction, true);
}

Tensor max(const Tensor& input) {
  return max_impl(input, true);
}

Tensor max(const Tensor& input, int64_t axis, bool keepdims) {
  return max_axis_impl(input, axis, keepdims, true);
}

Tensor min(const Tensor& input) {
  return min_impl(input, true);
}

Tensor min(const Tensor& input, int64_t axis, bool keepdims) {
  return min_axis_impl(input, axis, keepdims, true);
}

Tensor argmax(const Tensor& input, bool keepdims) {
  return arg_extreme_impl(input, keepdims, true, "argmax");
}

Tensor argmax(const Tensor& input, int64_t axis, bool keepdims) {
  return arg_extreme_axis_impl(input, axis, keepdims, true, "argmax");
}

Tensor argmin(const Tensor& input, bool keepdims) {
  return arg_extreme_impl(input, keepdims, false, "argmin");
}

Tensor argmin(const Tensor& input, int64_t axis, bool keepdims) {
  return arg_extreme_axis_impl(input, axis, keepdims, false, "argmin");
}

Tensor all(const Tensor& input, bool keepdims) {
  return all_impl(input, keepdims);
}

Tensor all(const Tensor& input, int64_t axis, bool keepdims) {
  return all_axis_impl(input, axis, keepdims);
}

Tensor any(const Tensor& input, bool keepdims) {
  return any_impl(input, keepdims);
}

Tensor any(const Tensor& input, int64_t axis, bool keepdims) {
  return any_axis_impl(input, axis, keepdims);
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

Tensor logsumexp(const Tensor& input) {
  return logsumexp_impl(input, true);
}

Tensor logsumexp(const Tensor& input, int64_t axis, bool keepdims) {
  return logsumexp_axis_impl(input, axis, keepdims, true);
}

Tensor softmax(const Tensor& input, int64_t axis) {
  return softmax_impl(input, axis, true);
}

Tensor log_softmax(const Tensor& input, int64_t axis) {
  return log_softmax_impl(input, axis, true);
}

Tensor log1p(const Tensor& input) {
  return log1p_impl(input, true);
}

Tensor sqrt(const Tensor& input) {
  return sqrt_impl(input, true);
}

Tensor rsqrt(const Tensor& input) {
  return rsqrt_impl(input, true);
}

Tensor sin(const Tensor& input) {
  return sin_impl(input, true);
}

Tensor cos(const Tensor& input) {
  return cos_impl(input, true);
}

Tensor tan(const Tensor& input) {
  return tan_impl(input, true);
}

Tensor asin(const Tensor& input) {
  return asin_impl(input, true);
}

Tensor acos(const Tensor& input) {
  return acos_impl(input, true);
}

Tensor atan(const Tensor& input) {
  return atan_impl(input, true);
}

Tensor abs(const Tensor& input) {
  return abs_impl(input, true);
}

Tensor clamp(const Tensor& input, double min_value, double max_value) {
  return clamp_impl(input, min_value, max_value, true);
}

Tensor astype(const Tensor& input, DType dtype) {
  return astype_impl(input, dtype, true);
}

Tensor concat(const std::vector<Tensor>& tensors, int64_t axis) {
  return concat_impl(tensors, axis, true);
}

Tensor stack(const std::vector<Tensor>& tensors, int64_t axis) {
  return stack_impl(tensors, axis, true);
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

Tensor transpose(const Tensor& input, int64_t axis0, int64_t axis1) {
  return transpose_impl(input, axis0, axis1, true);
}

Tensor permute(const Tensor& input, const Shape& axes) {
  return permute_impl(input, axes, true);
}

Tensor squeeze(const Tensor& input, std::optional<int64_t> axis) {
  return squeeze_impl(input, axis, true);
}

Tensor unsqueeze(const Tensor& input, int64_t axis) {
  return unsqueeze_impl(input, axis, true);
}

Tensor index(const Tensor& input, const std::vector<TensorIndex>& indices) {
  return index_impl(input, indices, true);
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
