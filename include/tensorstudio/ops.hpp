#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

enum class TensorIndexKind {
  Index,
  Slice,
  NewAxis,
};

struct TensorIndex {
  TensorIndexKind kind{TensorIndexKind::Slice};
  int64_t index{0};
  int64_t start{0};
  int64_t length{0};
  int64_t step{1};

  static TensorIndex at(int64_t index);
  static TensorIndex slice(int64_t start, int64_t length, int64_t step = 1);
  static TensorIndex new_axis();
};

Tensor zeros(const Shape& shape, DType dtype = DType::Float32);
Tensor ones(const Shape& shape, DType dtype = DType::Float32);
Tensor full(const Shape& shape, double fill_value, DType dtype = DType::Float32);
Tensor empty(const Shape& shape, DType dtype = DType::Float32);
Tensor arange(double start, double stop, double step = 1.0, DType dtype = DType::Float32);
Tensor eye(int64_t n, int64_t m = -1, DType dtype = DType::Float32);
Tensor linspace(double start, double stop, int64_t steps, DType dtype = DType::Float32);

Tensor add(const Tensor& left, const Tensor& right);
Tensor sub(const Tensor& left, const Tensor& right);
Tensor mul(const Tensor& left, const Tensor& right);
Tensor div(const Tensor& left, const Tensor& right);
Tensor equal(const Tensor& left, const Tensor& right);
Tensor not_equal(const Tensor& left, const Tensor& right);
Tensor less(const Tensor& left, const Tensor& right);
Tensor less_equal(const Tensor& left, const Tensor& right);
Tensor greater(const Tensor& left, const Tensor& right);
Tensor greater_equal(const Tensor& left, const Tensor& right);
Tensor maximum(const Tensor& left, const Tensor& right);
Tensor minimum(const Tensor& left, const Tensor& right);
Tensor where(const Tensor& condition, const Tensor& true_value, const Tensor& false_value);
Tensor neg(const Tensor& input);
Tensor pow(const Tensor& input, double exponent);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor bmm(const Tensor& left, const Tensor& right);
Tensor conv2d(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias = std::nullopt,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    int64_t dilation_h = 1,
    int64_t dilation_w = 1);
Tensor max_pool2d(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    int64_t dilation_h = 1,
    int64_t dilation_w = 1);
Tensor avg_pool2d(
    const Tensor& input,
    int64_t kernel_h,
    int64_t kernel_w,
    int64_t stride_h = 1,
    int64_t stride_w = 1,
    int64_t padding_h = 0,
    int64_t padding_w = 0,
    bool count_include_pad = false);
Tensor sum(const Tensor& input);
Tensor sum(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor mean(const Tensor& input);
Tensor mean(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor variance(const Tensor& input, int64_t correction = 0);
Tensor variance(const Tensor& input, int64_t axis, bool keepdims, int64_t correction = 0);
Tensor stddev(const Tensor& input, int64_t correction = 0);
Tensor stddev(const Tensor& input, int64_t axis, bool keepdims, int64_t correction = 0);
Tensor max(const Tensor& input);
Tensor max(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor min(const Tensor& input);
Tensor min(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor argmax(const Tensor& input, bool keepdims = false);
Tensor argmax(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor argmin(const Tensor& input, bool keepdims = false);
Tensor argmin(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor all(const Tensor& input, bool keepdims = false);
Tensor all(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor any(const Tensor& input, bool keepdims = false);
Tensor any(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor log(const Tensor& input);
Tensor logsumexp(const Tensor& input);
Tensor logsumexp(const Tensor& input, int64_t axis, bool keepdims = false);
Tensor softmax(const Tensor& input, int64_t axis = -1);
Tensor log_softmax(const Tensor& input, int64_t axis = -1);
Tensor log1p(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor rsqrt(const Tensor& input);
Tensor sin(const Tensor& input);
Tensor cos(const Tensor& input);
Tensor tan(const Tensor& input);
Tensor asin(const Tensor& input);
Tensor acos(const Tensor& input);
Tensor atan(const Tensor& input);
Tensor abs(const Tensor& input);
Tensor clamp(const Tensor& input, double min_value, double max_value);
Tensor astype(const Tensor& input, DType dtype);
Tensor concat(const std::vector<Tensor>& tensors, int64_t axis = 0);
Tensor stack(const std::vector<Tensor>& tensors, int64_t axis = 0);
Tensor reshape(const Tensor& input, const Shape& shape);
Tensor flatten(const Tensor& input);
Tensor transpose(const Tensor& input);
Tensor transpose(const Tensor& input, int64_t axis0, int64_t axis1);
Tensor permute(const Tensor& input, const Shape& axes);
Tensor squeeze(const Tensor& input, std::optional<int64_t> axis = std::nullopt);
Tensor unsqueeze(const Tensor& input, int64_t axis);
Tensor index(const Tensor& input, const std::vector<TensorIndex>& indices);

Tensor scalar_tensor(double value, DType dtype = DType::Float32);
Tensor unbroadcast(const Tensor& grad, const Shape& target_shape);

}  // namespace tensorstudio
