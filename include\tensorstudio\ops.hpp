#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

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
Tensor neg(const Tensor& input);
Tensor pow(const Tensor& input, double exponent);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor sum(const Tensor& input);
Tensor mean(const Tensor& input);
Tensor max(const Tensor& input);
Tensor min(const Tensor& input);
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor log(const Tensor& input);
Tensor sqrt(const Tensor& input);
Tensor abs(const Tensor& input);
Tensor clamp(const Tensor& input, double min_value, double max_value);
Tensor reshape(const Tensor& input, const Shape& shape);
Tensor flatten(const Tensor& input);
Tensor transpose(const Tensor& input);

Tensor scalar_tensor(double value, DType dtype = DType::Float32);
Tensor unbroadcast(const Tensor& grad, const Shape& target_shape);

}  // namespace tensorstudio
