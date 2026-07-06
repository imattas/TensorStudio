#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

Tensor zeros(const Shape& shape, DType dtype = DType::Float32);
Tensor ones(const Shape& shape, DType dtype = DType::Float32);
Tensor full(const Shape& shape, double fill_value, DType dtype = DType::Float32);
Tensor arange(double start, double stop, double step = 1.0, DType dtype = DType::Float32);

Tensor add(const Tensor& left, const Tensor& right);
Tensor sub(const Tensor& left, const Tensor& right);
Tensor mul(const Tensor& left, const Tensor& right);
Tensor div(const Tensor& left, const Tensor& right);
Tensor neg(const Tensor& input);
Tensor pow(const Tensor& input, double exponent);
Tensor matmul(const Tensor& left, const Tensor& right);
Tensor sum(const Tensor& input);
Tensor mean(const Tensor& input);
Tensor relu(const Tensor& input);
Tensor sigmoid(const Tensor& input);
Tensor tanh(const Tensor& input);
Tensor exp(const Tensor& input);
Tensor log(const Tensor& input);
Tensor reshape(const Tensor& input, const Shape& shape);
Tensor flatten(const Tensor& input);
Tensor transpose(const Tensor& input);

Tensor scalar_tensor(double value, DType dtype = DType::Float32);
Tensor unbroadcast(const Tensor& grad, const Shape& target_shape);

}  // namespace tensorstudio
