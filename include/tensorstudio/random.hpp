#pragma once

#include <optional>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

void manual_seed(uint64_t seed);
Tensor rand(const Shape& shape, DType dtype = DType::Float32, std::optional<uint64_t> seed = std::nullopt);
Tensor randn(const Shape& shape, DType dtype = DType::Float32, std::optional<uint64_t> seed = std::nullopt);
Tensor uniform(
    const Shape& shape,
    double low = 0.0,
    double high = 1.0,
    DType dtype = DType::Float32,
    std::optional<uint64_t> seed = std::nullopt);
Tensor normal(
    const Shape& shape,
    double mean = 0.0,
    double stddev = 1.0,
    DType dtype = DType::Float32,
    std::optional<uint64_t> seed = std::nullopt);
Tensor randint(
    const Shape& shape,
    int64_t low,
    int64_t high,
    DType dtype = DType::Int64,
    std::optional<uint64_t> seed = std::nullopt);
Tensor bernoulli(
    const Shape& shape,
    double probability = 0.5,
    DType dtype = DType::Bool,
    std::optional<uint64_t> seed = std::nullopt);

}  // namespace tensorstudio
