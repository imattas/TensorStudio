#pragma once

#include <optional>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

void manual_seed(uint64_t seed);
Tensor rand(const Shape& shape, DType dtype = DType::Float32, std::optional<uint64_t> seed = std::nullopt);
Tensor randn(const Shape& shape, DType dtype = DType::Float32, std::optional<uint64_t> seed = std::nullopt);

}  // namespace tensorstudio
