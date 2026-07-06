#pragma once

#include <optional>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

Tensor randn(const Shape& shape, DType dtype = DType::Float32, std::optional<uint64_t> seed = std::nullopt);

}  // namespace tensorstudio
