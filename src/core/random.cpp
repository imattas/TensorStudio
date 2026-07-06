#include "tensorstudio/random.hpp"

#include <random>

namespace tensorstudio {

Tensor randn(const Shape& shape, DType dtype, std::optional<uint64_t> seed) {
  std::mt19937_64 generator(seed.value_or(std::random_device{}()));
  std::normal_distribution<double> distribution(0.0, 1.0);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, distribution(generator));
  }
  return result;
}

}  // namespace tensorstudio
