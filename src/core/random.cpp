#include "tensorstudio/random.hpp"

#include <random>

namespace tensorstudio {
namespace {

std::mt19937_64& global_generator() {
  static std::mt19937_64 generator(0);
  return generator;
}

std::mt19937_64 make_generator(std::optional<uint64_t> seed) {
  if (seed.has_value()) {
    return std::mt19937_64(*seed);
  }
  return global_generator();
}

}  // namespace

void manual_seed(uint64_t seed) {
  global_generator().seed(seed);
}

Tensor rand(const Shape& shape, DType dtype, std::optional<uint64_t> seed) {
  std::mt19937_64 generator = make_generator(seed);
  std::uniform_real_distribution<double> distribution(0.0, 1.0);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, distribution(generator));
  }
  if (!seed.has_value()) {
    global_generator() = generator;
  }
  return result;
}

Tensor randn(const Shape& shape, DType dtype, std::optional<uint64_t> seed) {
  std::mt19937_64 generator = make_generator(seed);
  std::normal_distribution<double> distribution(0.0, 1.0);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, distribution(generator));
  }
  if (!seed.has_value()) {
    global_generator() = generator;
  }
  return result;
}

}  // namespace tensorstudio
