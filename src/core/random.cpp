#include "tensorstudio/random.hpp"

#include <random>

#include "tensorstudio/errors.hpp"

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
  return uniform(shape, 0.0, 1.0, dtype, seed);
}

Tensor randn(const Shape& shape, DType dtype, std::optional<uint64_t> seed) {
  return normal(shape, 0.0, 1.0, dtype, seed);
}

Tensor uniform(const Shape& shape, double low, double high, DType dtype, std::optional<uint64_t> seed) {
  if (high <= low) {
    throw TensorStudioError("uniform requires high > low");
  }
  std::mt19937_64 generator = make_generator(seed);
  std::uniform_real_distribution<double> distribution(low, high);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, distribution(generator));
  }
  if (!seed.has_value()) {
    global_generator() = generator;
  }
  return result;
}

Tensor normal(const Shape& shape, double mean, double stddev, DType dtype, std::optional<uint64_t> seed) {
  if (stddev < 0.0) {
    throw TensorStudioError("normal requires stddev >= 0");
  }
  std::mt19937_64 generator = make_generator(seed);
  std::normal_distribution<double> distribution(mean, stddev);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, distribution(generator));
  }
  if (!seed.has_value()) {
    global_generator() = generator;
  }
  return result;
}

Tensor randint(const Shape& shape, int64_t low, int64_t high, DType dtype, std::optional<uint64_t> seed) {
  if (high <= low) {
    throw TensorStudioError("randint requires high > low");
  }
  if (dtype != DType::Int32 && dtype != DType::Int64) {
    throw DTypeError("randint dtype must be int32 or int64, got " + dtype_name(dtype));
  }
  std::mt19937_64 generator = make_generator(seed);
  std::uniform_int_distribution<int64_t> distribution(low, high - 1);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, static_cast<double>(distribution(generator)));
  }
  if (!seed.has_value()) {
    global_generator() = generator;
  }
  return result;
}

Tensor bernoulli(const Shape& shape, double probability, DType dtype, std::optional<uint64_t> seed) {
  if (probability < 0.0 || probability > 1.0) {
    throw TensorStudioError("bernoulli probability must be in [0, 1]");
  }
  std::mt19937_64 generator = make_generator(seed);
  std::bernoulli_distribution distribution(probability);

  Tensor result(shape, dtype, false);
  for (int64_t i = 0; i < result.numel(); ++i) {
    result.set_value_at_logical(i, distribution(generator) ? 1.0 : 0.0);
  }
  if (!seed.has_value()) {
    global_generator() = generator;
  }
  return result;
}

}  // namespace tensorstudio
