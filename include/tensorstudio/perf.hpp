#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace tensorstudio::perf {

struct PerformanceInfo {
  int64_t num_threads{1};
  bool threads_enabled{false};
  bool storage_pool_enabled{true};
  bool blas_enabled{false};
  bool simd_enabled{false};
  std::string simd_level{"scalar"};
};

int64_t get_num_threads();
void set_num_threads(int64_t count);
bool threads_enabled();
bool storage_pool_enabled();
bool blas_enabled();
bool simd_enabled();
std::string simd_level();

void parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& fn);

PerformanceInfo performance_info();

}  // namespace tensorstudio::perf
