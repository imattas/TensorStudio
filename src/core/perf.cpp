#include "tensorstudio/perf.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <vector>

namespace tensorstudio::perf {
namespace {

constexpr int64_t kDefaultMaxThreads = 32;

std::atomic<int64_t> configured_threads{0};

std::string env_value(const char* name) {
#if defined(_WIN32)
  char* buffer = nullptr;
  std::size_t size = 0;
  if (_dupenv_s(&buffer, &size, name) != 0 || buffer == nullptr) {
    return {};
  }
  std::string value(buffer);
  std::free(buffer);
  return value;
#else
  const char* raw = std::getenv(name);
  return raw == nullptr ? std::string{} : std::string(raw);
#endif
}

bool env_flag_enabled(const char* name) {
  const std::string value = env_value(name);
  if (value.empty()) {
    return false;
  }
  return value == "1" || value == "true" || value == "TRUE" || value == "on" || value == "ON";
}

int64_t env_int(const char* name, int64_t fallback) {
  const std::string raw = env_value(name);
  if (raw.empty()) {
    return fallback;
  }
  char* end = nullptr;
  const long long parsed = std::strtoll(raw.c_str(), &end, 10);
  if (end == raw.c_str() || parsed <= 0) {
    return fallback;
  }
  return static_cast<int64_t>(parsed);
}

int64_t detected_threads() {
  const int64_t explicit_threads = configured_threads.load();
  if (explicit_threads > 0) {
    return explicit_threads;
  }
  const int64_t env_threads = env_int("TENSORSTUDIO_NUM_THREADS", 0);
  if (env_threads > 0) {
    return std::clamp<int64_t>(env_threads, 1, kDefaultMaxThreads);
  }
  const unsigned int hardware = std::thread::hardware_concurrency();
  if (hardware == 0) {
    return 1;
  }
  return std::clamp<int64_t>(static_cast<int64_t>(hardware), 1, kDefaultMaxThreads);
}

class ThreadPool {
 public:
  explicit ThreadPool(int64_t size) {
    workers_.reserve(static_cast<std::size_t>(size));
    for (int64_t i = 0; i < size; ++i) {
      workers_.emplace_back([this]() { worker_loop(); });
    }
  }

  ThreadPool(const ThreadPool&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;

  ~ThreadPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
      if (worker.joinable()) {
        worker.join();
      }
    }
  }

  std::future<void> enqueue(std::function<void()> task) {
    auto packaged = std::make_shared<std::packaged_task<void()>>(std::move(task));
    std::future<void> future = packaged->get_future();
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (stopping_) {
        throw std::runtime_error("TensorStudio thread pool is stopping");
      }
      tasks_.emplace([packaged]() { (*packaged)(); });
    }
    condition_.notify_one();
    return future;
  }

 private:
  void worker_loop() {
    while (true) {
      std::function<void()> task;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this]() { return stopping_ || !tasks_.empty(); });
        if (stopping_ && tasks_.empty()) {
          return;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      task();
    }
  }

  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::queue<std::function<void()>> tasks_;
  bool stopping_{false};
};

std::mutex global_pool_mutex;
std::unique_ptr<ThreadPool> global_pool;
int64_t global_pool_size = 0;

ThreadPool& pool_for_size(int64_t size) {
  std::lock_guard<std::mutex> lock(global_pool_mutex);
  if (!global_pool || global_pool_size != size) {
    global_pool = std::make_unique<ThreadPool>(size);
    global_pool_size = size;
  }
  return *global_pool;
}

}  // namespace

int64_t get_num_threads() {
  return detected_threads();
}

void set_num_threads(int64_t count) {
  if (count <= 0) {
    throw std::invalid_argument("TensorStudio thread count must be positive");
  }
  configured_threads.store(std::clamp<int64_t>(count, 1, kDefaultMaxThreads));
}

bool threads_enabled() {
  return !env_flag_enabled("TENSORSTUDIO_DISABLE_THREADS") && get_num_threads() > 1;
}

bool storage_pool_enabled() {
  return !env_flag_enabled("TENSORSTUDIO_DISABLE_STORAGE_POOL");
}

bool blas_enabled() {
#if defined(TENSORSTUDIO_HAS_CBLAS) || defined(TENSORSTUDIO_HAS_ACCELERATE)
  return true;
#else
  return false;
#endif
}

std::string simd_level() {
#if defined(__AVX2__) || defined(_M_AVX2)
  return "avx2";
#elif defined(__AVX__) || defined(_M_AVX)
  return "avx";
#elif defined(__SSE2__) || defined(_M_X64) || defined(_M_AMD64)
  return "sse2";
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
  return "neon";
#else
  return "compiler-autovectorized";
#endif
}

bool simd_enabled() {
  return simd_level() != "scalar";
}

void parallel_for(
    int64_t begin,
    int64_t end,
    int64_t grain_size,
    const std::function<void(int64_t, int64_t)>& fn) {
  if (end <= begin) {
    return;
  }

  const int64_t count = end - begin;
  const int64_t grain = std::max<int64_t>(grain_size, 1);
  if (!threads_enabled() || count <= grain * 2) {
    fn(begin, end);
    return;
  }

  const int64_t thread_count = get_num_threads();
  const int64_t chunk_count = std::min<int64_t>(thread_count * 4, (count + grain - 1) / grain);
  if (chunk_count <= 1) {
    fn(begin, end);
    return;
  }

  const int64_t chunk_size = (count + chunk_count - 1) / chunk_count;
  std::vector<std::future<void>> futures;
  futures.reserve(static_cast<std::size_t>(chunk_count));
  ThreadPool& pool = pool_for_size(thread_count);

  for (int64_t chunk = 0; chunk < chunk_count; ++chunk) {
    const int64_t chunk_begin = begin + chunk * chunk_size;
    const int64_t chunk_end = std::min(end, chunk_begin + chunk_size);
    if (chunk_begin >= chunk_end) {
      continue;
    }
    futures.push_back(pool.enqueue([chunk_begin, chunk_end, &fn]() { fn(chunk_begin, chunk_end); }));
  }

  for (std::future<void>& future : futures) {
    future.get();
  }
}

PerformanceInfo performance_info() {
  return PerformanceInfo{
      get_num_threads(),
      threads_enabled(),
      storage_pool_enabled(),
      blas_enabled(),
      simd_enabled(),
      simd_level(),
  };
}

}  // namespace tensorstudio::perf
