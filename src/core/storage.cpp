#include "tensorstudio/storage.hpp"

#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <mutex>
#include <unordered_map>

#include "tensorstudio/errors.hpp"
#include "tensorstudio/perf.hpp"
#include "tensorstudio/shape.hpp"

namespace tensorstudio {
namespace {

constexpr std::size_t kDefaultMaxPoolBlockBytes = 16 * 1024 * 1024;
constexpr std::size_t kDefaultMaxBlocksPerSize = 8;

std::mutex pool_mutex;
std::unordered_map<std::size_t, std::vector<Storage*>> storage_pool;

std::atomic<std::uint64_t>& total_allocations_counter() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

std::atomic<std::uint64_t>& active_allocations_counter() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

std::atomic<std::uint64_t>& peak_active_allocations_counter() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

std::atomic<std::uint64_t>& total_bytes_counter() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

std::atomic<std::uint64_t>& active_bytes_counter() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

std::atomic<std::uint64_t>& peak_active_bytes_counter() {
  static std::atomic<std::uint64_t> value{0};
  return value;
}

void raise_peak(std::atomic<std::uint64_t>& peak, std::uint64_t candidate) {
  std::uint64_t current = peak.load(std::memory_order_relaxed);
  while (candidate > current &&
         !peak.compare_exchange_weak(current, candidate, std::memory_order_relaxed)) {
  }
}

void record_checkout(std::size_t size_bytes) {
  const auto bytes = static_cast<std::uint64_t>(size_bytes);
  total_allocations_counter().fetch_add(1, std::memory_order_relaxed);
  total_bytes_counter().fetch_add(bytes, std::memory_order_relaxed);
  const auto active_allocations =
      active_allocations_counter().fetch_add(1, std::memory_order_relaxed) + 1;
  const auto active_bytes = active_bytes_counter().fetch_add(bytes, std::memory_order_relaxed) + bytes;
  raise_peak(peak_active_allocations_counter(), active_allocations);
  raise_peak(peak_active_bytes_counter(), active_bytes);
}

void record_release(std::size_t size_bytes) {
  active_allocations_counter().fetch_sub(1, std::memory_order_relaxed);
  active_bytes_counter().fetch_sub(static_cast<std::uint64_t>(size_bytes), std::memory_order_relaxed);
}

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

std::size_t env_size(const char* name, std::size_t fallback) {
  const std::string raw = env_value(name);
  if (raw.empty()) {
    return fallback;
  }
  char* end = nullptr;
  const unsigned long long parsed = std::strtoull(raw.c_str(), &end, 10);
  if (end == raw.c_str() || parsed == 0) {
    return fallback;
  }
  return static_cast<std::size_t>(parsed);
}

std::size_t max_pool_block_bytes() {
  return env_size("TENSORSTUDIO_STORAGE_POOL_MAX_BLOCK_BYTES", kDefaultMaxPoolBlockBytes);
}

Storage* acquire_storage(std::size_t size_bytes) {
  if (!perf::storage_pool_enabled() || size_bytes == 0 || size_bytes > max_pool_block_bytes()) {
    return nullptr;
  }
  std::lock_guard<std::mutex> lock(pool_mutex);
  auto found = storage_pool.find(size_bytes);
  if (found == storage_pool.end() || found->second.empty()) {
    return nullptr;
  }
  Storage* storage = found->second.back();
  found->second.pop_back();
  return storage;
}

}  // namespace

Storage::Storage(std::size_t size_bytes, Device device) : data_(size_bytes), device_(device) {}

std::shared_ptr<Storage> Storage::allocate(int64_t elements, DType dtype) {
  return allocate(elements, dtype, cpu_device());
}

std::shared_ptr<Storage> Storage::allocate(int64_t elements, DType dtype, const Device& device) {
  if (elements < 0) {
    throw ShapeError("cannot allocate storage for negative element count");
  }
  if (!is_device_available(device)) {
    throw DeviceError("cannot allocate storage on unavailable device '" + device.str() + "'");
  }
  if (!device.is_cpu()) {
    throw DeviceError("accelerator storage allocation is not implemented for device '" + device.str() + "'");
  }
  const std::size_t size_bytes = static_cast<std::size_t>(elements) * dtype_size(dtype);
  Storage* storage = acquire_storage(size_bytes);
  if (storage == nullptr) {
    storage = new Storage(size_bytes, device);
  } else {
    std::fill(storage->data_.begin(), storage->data_.end(), std::byte{0});
    storage->device_ = device;
  }
  record_checkout(size_bytes);
  return std::shared_ptr<Storage>(storage, [](Storage* item) { Storage::release(item); });
}

void Storage::release(Storage* storage) {
  if (storage == nullptr) {
    return;
  }
  const std::size_t size_bytes = storage->size_bytes();
  record_release(size_bytes);
  if (!perf::storage_pool_enabled() || size_bytes == 0 || size_bytes > max_pool_block_bytes()) {
    delete storage;
    return;
  }
  std::lock_guard<std::mutex> lock(pool_mutex);
  auto& bucket = storage_pool[size_bytes];
  if (bucket.size() >= kDefaultMaxBlocksPerSize) {
    delete storage;
    return;
  }
  bucket.push_back(storage);
}

std::byte* Storage::data() {
  return data_.data();
}

const std::byte* Storage::data() const {
  return data_.data();
}

std::size_t Storage::size_bytes() const {
  return data_.size();
}

Device Storage::device() const {
  return device_;
}

StorageTelemetry storage_telemetry() {
  StorageTelemetry telemetry;
  telemetry.total_allocations = total_allocations_counter().load(std::memory_order_relaxed);
  telemetry.active_allocations = active_allocations_counter().load(std::memory_order_relaxed);
  telemetry.peak_active_allocations = peak_active_allocations_counter().load(std::memory_order_relaxed);
  telemetry.total_bytes_allocated = total_bytes_counter().load(std::memory_order_relaxed);
  telemetry.active_bytes = active_bytes_counter().load(std::memory_order_relaxed);
  telemetry.peak_active_bytes = peak_active_bytes_counter().load(std::memory_order_relaxed);
  return telemetry;
}

void reset_storage_telemetry() {
  const auto active_allocations = active_allocations_counter().load(std::memory_order_relaxed);
  const auto active_bytes = active_bytes_counter().load(std::memory_order_relaxed);
  total_allocations_counter().store(0, std::memory_order_relaxed);
  total_bytes_counter().store(0, std::memory_order_relaxed);
  peak_active_allocations_counter().store(active_allocations, std::memory_order_relaxed);
  peak_active_bytes_counter().store(active_bytes, std::memory_order_relaxed);
}

}  // namespace tensorstudio
