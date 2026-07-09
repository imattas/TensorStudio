#include "tensorstudio/storage.hpp"

#include <algorithm>
#include <atomic>

#include "tensorstudio/errors.hpp"
#include "tensorstudio/shape.hpp"

namespace tensorstudio {
namespace {

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

void record_allocation(std::size_t size_bytes) {
  const auto bytes = static_cast<std::uint64_t>(size_bytes);
  total_allocations_counter().fetch_add(1, std::memory_order_relaxed);
  total_bytes_counter().fetch_add(bytes, std::memory_order_relaxed);
  const auto active_allocations =
      active_allocations_counter().fetch_add(1, std::memory_order_relaxed) + 1;
  const auto active_bytes = active_bytes_counter().fetch_add(bytes, std::memory_order_relaxed) + bytes;
  raise_peak(peak_active_allocations_counter(), active_allocations);
  raise_peak(peak_active_bytes_counter(), active_bytes);
}

void record_deallocation(std::size_t size_bytes) {
  active_allocations_counter().fetch_sub(1, std::memory_order_relaxed);
  active_bytes_counter().fetch_sub(static_cast<std::uint64_t>(size_bytes), std::memory_order_relaxed);
}

}  // namespace

Storage::Storage(std::size_t size_bytes) : data_(size_bytes), tracked_(true) {
  record_allocation(size_bytes);
}

Storage::~Storage() {
  if (tracked_) {
    record_deallocation(data_.size());
  }
}

std::shared_ptr<Storage> Storage::allocate(int64_t elements, DType dtype) {
  if (elements < 0) {
    throw ShapeError("cannot allocate storage for negative element count");
  }
  return std::make_shared<Storage>(static_cast<std::size_t>(elements) * dtype_size(dtype));
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

std::uint64_t Storage::version() const {
  return version_;
}

void Storage::bump_version() {
  ++version_;
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
