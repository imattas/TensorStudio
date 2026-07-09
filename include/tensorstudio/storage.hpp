#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorstudio/dtype.hpp"

namespace tensorstudio {

struct StorageTelemetry {
  std::uint64_t total_allocations{0};
  std::uint64_t active_allocations{0};
  std::uint64_t peak_active_allocations{0};
  std::uint64_t total_bytes_allocated{0};
  std::uint64_t active_bytes{0};
  std::uint64_t peak_active_bytes{0};
};

class Storage {
 public:
  Storage() = default;
  explicit Storage(std::size_t size_bytes);
  ~Storage();

  Storage(const Storage&) = delete;
  Storage& operator=(const Storage&) = delete;
  Storage(Storage&&) = delete;
  Storage& operator=(Storage&&) = delete;

  static std::shared_ptr<Storage> allocate(int64_t elements, DType dtype);

  std::byte* data();
  const std::byte* data() const;
  std::size_t size_bytes() const;
  std::uint64_t version() const;
  void bump_version();

 private:
  std::vector<std::byte> data_;
  std::uint64_t version_{0};
  bool tracked_{false};
};

StorageTelemetry storage_telemetry();
void reset_storage_telemetry();

}  // namespace tensorstudio
