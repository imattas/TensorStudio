#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorstudio/device.hpp"
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
  explicit Storage(std::size_t size_bytes, Device device = cpu_device());

  static std::shared_ptr<Storage> allocate(int64_t elements, DType dtype);
  static std::shared_ptr<Storage> allocate(int64_t elements, DType dtype, const Device& device);

  std::byte* data();
  const std::byte* data() const;
  std::size_t size_bytes() const;
  Device device() const;

 private:
  static void release(Storage* storage);

  std::vector<std::byte> data_;
  Device device_{};
};

StorageTelemetry storage_telemetry();
void reset_storage_telemetry();

}  // namespace tensorstudio
