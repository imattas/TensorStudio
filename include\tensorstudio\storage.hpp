#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "tensorstudio/dtype.hpp"

namespace tensorstudio {

class Storage {
 public:
  Storage() = default;
  explicit Storage(std::size_t size_bytes);

  static std::shared_ptr<Storage> allocate(int64_t elements, DType dtype);

  std::byte* data();
  const std::byte* data() const;
  std::size_t size_bytes() const;

 private:
  std::vector<std::byte> data_;
};

}  // namespace tensorstudio
