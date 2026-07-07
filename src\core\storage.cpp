#include "tensorstudio/storage.hpp"

#include "tensorstudio/errors.hpp"
#include "tensorstudio/shape.hpp"

namespace tensorstudio {

Storage::Storage(std::size_t size_bytes) : data_(size_bytes) {}

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

}  // namespace tensorstudio
