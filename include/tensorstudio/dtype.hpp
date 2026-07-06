#pragma once

#include <cstddef>
#include <string>

namespace tensorstudio {

enum class DType {
  Float32,
  Float64,
  Int32,
  Int64,
  Bool,
};

std::string dtype_name(DType dtype);
DType dtype_from_string(const std::string& name);
std::size_t dtype_size(DType dtype);
bool dtype_is_floating(DType dtype);
DType promote_types(DType left, DType right);
DType default_float_dtype();

}  // namespace tensorstudio
