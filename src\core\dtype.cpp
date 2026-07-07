#include "tensorstudio/dtype.hpp"

#include <algorithm>
#include <cctype>

#include "tensorstudio/errors.hpp"

namespace tensorstudio {

std::string dtype_name(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return "float32";
    case DType::Float64:
      return "float64";
    case DType::Int32:
      return "int32";
    case DType::Int64:
      return "int64";
    case DType::Bool:
      return "bool";
  }
  throw DTypeError("unknown dtype");
}

DType dtype_from_string(const std::string& name) {
  std::string normalized = name;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
    return static_cast<char>(std::tolower(ch));
  });
  if (normalized == "float32" || normalized == "f32") {
    return DType::Float32;
  }
  if (normalized == "float64" || normalized == "float" || normalized == "double" ||
      normalized == "f64") {
    return DType::Float64;
  }
  if (normalized == "int32" || normalized == "i32") {
    return DType::Int32;
  }
  if (normalized == "int64" || normalized == "int" || normalized == "i64") {
    return DType::Int64;
  }
  if (normalized == "bool" || normalized == "boolean") {
    return DType::Bool;
  }
  throw DTypeError("unsupported dtype '" + name + "'");
}

std::size_t dtype_size(DType dtype) {
  switch (dtype) {
    case DType::Float32:
      return sizeof(float);
    case DType::Float64:
      return sizeof(double);
    case DType::Int32:
      return sizeof(int32_t);
    case DType::Int64:
      return sizeof(int64_t);
    case DType::Bool:
      return sizeof(bool);
  }
  throw DTypeError("unknown dtype");
}

bool dtype_is_floating(DType dtype) {
  return dtype == DType::Float32 || dtype == DType::Float64;
}

DType promote_types(DType left, DType right) {
  if (left == DType::Float64 || right == DType::Float64) {
    return DType::Float64;
  }
  if (left == DType::Float32 || right == DType::Float32) {
    return DType::Float32;
  }
  if (left == DType::Int64 || right == DType::Int64) {
    return DType::Int64;
  }
  if (left == DType::Int32 || right == DType::Int32) {
    return DType::Int32;
  }
  return DType::Bool;
}

DType default_float_dtype() {
  return DType::Float32;
}

}  // namespace tensorstudio
