#include "tensorstudio/shape.hpp"

#include <algorithm>
#include <numeric>
#include <sstream>

#include "tensorstudio/errors.hpp"

namespace tensorstudio {

void validate_shape(const Shape& shape) {
  for (const auto dim : shape) {
    if (dim < 0) {
      throw ShapeError("shape dimensions must be non-negative, got " + shape_to_string(shape));
    }
  }
}

int64_t numel(const Shape& shape) {
  validate_shape(shape);
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), int64_t{1}, [](int64_t acc, int64_t dim) {
    return acc * dim;
  });
}

Shape contiguous_strides(const Shape& shape) {
  validate_shape(shape);
  Shape strides(shape.size(), 1);
  int64_t stride = 1;
  for (int64_t i = static_cast<int64_t>(shape.size()) - 1; i >= 0; --i) {
    strides[static_cast<std::size_t>(i)] = stride;
    stride *= shape[static_cast<std::size_t>(i)];
  }
  return strides;
}

bool is_contiguous(const Shape& shape, const Shape& strides) {
  return strides == contiguous_strides(shape);
}

std::string shape_to_string(const Shape& shape) {
  std::ostringstream out;
  out << "(";
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) {
      out << ", ";
    }
    out << shape[i];
  }
  if (shape.size() == 1) {
    out << ",";
  }
  out << ")";
  return out.str();
}

Shape broadcast_shapes(const Shape& left, const Shape& right) {
  const auto rank = std::max(left.size(), right.size());
  Shape result(rank, 1);
  for (std::size_t i = 0; i < rank; ++i) {
    const auto left_index = static_cast<int64_t>(left.size()) - 1 - static_cast<int64_t>(i);
    const auto right_index = static_cast<int64_t>(right.size()) - 1 - static_cast<int64_t>(i);
    const int64_t left_dim = left_index >= 0 ? left[static_cast<std::size_t>(left_index)] : 1;
    const int64_t right_dim = right_index >= 0 ? right[static_cast<std::size_t>(right_index)] : 1;
    if (left_dim != right_dim && left_dim != 1 && right_dim != 1) {
      throw ShapeError(
          "cannot broadcast shapes " + shape_to_string(left) + " and " + shape_to_string(right));
    }
    result[rank - 1 - i] = std::max(left_dim, right_dim);
  }
  return result;
}

int64_t logical_to_storage_offset(
    int64_t linear_index,
    const Shape& logical_shape,
    const Shape& tensor_shape,
    const Shape& tensor_strides,
    int64_t tensor_offset) {
  if (logical_shape.empty()) {
    return tensor_offset;
  }

  int64_t storage_offset = tensor_offset;
  const Shape coords = coordinates_from_linear(linear_index, logical_shape);
  const auto logical_rank = logical_shape.size();
  const auto tensor_rank = tensor_shape.size();

  for (std::size_t dim = 0; dim < logical_rank; ++dim) {
    const auto tensor_dim_aligned =
        static_cast<int64_t>(dim) - static_cast<int64_t>(logical_rank - tensor_rank);
    if (tensor_dim_aligned < 0) {
      continue;
    }
    const auto tensor_dim = static_cast<std::size_t>(tensor_dim_aligned);
    const int64_t used_coord = tensor_shape[tensor_dim] == 1 ? 0 : coords[dim];
    storage_offset += used_coord * tensor_strides[tensor_dim];
  }
  return storage_offset;
}

Shape coordinates_from_linear(int64_t linear_index, const Shape& shape) {
  Shape coords(shape.size(), 0);
  int64_t remaining = linear_index;
  for (std::size_t dim = 0; dim < shape.size(); ++dim) {
    int64_t suffix = 1;
    for (std::size_t j = dim + 1; j < shape.size(); ++j) {
      suffix *= shape[j];
    }
    coords[dim] = suffix == 0 ? 0 : remaining / suffix;
    remaining = suffix == 0 ? 0 : remaining % suffix;
  }
  return coords;
}

Shape normalize_shape(const Shape& shape, int64_t expected_numel) {
  Shape normalized = shape;
  int64_t infer_index = -1;
  int64_t known_product = 1;
  for (std::size_t i = 0; i < normalized.size(); ++i) {
    const auto dim = normalized[i];
    if (dim == -1) {
      if (infer_index != -1) {
        throw ShapeError("only one reshape dimension can be inferred");
      }
      infer_index = static_cast<int64_t>(i);
      continue;
    }
    if (dim < 0) {
      throw ShapeError("reshape dimensions must be non-negative or -1");
    }
    known_product *= dim;
  }

  if (infer_index != -1) {
    if (known_product == 0 || expected_numel % known_product != 0) {
      throw ShapeError("cannot infer reshape dimension for " + shape_to_string(shape));
    }
    normalized[static_cast<std::size_t>(infer_index)] = expected_numel / known_product;
  }

  if (numel(normalized) != expected_numel) {
    throw ShapeError(
        "cannot reshape tensor with " + std::to_string(expected_numel) + " elements to shape " +
        shape_to_string(shape));
  }
  return normalized;
}

}  // namespace tensorstudio
