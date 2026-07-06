#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tensorstudio {

using Shape = std::vector<int64_t>;

void validate_shape(const Shape& shape);
int64_t numel(const Shape& shape);
Shape contiguous_strides(const Shape& shape);
bool is_contiguous(const Shape& shape, const Shape& strides);
std::string shape_to_string(const Shape& shape);
Shape broadcast_shapes(const Shape& left, const Shape& right);
int64_t logical_to_storage_offset(
    int64_t linear_index,
    const Shape& logical_shape,
    const Shape& tensor_shape,
    const Shape& tensor_strides,
    int64_t tensor_offset);
Shape coordinates_from_linear(int64_t linear_index, const Shape& shape);
Shape normalize_shape(const Shape& shape, int64_t expected_numel);

}  // namespace tensorstudio
