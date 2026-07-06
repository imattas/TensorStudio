#include "tensorstudio/tensor.hpp"

#include <algorithm>
#include <cmath>
#include <sstream>

#include "tensorstudio/errors.hpp"

namespace tensorstudio {
namespace {

template <typename T>
T* typed_ptr(const std::shared_ptr<Storage>& storage) {
  return reinterpret_cast<T*>(storage->data());
}

template <typename T>
const T* typed_ptr_const(const std::shared_ptr<Storage>& storage) {
  return reinterpret_cast<const T*>(storage->data());
}

double read_value(const std::shared_ptr<Storage>& storage, DType dtype, int64_t storage_index) {
  const auto index = static_cast<std::size_t>(storage_index);
  switch (dtype) {
    case DType::Float32:
      return static_cast<double>(typed_ptr_const<float>(storage)[index]);
    case DType::Float64:
      return typed_ptr_const<double>(storage)[index];
    case DType::Int32:
      return static_cast<double>(typed_ptr_const<int32_t>(storage)[index]);
    case DType::Int64:
      return static_cast<double>(typed_ptr_const<int64_t>(storage)[index]);
    case DType::Bool:
      return typed_ptr_const<bool>(storage)[index] ? 1.0 : 0.0;
  }
  throw DTypeError("unknown dtype");
}

void write_value(
    const std::shared_ptr<Storage>& storage,
    DType dtype,
    int64_t storage_index,
    double value) {
  const auto index = static_cast<std::size_t>(storage_index);
  switch (dtype) {
    case DType::Float32:
      typed_ptr<float>(storage)[index] = static_cast<float>(value);
      return;
    case DType::Float64:
      typed_ptr<double>(storage)[index] = value;
      return;
    case DType::Int32:
      typed_ptr<int32_t>(storage)[index] = static_cast<int32_t>(value);
      return;
    case DType::Int64:
      typed_ptr<int64_t>(storage)[index] = static_cast<int64_t>(value);
      return;
    case DType::Bool:
      typed_ptr<bool>(storage)[index] = value != 0.0;
      return;
  }
  throw DTypeError("unknown dtype");
}

}  // namespace

Tensor::Tensor() = default;

Tensor::Tensor(const Shape& shape, DType dtype, bool requires_grad)
    : Tensor(Storage::allocate(tensorstudio::numel(shape), dtype),
             dtype,
             shape,
             contiguous_strides(shape),
             0,
             requires_grad) {}

Tensor::Tensor(
    std::shared_ptr<Storage> storage,
    DType dtype,
    Shape shape,
    Shape strides,
    int64_t offset,
    bool requires_grad)
    : impl_(std::make_shared<TensorImpl>()) {
  validate_shape(shape);
  if (shape.size() != strides.size()) {
    throw ShapeError("shape and strides must have the same rank");
  }
  impl_->storage = std::move(storage);
  impl_->dtype = dtype;
  impl_->shape = std::move(shape);
  impl_->strides = std::move(strides);
  impl_->offset = offset;
  impl_->device = cpu_device();
  impl_->autograd = std::make_shared<AutogradMeta>();
  set_requires_grad(requires_grad);
}

Tensor Tensor::empty(const Shape& shape, DType dtype, bool requires_grad) {
  return Tensor(shape, dtype, requires_grad);
}

Tensor Tensor::from_flat_values(
    const std::vector<double>& values,
    const Shape& shape,
    DType dtype,
    bool requires_grad) {
  if (static_cast<int64_t>(values.size()) != tensorstudio::numel(shape)) {
    throw ShapeError(
        "expected " + std::to_string(tensorstudio::numel(shape)) + " values for shape " +
        shape_to_string(shape) + ", got " + std::to_string(values.size()));
  }
  Tensor tensor(shape, dtype, requires_grad);
  for (int64_t i = 0; i < tensor.numel(); ++i) {
    tensor.set_value_at_logical(i, values[static_cast<std::size_t>(i)]);
  }
  return tensor;
}

bool Tensor::defined() const {
  return static_cast<bool>(impl_);
}

void Tensor::ensure_defined() const {
  if (!defined()) {
    throw TensorStudioError("operation received an undefined Tensor");
  }
}

DType Tensor::dtype() const {
  ensure_defined();
  return impl_->dtype;
}

const Shape& Tensor::shape() const {
  ensure_defined();
  return impl_->shape;
}

const Shape& Tensor::strides() const {
  ensure_defined();
  return impl_->strides;
}

int64_t Tensor::offset() const {
  ensure_defined();
  return impl_->offset;
}

int64_t Tensor::ndim() const {
  ensure_defined();
  return static_cast<int64_t>(impl_->shape.size());
}

int64_t Tensor::size() const {
  return numel();
}

int64_t Tensor::numel() const {
  ensure_defined();
  return tensorstudio::numel(impl_->shape);
}

bool Tensor::is_contiguous() const {
  ensure_defined();
  return tensorstudio::is_contiguous(impl_->shape, impl_->strides);
}

Device Tensor::device() const {
  ensure_defined();
  return impl_->device;
}

Tensor Tensor::clone() const {
  ensure_defined();
  Tensor out(shape(), dtype(), requires_grad());
  out.copy_from(*this);
  return out;
}

Tensor Tensor::detach() const {
  ensure_defined();
  return Tensor(impl_->storage, impl_->dtype, impl_->shape, impl_->strides, impl_->offset, false);
}

bool Tensor::requires_grad() const {
  ensure_defined();
  return impl_->autograd && impl_->autograd->requires_grad;
}

void Tensor::set_requires_grad(bool value) {
  ensure_defined();
  if (value && !dtype_is_floating(impl_->dtype)) {
    throw DTypeError("requires_grad is only supported for floating point tensors");
  }
  if (!impl_->autograd) {
    impl_->autograd = std::make_shared<AutogradMeta>();
  }
  impl_->autograd->requires_grad = value;
}

bool Tensor::has_grad() const {
  ensure_defined();
  return impl_->autograd && impl_->autograd->grad && impl_->autograd->grad->defined();
}

Tensor Tensor::grad() const {
  ensure_defined();
  if (!has_grad()) {
    return Tensor();
  }
  return *impl_->autograd->grad;
}

void Tensor::set_grad(const Tensor& grad) {
  ensure_defined();
  if (grad.defined() && grad.shape() != shape()) {
    throw ShapeError(
        "gradient shape " + shape_to_string(grad.shape()) + " does not match tensor shape " +
        shape_to_string(shape()));
  }
  impl_->autograd->grad = std::make_shared<Tensor>(grad);
}

void Tensor::zero_grad() {
  ensure_defined();
  impl_->autograd->grad.reset();
}

double Tensor::value_at_logical(int64_t linear_index) const {
  ensure_defined();
  if (linear_index < 0 || linear_index >= numel()) {
    throw TensorStudioError("tensor index out of range");
  }
  const auto storage_index =
      logical_to_storage_offset(linear_index, impl_->shape, impl_->shape, impl_->strides, impl_->offset);
  return value_at_storage(storage_index);
}

void Tensor::set_value_at_logical(int64_t linear_index, double value) {
  ensure_defined();
  if (linear_index < 0 || linear_index >= numel()) {
    throw TensorStudioError("tensor index out of range");
  }
  const auto storage_index =
      logical_to_storage_offset(linear_index, impl_->shape, impl_->shape, impl_->strides, impl_->offset);
  set_value_at_storage(storage_index, value);
}

double Tensor::value_at_storage(int64_t storage_index) const {
  ensure_defined();
  if (storage_index < 0 ||
      static_cast<std::size_t>(storage_index) * dtype_size(impl_->dtype) >= impl_->storage->size_bytes()) {
    throw TensorStudioError("storage index out of range");
  }
  return read_value(impl_->storage, impl_->dtype, storage_index);
}

void Tensor::set_value_at_storage(int64_t storage_index, double value) {
  ensure_defined();
  if (storage_index < 0 ||
      static_cast<std::size_t>(storage_index) * dtype_size(impl_->dtype) >= impl_->storage->size_bytes()) {
    throw TensorStudioError("storage index out of range");
  }
  write_value(impl_->storage, impl_->dtype, storage_index, value);
}

std::vector<double> Tensor::to_flat_vector() const {
  ensure_defined();
  std::vector<double> values(static_cast<std::size_t>(numel()));
  if (is_contiguous()) {
    for (int64_t i = 0; i < numel(); ++i) {
      values[static_cast<std::size_t>(i)] = value_at_storage(offset() + i);
    }
  } else {
    for (int64_t i = 0; i < numel(); ++i) {
      values[static_cast<std::size_t>(i)] = value_at_logical(i);
    }
  }
  return values;
}

void Tensor::copy_from(const Tensor& other) {
  ensure_defined();
  if (shape() != other.shape()) {
    throw ShapeError(
        "cannot assign tensor with shape " + shape_to_string(other.shape()) + " into tensor with shape " +
        shape_to_string(shape()));
  }
  if (is_contiguous() && other.is_contiguous()) {
    for (int64_t i = 0; i < numel(); ++i) {
      set_value_at_storage(offset() + i, other.value_at_storage(other.offset() + i));
    }
  } else {
    for (int64_t i = 0; i < numel(); ++i) {
      set_value_at_logical(i, other.value_at_logical(i));
    }
  }
}

void Tensor::add_scaled_(const Tensor& other, double scale) {
  ensure_defined();
  if (shape() != other.shape()) {
    throw ShapeError(
        "cannot add tensor with shape " + shape_to_string(other.shape()) + " into tensor with shape " +
        shape_to_string(shape()));
  }
  if (is_contiguous() && other.is_contiguous()) {
    for (int64_t i = 0; i < numel(); ++i) {
      set_value_at_storage(
          offset() + i,
          value_at_storage(offset() + i) + other.value_at_storage(other.offset() + i) * scale);
    }
  } else {
    for (int64_t i = 0; i < numel(); ++i) {
      set_value_at_logical(i, value_at_logical(i) + other.value_at_logical(i) * scale);
    }
  }
}

std::shared_ptr<TensorImpl> Tensor::impl() const {
  ensure_defined();
  return impl_;
}

std::string Tensor::repr() const {
  if (!defined()) {
    return "Tensor(undefined)";
  }
  std::ostringstream out;
  out << "Tensor(shape=" << shape_to_string(shape()) << ", dtype='" << dtype_name(dtype()) << "'";
  if (requires_grad()) {
    out << ", requires_grad=True";
  }
  out << ")";
  return out.str();
}

}  // namespace tensorstudio
