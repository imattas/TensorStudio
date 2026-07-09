#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "tensorstudio/device.hpp"
#include "tensorstudio/dtype.hpp"
#include "tensorstudio/shape.hpp"
#include "tensorstudio/storage.hpp"

namespace tensorstudio {

struct TensorImpl;

class Tensor {
 public:
  Tensor();
  Tensor(const Shape& shape, DType dtype, bool requires_grad = false);
  Tensor(
      std::shared_ptr<Storage> storage,
      DType dtype,
      Shape shape,
      Shape strides,
      int64_t offset,
      bool requires_grad = false);

  static Tensor empty(const Shape& shape, DType dtype, bool requires_grad = false);
  static Tensor from_flat_values(
      const std::vector<double>& values,
      const Shape& shape,
      DType dtype,
      bool requires_grad = false);

  bool defined() const;
  DType dtype() const;
  const Shape& shape() const;
  const Shape& strides() const;
  int64_t offset() const;
  int64_t ndim() const;
  int64_t size() const;
  int64_t numel() const;
  bool is_contiguous() const;
  Device device() const;
  Tensor to_device(const Device& device) const;
  Tensor cpu() const;
  Tensor clone() const;
  Tensor detach() const;
  void detach_();
  bool is_leaf() const;
  void clear_history();

  bool requires_grad() const;
  void set_requires_grad(bool value);
  bool has_grad() const;
  Tensor grad() const;
  void set_grad(const Tensor& grad);
  void zero_grad();

  double value_at_logical(int64_t linear_index) const;
  void set_value_at_logical(int64_t linear_index, double value);
  double value_at_storage(int64_t storage_index) const;
  void set_value_at_storage(int64_t storage_index, double value);
  std::vector<double> to_flat_vector() const;
  void copy_from(const Tensor& other);
  void add_scaled_(const Tensor& other, double scale);
  void add_(const Tensor& other, double alpha = 1.0);
  void fill_(double value);
  void zero_();

  std::shared_ptr<TensorImpl> impl() const;
  std::string repr() const;

 private:
  std::shared_ptr<TensorImpl> impl_;

  void ensure_defined() const;
  void ensure_inplace_allowed(const std::string& op_name) const;
};

using BackwardFn = std::function<std::vector<Tensor>(const Tensor& out_grad)>;

struct AutogradMeta {
  bool requires_grad{false};
  std::shared_ptr<Tensor> grad{};
  BackwardFn backward{};
  std::vector<Tensor> parents{};
  bool graph_consumed{false};
};

struct TensorImpl {
  std::shared_ptr<Storage> storage{};
  DType dtype{DType::Float32};
  Shape shape{};
  Shape strides{};
  int64_t offset{0};
  Device device{};
  std::shared_ptr<AutogradMeta> autograd{};
};

}  // namespace tensorstudio
