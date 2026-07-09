#include "tensorstudio/autograd.hpp"

#include <algorithm>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorstudio/errors.hpp"
#include "tensorstudio/ops.hpp"

namespace tensorstudio {
namespace {

thread_local bool grad_enabled_state = true;

}  // namespace

bool any_requires_grad(const std::vector<Tensor>& tensors) {
  return std::any_of(tensors.begin(), tensors.end(), [](const Tensor& tensor) {
    return tensor.defined() && tensor.requires_grad();
  });
}

bool grad_enabled() {
  return grad_enabled_state;
}

void set_grad_enabled(bool enabled) {
  grad_enabled_state = enabled;
}

std::vector<std::uint64_t> capture_parent_versions(const std::vector<Tensor>& parents) {
  std::vector<std::uint64_t> versions;
  versions.reserve(parents.size());
  for (const auto& parent : parents) {
    versions.push_back(parent.defined() ? parent.storage_version() : 0);
  }
  return versions;
}

void set_history(Tensor& output, std::vector<Tensor> parents, BackwardFn backward_fn) {
  if (!output.defined()) {
    throw TensorStudioError("cannot set autograd history on an undefined tensor");
  }
  if (!grad_enabled()) {
    return;
  }
  if (!dtype_is_floating(output.dtype())) {
    return;
  }
  if (!any_requires_grad(parents)) {
    return;
  }
  auto parent_versions = capture_parent_versions(parents);
  output.set_requires_grad(true);
  auto meta = output.impl()->autograd;
  meta->parents = std::move(parents);
  meta->parent_versions = std::move(parent_versions);
  meta->backward = std::move(backward_fn);
}

void accumulate_grad(Tensor& tensor, const Tensor& gradient) {
  if (!tensor.requires_grad()) {
    return;
  }
  if (gradient.shape() != tensor.shape()) {
    throw ShapeError(
        "cannot accumulate gradient with shape " + shape_to_string(gradient.shape()) +
        " into tensor with shape " + shape_to_string(tensor.shape()));
  }
  if (!tensor.has_grad()) {
    tensor.set_grad(gradient);
    return;
  }
  Tensor updated = add(tensor.grad(), gradient);
  updated.set_requires_grad(false);
  tensor.set_grad(updated);
}

namespace {

void build_topology(
    const Tensor& tensor,
    std::unordered_set<TensorImpl*>& seen,
    std::vector<Tensor>& topo) {
  if (!tensor.defined()) {
    return;
  }
  TensorImpl* key = tensor.impl().get();
  if (seen.contains(key)) {
    return;
  }
  seen.insert(key);
  const auto meta = tensor.impl()->autograd;
  if (meta) {
    for (const auto& parent : meta->parents) {
      build_topology(parent, seen, topo);
    }
  }
  topo.push_back(tensor);
}

void validate_saved_versions(const Tensor& tensor) {
  const auto meta = tensor.impl()->autograd;
  if (!meta || meta->parents.empty()) {
    return;
  }
  if (meta->parent_versions.size() != meta->parents.size()) {
    throw AutogradError("autograd metadata is missing saved tensor version information");
  }
  for (std::size_t i = 0; i < meta->parents.size(); ++i) {
    const Tensor& parent = meta->parents[i];
    if (!parent.defined()) {
      continue;
    }
    const auto current_version = parent.storage_version();
    const auto saved_version = meta->parent_versions[i];
    if (current_version != saved_version) {
      throw AutogradError(
          "a tensor needed for gradient computation was modified in-place after autograd history was "
          "recorded: parent " +
          std::to_string(i) + " with shape " + shape_to_string(parent.shape()) + " is at storage version " +
          std::to_string(current_version) + ", expected version " + std::to_string(saved_version));
    }
  }
}

}  // namespace

void backward(Tensor& output, const std::optional<Tensor>& gradient) {
  if (!output.defined()) {
    throw AutogradError("cannot run backward on an undefined tensor");
  }
  if (!output.requires_grad()) {
    return;
  }

  Tensor initial_grad;
  if (gradient.has_value()) {
    initial_grad = *gradient;
    if (initial_grad.shape() != output.shape()) {
      throw ShapeError(
          "backward gradient shape " + shape_to_string(initial_grad.shape()) +
          " does not match output shape " + shape_to_string(output.shape()));
    }
  } else {
    if (output.numel() != 1) {
      throw AutogradError(
          "backward() without a gradient is only supported for scalar outputs; got shape " +
          shape_to_string(output.shape()));
    }
    initial_grad = full(output.shape(), 1.0, output.dtype());
  }

  output.set_grad(initial_grad);

  std::unordered_set<TensorImpl*> seen;
  std::vector<Tensor> topo;
  build_topology(output, seen, topo);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Tensor current = *it;
    const auto meta = current.impl()->autograd;
    if (!meta || !meta->backward || !current.has_grad()) {
      continue;
    }
    validate_saved_versions(current);
    const auto parent_grads = meta->backward(current.grad());
    if (parent_grads.size() != meta->parents.size()) {
      throw AutogradError("backward function returned the wrong number of gradients");
    }
    for (std::size_t i = 0; i < parent_grads.size(); ++i) {
      Tensor parent = meta->parents[i];
      if (parent.requires_grad()) {
        Tensor grad = parent_grads[i];
        grad.set_requires_grad(false);
        accumulate_grad(parent, grad);
      }
    }
  }
}

}  // namespace tensorstudio
