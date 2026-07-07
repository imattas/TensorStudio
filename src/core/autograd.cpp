#include "tensorstudio/autograd.hpp"

#include <algorithm>
#include <unordered_set>

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
  output.set_requires_grad(true);
  auto meta = output.impl()->autograd;
  meta->parents = std::move(parents);
  meta->backward = std::move(backward_fn);
  meta->graph_consumed = false;
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

void release_graph(const std::vector<Tensor>& topo) {
  for (const Tensor& tensor : topo) {
    const auto meta = tensor.impl()->autograd;
    if (!meta || !meta->backward) {
      continue;
    }
    meta->parents.clear();
    meta->backward = {};
    meta->graph_consumed = true;
  }
}

void clear_intermediate_grads(const std::vector<Tensor>& topo) {
  for (const Tensor& tensor : topo) {
    const auto meta = tensor.impl()->autograd;
    if (!meta || !meta->backward) {
      continue;
    }
    Tensor current = tensor;
    current.zero_grad();
  }
}

}  // namespace

void backward(Tensor& output, const std::optional<Tensor>& gradient, bool retain_graph) {
  if (!output.defined()) {
    throw AutogradError("cannot run backward on an undefined tensor");
  }
  if (!output.requires_grad()) {
    return;
  }
  const auto output_meta = output.impl()->autograd;
  if (output_meta && output_meta->graph_consumed) {
    throw AutogradError(
        "cannot run backward through a graph that has already been freed; pass retain_graph=True "
        "on the first backward call if the graph must be reused");
  }

  Tensor initial_grad;
  if (gradient.has_value()) {
    initial_grad = gradient->detach();
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

  std::unordered_set<TensorImpl*> seen;
  std::vector<Tensor> topo;
  build_topology(output, seen, topo);
  clear_intermediate_grads(topo);
  output.set_grad(initial_grad);

  for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
    Tensor current = *it;
    const auto meta = current.impl()->autograd;
    if (!meta || !meta->backward || !current.has_grad()) {
      continue;
    }
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

  if (!retain_graph) {
    release_graph(topo);
  }
}

}  // namespace tensorstudio
