#pragma once

#include <optional>
#include <vector>

#include "tensorstudio/tensor.hpp"

namespace tensorstudio {

bool any_requires_grad(const std::vector<Tensor>& tensors);
bool grad_enabled();
void set_grad_enabled(bool enabled);
void set_history(Tensor& output, std::vector<Tensor> parents, BackwardFn backward);
void backward(
    Tensor& output,
    const std::optional<Tensor>& gradient = std::nullopt,
    bool retain_graph = false);
void accumulate_grad(Tensor& tensor, const Tensor& gradient);

}  // namespace tensorstudio
