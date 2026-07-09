"""Autograd helpers."""

from __future__ import annotations

from collections.abc import Callable

from . import _C
from .grad_mode import no_grad
from .tensor import Tensor, zeros_like


def backward(output: Tensor, gradient: Tensor | None = None) -> None:
    """Run reverse-mode automatic differentiation from an output tensor."""

    _C.backward(output, gradient)


def checkpoint(function: Callable[..., Tensor], *args: Tensor) -> Tensor:
    """Run ``function`` without saving intermediates and recompute it in backward.

    This eager checkpoint helper supports Tensor outputs and Tensor positional
    inputs. Non-tensor configuration should be closed over by ``function``.
    """

    if not args:
        raise ValueError("checkpoint requires at least one Tensor input")
    if any(not isinstance(arg, Tensor) for arg in args):
        raise TypeError("checkpoint currently accepts Tensor positional inputs only")

    saved_inputs = tuple(arg.detach().clone() for arg in args)
    requires_grad = tuple(arg.requires_grad for arg in args)

    with no_grad():
        output = function(*args)
    if not isinstance(output, Tensor):
        raise TypeError("checkpointed function must return a Tensor")
    if not any(requires_grad):
        return output

    def backward_fn(grad: Tensor) -> list[Tensor]:
        recompute_inputs = [saved.detach().clone() for saved in saved_inputs]
        for recompute_input, needs_grad in zip(recompute_inputs, requires_grad, strict=True):
            recompute_input.requires_grad = needs_grad
        recomputed = function(*recompute_inputs)
        if not isinstance(recomputed, Tensor):
            raise TypeError("checkpointed function must return a Tensor during backward")
        recomputed.backward(grad)
        grads: list[Tensor] = []
        for recompute_input in recompute_inputs:
            if recompute_input.requires_grad and recompute_input.grad is not None:
                grads.append(recompute_input.grad)
            else:
                grads.append(zeros_like(recompute_input, requires_grad=False))
        return grads

    _C._set_history(output, list(args), backward_fn)
    return output


__all__ = ["backward", "checkpoint"]
