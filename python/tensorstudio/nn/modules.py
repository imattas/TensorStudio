"""Small neural network module system built on TensorStudio tensors."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

from tensorstudio.tensor import Tensor, randn, tensor, zeros


class Parameter:
    """Factory class that creates tensors with ``requires_grad=True``."""

    def __new__(cls, data: Any, dtype: str | None = None) -> Tensor:  # type: ignore[misc]
        if isinstance(data, Tensor):
            data.requires_grad = True
            return data
        return tensor(data, dtype=dtype, requires_grad=True)


class Module:
    """Base class for Python-level neural network modules."""

    training: bool

    def __init__(self) -> None:
        self.training = True

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        seen: set[int] = set()
        return list(self._iter_parameters(seen))

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.zero_grad()

    def train(self) -> Module:
        self.training = True
        for module in self._children():
            module.train()
        return self

    def eval(self) -> Module:
        self.training = False
        for module in self._children():
            module.eval()
        return self

    def _children(self) -> Iterator[Module]:
        for value in self.__dict__.values():
            yield from self._modules_from_value(value)

    def _iter_parameters(self, seen: set[int]) -> Iterator[Tensor]:
        for value in self.__dict__.values():
            yield from self._parameters_from_value(value, seen)

    @classmethod
    def _parameters_from_value(cls, value: Any, seen: set[int]) -> Iterator[Tensor]:
        if isinstance(value, Tensor):
            marker = id(value)
            if value.requires_grad and marker not in seen:
                seen.add(marker)
                yield value
        elif isinstance(value, Module):
            yield from value._iter_parameters(seen)
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls._parameters_from_value(item, seen)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._parameters_from_value(item, seen)

    @classmethod
    def _modules_from_value(cls, value: Any) -> Iterator[Module]:
        if isinstance(value, Module):
            yield value
        elif isinstance(value, dict):
            for item in value.values():
                yield from cls._modules_from_value(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from cls._modules_from_value(item)


class Linear(Module):
    """Fully connected layer with shape ``input @ weight.T + bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        scale = 1.0 / math.sqrt(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = cast(Tensor, Parameter(randn((out_features, in_features), seed=42) * scale))
        self.bias = cast(Tensor, Parameter(zeros((out_features,)))) if bias else None

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output


class Sequential(Module):
    """A simple ordered container of modules."""

    def __init__(self, *modules: Module) -> None:
        super().__init__()
        self.modules = list(modules)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for module in self.modules:
            output = module(output)
        return output

    def __iter__(self) -> Iterator[Module]:
        return iter(self.modules)

    def __len__(self) -> int:
        return len(self.modules)

    def __getitem__(self, index: int) -> Module:
        return self.modules[index]


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()


__all__ = ["Linear", "Module", "Parameter", "ReLU", "Sequential", "Sigmoid", "Tanh"]
