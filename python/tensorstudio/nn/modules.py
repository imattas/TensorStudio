"""Small neural network module system built on TensorStudio tensors."""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Any, cast

from tensorstudio.tensor import Tensor, rand, randn, tensor, zeros


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
        return [parameter for _, parameter in self.named_parameters()]

    def named_parameters(self, prefix: str = "") -> list[tuple[str, Tensor]]:
        seen: set[int] = set()
        return list(self._iter_named_parameters(prefix, seen))

    def modules(self) -> list[Module]:
        result = [self]
        for child in self._children():
            result.extend(child.modules())
        return result

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

    def state_dict(self) -> dict[str, Tensor]:
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters()}

    def load_state_dict(self, state: dict[str, Tensor], strict: bool = True) -> None:
        current = dict(self.named_parameters())
        missing = [name for name in current if name not in state]
        unexpected = [name for name in state if name not in current]
        if strict and (missing or unexpected):
            raise ValueError(f"state_dict mismatch: missing={missing}, unexpected={unexpected}")
        for name, value in state.items():
            if name in current:
                current[name]._assign(value)

    def __repr__(self) -> str:
        children = list(self._named_children())
        if not children:
            return f"{self.__class__.__name__}()"
        body = ", ".join(f"{name}={module!r}" for name, module in children)
        return f"{self.__class__.__name__}({body})"

    def _children(self) -> Iterator[Module]:
        for _, module in self._named_children():
            yield module

    def _named_children(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        for name, value in self.__dict__.items():
            if name == "training":
                continue
            yield from self._modules_from_value(name, value, prefix)

    def _iter_parameters(self, seen: set[int]) -> Iterator[Tensor]:
        for _, parameter in self._iter_named_parameters("", seen):
            yield parameter

    def _iter_named_parameters(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
        for name, value in self.__dict__.items():
            yield from self._named_parameters_from_value(name, value, prefix, seen)

    @classmethod
    def _join_name(cls, prefix: str, name: str) -> str:
        if not prefix:
            return name
        if not name:
            return prefix
        return f"{prefix}.{name}"

    @classmethod
    def _named_parameters_from_value(
        cls,
        name: str,
        value: Any,
        prefix: str,
        seen: set[int],
    ) -> Iterator[tuple[str, Tensor]]:
        full_name = cls._join_name(prefix, name)
        if isinstance(value, Tensor):
            marker = id(value)
            if value.requires_grad and marker not in seen:
                seen.add(marker)
                yield full_name, value
        elif isinstance(value, Module):
            yield from value._iter_named_parameters(full_name, seen)
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from cls._named_parameters_from_value(str(key), item, full_name, seen)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from cls._named_parameters_from_value(str(index), item, full_name, seen)

    @classmethod
    def _modules_from_value(
        cls,
        name: str,
        value: Any,
        prefix: str,
    ) -> Iterator[tuple[str, Module]]:
        full_name = cls._join_name(prefix, name)
        if isinstance(value, Module):
            yield full_name, value
        elif isinstance(value, dict):
            for key, item in value.items():
                yield from cls._modules_from_value(str(key), item, full_name)
        elif isinstance(value, (list, tuple)):
            for index, item in enumerate(value):
                yield from cls._modules_from_value(str(index), item, full_name)


class Linear(Module):
    """Fully connected layer with shape ``input @ weight.T + bias``."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        scale = 1.0 / math.sqrt(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.weight = cast(Tensor, Parameter(randn((out_features, in_features)) * scale))
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
        self.layers = list(modules)

    def forward(self, input: Tensor) -> Tensor:
        output = input
        for module in self.layers:
            output = module(output)
        return output

    def __iter__(self) -> Iterator[Module]:
        return iter(self.layers)

    def __len__(self) -> int:
        return len(self.layers)

    def __getitem__(self, index: int) -> Module:
        return self.layers[index]

    def _named_children(self, prefix: str = "") -> Iterator[tuple[str, Module]]:
        for index, module in enumerate(self.layers):
            yield self._join_name(prefix, str(index)), module

    def _iter_named_parameters(self, prefix: str, seen: set[int]) -> Iterator[tuple[str, Tensor]]:
        for index, module in enumerate(self.layers):
            yield from module._iter_named_parameters(self._join_name(prefix, str(index)), seen)


class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.relu()


class Sigmoid(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.sigmoid()


class Tanh(Module):
    def forward(self, input: Tensor) -> Tensor:
        return input.tanh()


class Dropout(Module):
    """Inverted dropout using TensorStudio random tensors."""

    def __init__(self, p: float = 0.5, seed: int | None = None) -> None:
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError("dropout probability p must satisfy 0 <= p < 1")
        self.p = p
        self.seed = seed

    def forward(self, input: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return input
        mask = rand(input.shape, dtype=input.dtype, seed=self.seed) > self.p
        return input * mask / (1.0 - self.p)


class Flatten(Module):
    """Flatten a tensor from ``start_dim`` through the last dimension."""

    def __init__(self, start_dim: int = 1) -> None:
        super().__init__()
        self.start_dim = start_dim

    def forward(self, input: Tensor) -> Tensor:
        ndim = input.ndim
        start = self.start_dim if self.start_dim >= 0 else ndim + self.start_dim
        if start < 0 or start > ndim:
            raise ValueError("start_dim is out of range")
        prefix = input.shape[:start]
        flattened = math.prod(input.shape[start:]) if start < ndim else 1
        return input.reshape((*prefix, flattened))


__all__ = [
    "Dropout",
    "Flatten",
    "Linear",
    "Module",
    "Parameter",
    "ReLU",
    "Sequential",
    "Sigmoid",
    "Tanh",
]
