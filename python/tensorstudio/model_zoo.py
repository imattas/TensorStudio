"""Small reproducible model zoo factories."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from . import nn

ModelFactory = Callable[..., nn.Module]


@dataclass(frozen=True)
class ModelCard:
    """Metadata for a registered model factory."""

    name: str
    task: str
    description: str
    default_kwargs: dict[str, Any]
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": self.task,
            "description": self.description,
            "default_kwargs": dict(self.default_kwargs),
            "tags": list(self.tags),
        }


_MODELS: dict[str, tuple[ModelCard, ModelFactory]] = {}


def register_model(card: ModelCard, factory: ModelFactory, *, overwrite: bool = False) -> None:
    if not overwrite and card.name in _MODELS:
        raise ValueError(f"model {card.name!r} is already registered")
    _MODELS[card.name] = (card, factory)


def list_models(*, task: str | None = None) -> list[str]:
    names = sorted(_MODELS)
    if task is None:
        return names
    return [name for name in names if _MODELS[name][0].task == task]


def model_info(name: str) -> dict[str, Any]:
    if name not in _MODELS:
        raise KeyError(f"unknown TensorStudio model: {name}")
    return _MODELS[name][0].to_dict()


def create_model(name: str, **kwargs: Any) -> nn.Module:
    if name not in _MODELS:
        raise KeyError(f"unknown TensorStudio model: {name}")
    card, factory = _MODELS[name]
    options = dict(card.default_kwargs)
    options.update(kwargs)
    return factory(**options)


def tiny_mlp(
    input_dim: int = 2,
    hidden_dim: int = 8,
    output_dim: int = 1,
) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, output_dim),
    )


def tiny_cnn(
    in_channels: int = 1,
    num_classes: int = 10,
    hidden_channels: int = 8,
) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
        nn.GELU(),
        nn.GlobalAvgPool2d(),
        nn.Flatten(),
        nn.Linear(hidden_channels, num_classes),
    )


def tiny_bigram_language_model(
    vocab_size: int = 16,
    embedding_dim: int = 8,
    max_length: int = 32,
) -> nn.Module:
    from .nn.language import CausalLanguageModel

    return CausalLanguageModel(vocab_size, embedding_dim=embedding_dim, max_length=max_length)


def _register_defaults() -> None:
    register_model(
        ModelCard(
            "tiny_mlp",
            "regression",
            "Two-layer MLP for tiny tabular or regression examples.",
            {"input_dim": 2, "hidden_dim": 8, "output_dim": 1},
            ("mlp", "example"),
        ),
        tiny_mlp,
        overwrite=True,
    )
    register_model(
        ModelCard(
            "tiny_cnn",
            "vision",
            "Compact convolutional classifier for small image examples.",
            {"in_channels": 1, "num_classes": 10, "hidden_channels": 8},
            ("vision", "cnn", "example"),
        ),
        tiny_cnn,
        overwrite=True,
    )
    register_model(
        ModelCard(
            "tiny_bigram_language_model",
            "language-modeling",
            "Tiny token model for smoke-testing language-model training loops.",
            {"vocab_size": 16, "embedding_dim": 8, "max_length": 32},
            ("language", "example"),
        ),
        tiny_bigram_language_model,
        overwrite=True,
    )


_register_defaults()


__all__ = [
    "ModelCard",
    "create_model",
    "list_models",
    "model_info",
    "register_model",
    "tiny_bigram_language_model",
    "tiny_cnn",
    "tiny_mlp",
]
