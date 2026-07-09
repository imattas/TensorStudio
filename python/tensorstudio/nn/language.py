"""Language-model-oriented layers and helpers."""

from __future__ import annotations

from dataclasses import dataclass

from tensorstudio.tensor import Tensor, tensor

from .functional import cross_entropy, embedding
from .modules import Embedding, Linear, Module


@dataclass
class Vocabulary:
    """Small deterministic token vocabulary."""

    token_to_id: dict[str, int]
    unknown_token: str = "<unk>"

    @classmethod
    def build(
        cls,
        texts: list[str],
        *,
        min_freq: int = 1,
        special_tokens: tuple[str, ...] = ("<pad>", "<unk>"),
    ) -> Vocabulary:
        counts: dict[str, int] = {}
        for text in texts:
            for token in text.split():
                counts[token] = counts.get(token, 0) + 1
        token_to_id = {token: index for index, token in enumerate(special_tokens)}
        for token in sorted(counts):
            if counts[token] >= min_freq and token not in token_to_id:
                token_to_id[token] = len(token_to_id)
        unknown = "<unk>" if "<unk>" in token_to_id else next(iter(token_to_id))
        return cls(token_to_id, unknown_token=unknown)

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        unknown_id = self.token_to_id[self.unknown_token]
        return [self.token_to_id.get(token, unknown_id) for token in text.split()]

    def decode(self, ids: list[int]) -> str:
        id_to_token = {index: token for token, index in self.token_to_id.items()}
        return " ".join(id_to_token.get(int(index), self.unknown_token) for index in ids)


class TokenEmbedding(Embedding):
    """Named alias for token embedding tables."""


class PositionalEmbedding(Module):
    """Learned position embeddings for rank-2 token id tensors."""

    def __init__(self, max_length: int, embedding_dim: int) -> None:
        super().__init__()
        if max_length <= 0 or embedding_dim <= 0:
            raise ValueError("max_length and embedding_dim must be positive")
        self.max_length = max_length
        self.embedding_dim = embedding_dim
        self.embedding = Embedding(max_length, embedding_dim)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("PositionalEmbedding expects input_ids with shape (batch, time)")
        length = input_ids.shape[1]
        if length > self.max_length:
            raise ValueError("input sequence is longer than max_length")
        positions = tensor(list(range(length)), dtype="int64")
        return embedding(positions, self.embedding.weight).reshape((1, length, self.embedding_dim))


class CausalLanguageModel(Module):
    """Tiny embedding-plus-projection causal language model."""

    def __init__(self, vocab_size: int, embedding_dim: int = 32, max_length: int = 128) -> None:
        super().__init__()
        if vocab_size <= 0 or embedding_dim <= 0 or max_length <= 0:
            raise ValueError("vocab_size, embedding_dim, and max_length must be positive")
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length = max_length
        self.token_embedding = TokenEmbedding(vocab_size, embedding_dim)
        self.position_embedding = PositionalEmbedding(max_length, embedding_dim)
        self.projection = Linear(embedding_dim, vocab_size)

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2:
            raise ValueError("CausalLanguageModel expects input_ids with shape (batch, time)")
        batch, length = input_ids.shape
        token_features = self.token_embedding(input_ids)
        position_features = self.position_embedding(input_ids)
        hidden = token_features + position_features
        logits = self.projection(hidden.reshape((batch * length, self.embedding_dim)))
        return logits.reshape((batch, length, self.vocab_size))


def make_causal_lm_batch(token_ids: list[int], sequence_length: int) -> tuple[Tensor, Tensor]:
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if len(token_ids) <= sequence_length:
        raise ValueError("token_ids must contain at least sequence_length + 1 entries")
    inputs: list[list[int]] = []
    targets: list[list[int]] = []
    for start in range(0, len(token_ids) - sequence_length):
        window = token_ids[start : start + sequence_length + 1]
        inputs.append(window[:-1])
        targets.append(window[1:])
    return tensor(inputs, dtype="int64"), tensor(targets, dtype="int64")


def causal_language_model_loss(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 3:
        raise ValueError("language-model logits must have shape (batch, time, vocab)")
    if targets.shape != logits.shape[:2]:
        raise ValueError("language-model targets must have shape (batch, time)")
    batch, length, vocab_size = logits.shape
    return cross_entropy(
        logits.reshape((batch * length, vocab_size)),
        targets.reshape((batch * length,)),
    )


__all__ = [
    "CausalLanguageModel",
    "PositionalEmbedding",
    "TokenEmbedding",
    "Vocabulary",
    "causal_language_model_loss",
    "make_causal_lm_batch",
]
