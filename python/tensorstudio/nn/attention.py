"""Small attention and transformer blocks."""

from __future__ import annotations

import math

from tensorstudio.ops import softmax, where
from tensorstudio.tensor import Tensor, tensor

from .modules import GELU, LayerNorm, Linear, Module


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    mask: Tensor | None = None,
    causal: bool = False,
) -> Tensor:
    """Compute scaled dot-product attention for rank-4 tensors."""

    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention expects query, key, and value with rank 4")
    batch, heads, query_time, features = query.shape
    key_batch, key_heads, key_time, key_features = key.shape
    value_batch, value_heads, value_time, value_features = value.shape
    if (key_batch, key_heads) != (batch, heads) or (value_batch, value_heads) != (
        batch,
        heads,
    ):
        raise ValueError("attention batch/head dimensions must match")
    if features != key_features or key_time != value_time:
        raise ValueError("attention feature and source-time dimensions must match")

    flat_batch = batch * heads
    q = query.clone().reshape((flat_batch, query_time, features))
    k = key.clone().reshape((flat_batch, key_time, key_features))
    v = value.clone().reshape((flat_batch, value_time, value_features))
    scores = (q @ k.transpose(1, 2)) / math.sqrt(float(features))

    if causal:
        causal_mask = _causal_mask(query_time, key_time, batch, heads)
        scores = where(causal_mask.reshape((flat_batch, query_time, key_time)), scores, -1.0e9)
    if mask is not None:
        if mask.shape != (batch, heads, query_time, key_time):
            raise ValueError(
                "attention mask must have shape "
                f"{(batch, heads, query_time, key_time)}, got {mask.shape}"
            )
        scores = where(mask.reshape((flat_batch, query_time, key_time)), scores, -1.0e9)

    weights = softmax(scores, axis=-1)
    output = weights @ v
    return output.reshape((batch, heads, query_time, value_features))


class MultiHeadSelfAttention(Module):
    """Compact multi-head self-attention module for small CPU models."""

    def __init__(self, embed_dim: int, num_heads: int, *, bias: bool = True) -> None:
        super().__init__()
        if embed_dim <= 0 or num_heads <= 0:
            raise ValueError("embed_dim and num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.q_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, input: Tensor, *, mask: Tensor | None = None, causal: bool = False) -> Tensor:
        if input.ndim != 3:
            raise ValueError("MultiHeadSelfAttention expects input shape (batch, time, embed_dim)")
        batch, length, embed_dim = input.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"expected embed_dim={self.embed_dim}, got {embed_dim}")

        flat = input.reshape((batch * length, embed_dim))
        query = self._split_heads(self.q_proj(flat).reshape((batch, length, embed_dim)))
        key = self._split_heads(self.k_proj(flat).reshape((batch, length, embed_dim)))
        value = self._split_heads(self.v_proj(flat).reshape((batch, length, embed_dim)))
        attended = scaled_dot_product_attention(query, key, value, mask=mask, causal=causal)
        merged = attended.permute(0, 2, 1, 3).clone().reshape((batch * length, embed_dim))
        return self.out_proj(merged).reshape((batch, length, embed_dim))

    def _split_heads(self, input: Tensor) -> Tensor:
        batch, length, _ = input.shape
        return input.reshape((batch, length, self.num_heads, self.head_dim)).permute(0, 2, 1, 3)

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"


class TransformerEncoderBlock(Module):
    """Small pre-norm Transformer encoder block."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        mlp_ratio: float = 4.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        hidden_dim = max(1, int(round(embed_dim * mlp_ratio)))
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.norm1 = LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, bias=bias)
        self.norm2 = LayerNorm(embed_dim)
        self.fc1 = Linear(embed_dim, hidden_dim, bias=bias)
        self.activation = GELU()
        self.fc2 = Linear(hidden_dim, embed_dim, bias=bias)

    def forward(self, input: Tensor, *, mask: Tensor | None = None, causal: bool = False) -> Tensor:
        if input.ndim != 3:
            raise ValueError("TransformerEncoderBlock expects input shape (batch, time, embed_dim)")
        batch, length, embed_dim = input.shape
        attended = self.attention(self.norm1(input), mask=mask, causal=causal)
        hidden = input + attended
        feedforward_input = self.norm2(hidden).reshape((batch * length, embed_dim))
        feedforward = self.fc2(self.activation(self.fc1(feedforward_input))).reshape(
            (batch, length, embed_dim)
        )
        return hidden + feedforward

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, num_heads={self.num_heads}"


def _causal_mask(query_time: int, key_time: int, batch: int, heads: int) -> Tensor:
    mask = [
        [
            [
                [key_index <= query_index for key_index in range(key_time)]
                for query_index in range(query_time)
            ]
            for _ in range(heads)
        ]
        for _ in range(batch)
    ]
    return tensor(mask, dtype="bool")


__all__ = [
    "MultiHeadSelfAttention",
    "TransformerEncoderBlock",
    "scaled_dot_product_attention",
]
