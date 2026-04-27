"""Joint-tokenized Transformer encoder for RESONA.

Input  : (B, T, 234)   — 77 joints × (x, y, validity) flat per frame
Output : (B, T, d_model)
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrameEmbedding(nn.Module):
    """Linear projection (234 -> d_model) + learned positional embedding."""

    def __init__(self, in_dim: int = 234, d_model: int = 1024, max_len: int = 512):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)
        nn.init.trunc_normal_(self.pos.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,T,234) -> (B,T,d)
        B, T, _ = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        return self.norm(self.proj(x) + self.pos(pos))


class TransformerEncoder(nn.Module):
    """Standard pre-LN Transformer encoder. Uses SDPA (or flash-attn if installed)."""

    def __init__(
        self,
        in_dim: int = 234,
        d_model: int = 1024,
        n_heads: int = 16,
        n_layers: int = 24,
        ffn_dim: int = 4096,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.embed = FrameEmbedding(in_dim, d_model, max_len)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.d_model = d_model

    def forward(
        self,
        x: torch.Tensor,                      # (B, T, in_dim)
        valid_mask: torch.Tensor | None = None,  # (B, T) bool, True = valid
    ) -> torch.Tensor:
        h = self.embed(x)
        key_padding_mask = None if valid_mask is None else ~valid_mask
        return self.encoder(h, src_key_padding_mask=key_padding_mask)
