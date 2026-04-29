"""Modern Transformer encoder for RESONA: RoPE + RMSNorm + SwiGLU.

Input pipeline:
    (B, T, 234)
        → MultiSubPoseEncoder (4 sub-pose ST-GCN, weight-not-shared)
        → (B, T, 4 × 256 = 1024)
        → Linear projection → (B, T, d_model)
        → 18× TransformerBlock (RMSNorm pre-norm, RoPE, SwiGLU FFN)
        → (B, T, d_model)

Total params (default d=768, L=18, FFN=2048):
    sub-pose ST-GCN ~6M + projection ~1M + transformer ~95M = ~100M.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stgcn import MultiSubPoseEncoder


# -------------- RMSNorm --------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., dim)
        norm = x.float().pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm.to(x.dtype)) * self.weight


# -------------- RoPE -----------------------------------------------------

def precompute_rope_cache(dim_head: int, max_seq: int, base: float = 10000.0) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (cos, sin) of shape (max_seq, dim_head)."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim_head, 2).float() / dim_head))
    t = torch.arange(max_seq).float()
    freqs = torch.einsum("i,j->ij", t, inv_freq)               # (T, dim/2)
    cos = freqs.cos()
    sin = freqs.sin()
    # Repeat each freq twice so we can pair (x_2k, x_2k+1)
    cos = torch.cat([cos, cos], dim=-1)                        # (T, dim)
    sin = torch.cat([sin, sin], dim=-1)
    return cos, sin


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: (B, H, T, D_head)
    cos = cos.unsqueeze(0).unsqueeze(0)                        # (1,1,T,D_head)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = q * cos + _rotate_half(q) * sin
    k_out = k * cos + _rotate_half(k) * sin
    return q_out, k_out


# -------------- Attention with RoPE --------------------------------------

class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, d_model * 3, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, d_model); key_padding_mask: (B, T) True=pad
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)                        # (3, B, H, T, D_head)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, k, cos, sin)

        # SDPA — also handles flash-attn auto-fallback if installed
        attn_mask = None
        if key_padding_mask is not None:
            # SDPA expects (B, 1, 1, T) bool mask of POSITIONS TO MASK (True=mask)
            attn_mask = key_padding_mask[:, None, None, :]
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=~attn_mask if attn_mask is not None else None,
            dropout_p=self.dropout if self.training else 0.0,
        )
        out = out.transpose(1, 2).reshape(B, T, D)              # (B, T, D)
        return self.proj(out)


# -------------- SwiGLU FFN -----------------------------------------------

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w_gate = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_up = nn.Linear(d_model, ffn_dim, bias=False)
        self.w_down = nn.Linear(ffn_dim, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(self.dropout(F.silu(self.w_gate(x)) * self.w_up(x)))


# -------------- Block ----------------------------------------------------

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = Attention(d_model, n_heads, dropout=dropout)
        self.norm2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, ffn_dim, dropout=dropout)

    def forward(self, x, cos, sin, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), cos, sin, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# -------------- Top-level encoder ----------------------------------------

class TransformerEncoder(nn.Module):
    """Sub-pose ST-GCN + modern Temporal Transformer.

    Input  : (B, T, 234)
    Output : (B, T, d_model)
    """

    def __init__(
        self,
        in_dim: int = 234,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 18,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
        max_len: int = 512,
        subpose_ch: int = 256,
    ):
        super().__init__()
        del in_dim                                                  # not used (sub-pose splits 234→joints)
        self.d_model = d_model

        self.spatial = MultiSubPoseEncoder(out_ch=subpose_ch)        # output (B, T, 4*subpose_ch)
        self.proj = nn.Sequential(
            nn.Linear(4 * subpose_ch, d_model),
            RMSNorm(d_model),
        )

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, ffn_dim, dropout) for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

        # RoPE cache
        d_head = d_model // n_heads
        cos, sin = precompute_rope_cache(d_head, max_len)
        self.register_buffer("rope_cos", cos)
        self.register_buffer("rope_sin", sin)

    def forward(self, x: torch.Tensor, valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # x: (B, T, 234); valid_mask: (B, T) True=valid
        B, T, _ = x.shape
        h = self.spatial(x)                                          # (B, T, 4*256)
        h = self.proj(h)                                             # (B, T, d_model)

        cos = self.rope_cos[:T].to(h.dtype)
        sin = self.rope_sin[:T].to(h.dtype)

        key_padding_mask = None if valid_mask is None else ~valid_mask
        for block in self.blocks:
            h = block(h, cos, sin, key_padding_mask)
        return self.norm(h)
