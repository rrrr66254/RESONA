"""Masked Skeleton Modeling (MSM) head + masking utilities.

Strategy: random *joint*-level mask. Each (frame, joint) pair masked
independently with probability = mask_ratio. Reconstruct masked positions
with Smooth-L1 over (x, y) only (validity flag excluded from loss).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_joint_mask(
    x: torch.Tensor,          # (B, T, 234)
    n_joints: int = 77,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """Returns (B, T, n_joints) bool — True = masked."""
    B, T, D = x.shape
    assert D == n_joints * 3, f"expected {n_joints * 3} dims, got {D}"
    return torch.rand(B, T, n_joints, device=x.device) < mask_ratio


def apply_mask(x: torch.Tensor, mask: torch.Tensor, n_joints: int = 77) -> torch.Tensor:
    """Zero out (x, y, validity) of masked joints. Returns (B, T, 234)."""
    B, T, D = x.shape
    x_view = x.view(B, T, n_joints, 3)
    out = x_view.clone()
    out[mask] = 0.0
    return out.view(B, T, D)


class MSMHead(nn.Module):
    """Predicts (x, y) for every joint from encoder output."""

    def __init__(self, d_model: int = 1024, n_joints: int = 77):
        super().__init__()
        self.n_joints = n_joints
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_joints * 2),     # x, y per joint
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:  # (B,T,d) -> (B,T,n_joints,2)
        B, T, _ = h.shape
        return self.proj(h).view(B, T, self.n_joints, 2)


def msm_loss(
    pred: torch.Tensor,        # (B, T, n_joints, 2)
    target: torch.Tensor,      # (B, T, 234)
    mask: torch.Tensor,        # (B, T, n_joints) — True = masked, predict here
    valid_mask: torch.Tensor | None = None,  # (B, T) — True = real frame (not pad)
    n_joints: int = 77,
) -> torch.Tensor:
    B, T, _ = target.shape
    tgt = target.view(B, T, n_joints, 3)
    xy_tgt = tgt[..., :2]            # (B,T,n_joints,2)
    joint_valid = tgt[..., 2] > 0.5  # (B,T,n_joints) — only count joints flagged valid

    sel = mask & joint_valid
    if valid_mask is not None:
        sel = sel & valid_mask.unsqueeze(-1)
    if sel.sum() == 0:
        return pred.new_zeros(())
    diff = F.smooth_l1_loss(pred[sel], xy_tgt[sel], reduction="mean")
    return diff
