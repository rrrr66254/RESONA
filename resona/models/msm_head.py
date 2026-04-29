"""Masked Skeleton Modeling (MSM) head + masking utilities.

Schema (per frame): 234 floats = first 231 as (77 joints × [x, y, conf]) + 3 part validity tail.

Strategy: random joint-level mask. Each (frame, joint) pair masked independently
with probability = mask_ratio. Reconstruct (x, y) of masked joints with Smooth-L1
loss; confidence channel ignored in loss; tail validity used as gating signal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


N_JOINTS_DEFAULT = 77
JOINT_BLOCK = N_JOINTS_DEFAULT * 3   # 231
TAIL = 3
TOTAL = JOINT_BLOCK + TAIL           # 234


def make_joint_mask(
    x: torch.Tensor,          # (B, T, 234)
    n_joints: int = N_JOINTS_DEFAULT,
    mask_ratio: float = 0.5,
) -> torch.Tensor:
    """Returns (B, T, n_joints) bool — True = masked.

    Note: D is verified to be (n_joints*3 + tail). For RESONA-77 schema,
    D=234 = 77*3 + 3.
    """
    B, T, D = x.shape
    expected = n_joints * 3 + TAIL
    assert D == expected, f"expected D={expected}, got D={D}"
    return torch.rand(B, T, n_joints, device=x.device) < mask_ratio


def apply_mask(x: torch.Tensor, mask: torch.Tensor, n_joints: int = N_JOINTS_DEFAULT) -> torch.Tensor:
    """Zero out (x, y, conf) of masked joints. Tail validity untouched.

    Args:
        x: (B, T, 234) input
        mask: (B, T, n_joints) bool, True = mask this joint
    Returns:
        (B, T, 234) with masked joints zeroed
    """
    B, T, D = x.shape
    out = x.clone()
    j = out[:, :, : n_joints * 3].view(B, T, n_joints, 3)
    j[mask] = 0.0
    out[:, :, : n_joints * 3] = j.reshape(B, T, n_joints * 3)
    # tail (out[:, :, n_joints*3:]) preserved as-is
    return out


class MSMHead(nn.Module):
    """Predicts (x, y) for every joint from encoder output."""

    def __init__(self, d_model: int = 1024, n_joints: int = N_JOINTS_DEFAULT):
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
    n_joints: int = N_JOINTS_DEFAULT,
) -> torch.Tensor:
    B, T, D = target.shape
    # Joint block (T, 77, 3): [x, y, conf]
    j = target[:, :, : n_joints * 3].view(B, T, n_joints, 3)
    xy_tgt = j[..., :2]              # (B,T,n_joints,2)
    joint_valid = j[..., 2] > 0.5    # (B,T,n_joints) — confidence threshold as proxy validity

    sel = mask & joint_valid
    if valid_mask is not None:
        sel = sel & valid_mask.unsqueeze(-1)
    if sel.sum() == 0:
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[sel], xy_tgt[sel], reduction="mean")
