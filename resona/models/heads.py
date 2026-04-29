"""Pretraining heads + losses for RESONA encoder.

- MSMHead     : predicts (x, y) per joint
- ForwardPred : predicts h_{t+k} from h_t (k=16 default)
- ContrastiveProj : projection head for BYOL EMA contrastive

All heads operate on encoder output (B, T, d_model).
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- MSM ----------

N_JOINTS = 77
JOINT_BLOCK = N_JOINTS * 3  # 231 (joints) + 3 (tail validity) = 234 total
TAIL = 3


class MSMHead(nn.Module):
    """Predicts (x, y) per joint from encoder hidden state."""

    def __init__(self, d_model: int, n_joints: int = N_JOINTS):
        super().__init__()
        self.n_joints = n_joints
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_joints * 2),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        B, T, _ = h.shape
        return self.proj(h).view(B, T, self.n_joints, 2)


def make_joint_mask(x: torch.Tensor, n_joints: int = N_JOINTS, mask_ratio: float = 0.5) -> torch.Tensor:
    B, T, D = x.shape
    expected = n_joints * 3 + TAIL
    assert D == expected, f"expected D={expected}, got D={D}"
    return torch.rand(B, T, n_joints, device=x.device) < mask_ratio


def apply_mask(x: torch.Tensor, mask: torch.Tensor, n_joints: int = N_JOINTS) -> torch.Tensor:
    """Zero out (x, y, conf) of masked joints. Tail validity untouched."""
    B, T, _ = x.shape
    out = x.clone()
    j = out[:, :, : n_joints * 3].view(B, T, n_joints, 3)
    j[mask] = 0.0
    out[:, :, : n_joints * 3] = j.reshape(B, T, n_joints * 3)
    return out


def msm_loss(
    pred: torch.Tensor,        # (B, T, n_joints, 2)
    target: torch.Tensor,      # (B, T, 234)
    mask: torch.Tensor,        # (B, T, n_joints) bool
    valid_mask: torch.Tensor | None = None,
    n_joints: int = N_JOINTS,
) -> torch.Tensor:
    B, T, _ = target.shape
    j = target[:, :, : n_joints * 3].view(B, T, n_joints, 3)
    xy_tgt = j[..., :2]
    conf = j[..., 2] > 0.5
    sel = mask & conf
    if valid_mask is not None:
        sel = sel & valid_mask.unsqueeze(-1)
    if sel.sum() == 0:
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred[sel], xy_tgt[sel], reduction="mean")


# ---------- Forward Prediction ----------

class ForwardPredHead(nn.Module):
    """Predicts h_{t+k} from h_t (causal feature regression)."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.proj(h)


def forward_pred_loss(
    pred: torch.Tensor,        # (B, T, d) — predicted h_{t+k}
    target_h: torch.Tensor,    # (B, T, d) — actual h_t (we shift by k)
    k: int = 16,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Smooth-L1 between pred[:, :-k] and target_h[:, k:]."""
    B, T, _ = pred.shape
    if T <= k:
        return pred.new_zeros(())
    p = pred[:, :-k]
    t = target_h[:, k:].detach()                              # stop-grad
    if valid_mask is not None:
        m = valid_mask[:, :-k] & valid_mask[:, k:]
        if m.sum() == 0:
            return pred.new_zeros(())
        return F.smooth_l1_loss(p[m], t[m], reduction="mean")
    return F.smooth_l1_loss(p, t, reduction="mean")


# ---------- Contrastive (BYOL EMA) ----------

class ProjectionHead(nn.Module):
    def __init__(self, d_in: int, d_out: int = 256, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionHead(nn.Module):
    def __init__(self, d: int = 256, hidden: int = 2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, hidden),
            nn.BatchNorm1d(hidden),
            nn.GELU(),
            nn.Linear(hidden, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TemporalContrastive(nn.Module):
    """BYOL-style EMA target with two clip views.

    Online encoder → projection → prediction
    Target (EMA) encoder → projection
    Loss: 2 - 2 cos(p_online, z_target).
    """

    def __init__(self, encoder: nn.Module, d_model: int, proj_dim: int = 256, ema_decay: float = 0.996):
        super().__init__()
        self.online = encoder
        self.target = copy.deepcopy(encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.proj_online = ProjectionHead(d_model, proj_dim)
        self.proj_target = ProjectionHead(d_model, proj_dim)
        for p in self.proj_target.parameters():
            p.requires_grad_(False)
        self.predict = PredictionHead(proj_dim)
        self.ema_decay = ema_decay

    @torch.no_grad()
    def update_target(self) -> None:
        d = self.ema_decay
        for po, pt in zip(self.online.parameters(), self.target.parameters()):
            pt.data.mul_(d).add_(po.data, alpha=1 - d)
        for po, pt in zip(self.proj_online.parameters(), self.proj_target.parameters()):
            pt.data.mul_(d).add_(po.data, alpha=1 - d)

    def _pool(self, h: torch.Tensor, valid_mask: torch.Tensor | None) -> torch.Tensor:
        if valid_mask is None:
            return h.mean(dim=1)
        m = valid_mask.float().unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(
        self,
        x_o: torch.Tensor, x_t: torch.Tensor,
        m_o: torch.Tensor | None = None, m_t: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_o = self.online(x_o, m_o)
        z_o = self.proj_online(self._pool(h_o, m_o))
        p_o = self.predict(z_o)
        with torch.no_grad():
            h_t = self.target(x_t, m_t)
            z_t = self.proj_target(self._pool(h_t, m_t))
        return 2 - 2 * F.cosine_similarity(p_o, z_t.detach(), dim=-1).mean()
