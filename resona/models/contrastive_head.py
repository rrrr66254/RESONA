"""Temporal contrastive head — BYOL-style EMA target with two clip views.

Two views = two random temporal crops of the same clip. Encoder + projector
produce online embeddings; an EMA target network produces target embeddings.
Loss: 2 - 2 * cos(online_pred, target).
"""
from __future__ import annotations

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    """Wraps an encoder with EMA target + projection/prediction heads."""

    def __init__(
        self,
        encoder: nn.Module,
        d_model: int = 1024,
        proj_dim: int = 256,
        ema_decay: float = 0.996,
    ):
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

    def _pool(self, h: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        if mask is None:
            return h.mean(dim=1)
        m = mask.float().unsqueeze(-1)
        return (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)

    def forward(
        self,
        x_online: torch.Tensor,
        x_target: torch.Tensor,
        m_online: torch.Tensor | None = None,
        m_target: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_o = self.online(x_online, m_online)
        z_o = self.proj_online(self._pool(h_o, m_online))
        p_o = self.predict(z_o)

        with torch.no_grad():
            h_t = self.target(x_target, m_target)
            z_t = self.proj_target(self._pool(h_t, m_target))

        loss = 2 - 2 * F.cosine_similarity(p_o, z_t.detach(), dim=-1).mean()
        return loss
