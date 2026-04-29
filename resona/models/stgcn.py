"""ST-GCN encoder for RESONA sub-pose blocks.

Each sub-pose (body/lhand/rhand/mouth/face) is encoded by:
  - 3 spatial GCN layers (channels: in_ch -> 64 -> 128 -> 256)
  - 3 temporal Conv1d layers (kernel=5, channel=256)

Output: (B, T, 256) per group (after spatial mean pool over joints).

Per-group params: ~1.5M. 4 (or 5) groups: ~6-8M total. Weight not shared.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """A_hat · X · W (single graph conv with normalized adjacency).

    Args:
        in_ch / out_ch: feature dims
        adj: (V, V) normalized adjacency, persistent buffer
    """

    def __init__(self, in_ch: int, out_ch: int, adj: torch.Tensor):
        super().__init__()
        self.lin = nn.Linear(in_ch, out_ch)
        self.register_buffer("adj", adj)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C_in)
        x = self.lin(x)                                          # (B, T, V, C_out)
        x = torch.einsum("vu,btuc->btvc", self.adj, x)          # graph conv
        # BN over channels: flatten (B*T*V) batch dim
        B, T, V, C = x.shape
        x = self.bn(x.reshape(B * T * V, C)).reshape(B, T, V, C)
        return F.relu(x)


class TemporalConv(nn.Module):
    """1D temporal conv along T axis, per joint."""

    def __init__(self, ch: int, kernel: int = 5):
        super().__init__()
        pad = (kernel - 1) // 2
        self.conv = nn.Conv2d(ch, ch, kernel_size=(kernel, 1), padding=(pad, 0))
        self.bn = nn.BatchNorm2d(ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C) -> (B, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.bn(self.conv(x))
        x = x.permute(0, 2, 3, 1)
        return F.relu(x)


class SubPoseEncoder(nn.Module):
    """3-layer spatial GCN + 3-layer temporal conv for one sub-pose.

    in_ch: per-joint input channels (3 for [x, y, conf])
    out_ch: final dim (256 default)
    """

    def __init__(self, n_joints: int, adj: torch.Tensor, in_ch: int = 3, out_ch: int = 256):
        super().__init__()
        self.n_joints = n_joints
        # Spatial GCN: in_ch -> 64 -> 128 -> 256
        chs = [in_ch, 64, 128, out_ch]
        self.spatial = nn.ModuleList([
            GraphConv(chs[i], chs[i + 1], adj) for i in range(3)
        ])
        # Temporal: 3 layers @ out_ch
        self.temporal = nn.ModuleList([
            TemporalConv(out_ch, kernel=5) for _ in range(3)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, V, C_in)
        for layer in self.spatial:
            x = layer(x)
        for layer in self.temporal:
            x = layer(x)
        # Mean pool over joints -> (B, T, C)
        return x.mean(dim=2)


class MultiSubPoseEncoder(nn.Module):
    """4-group sub-pose encoder for RESONA-77.

    Reads (B, T, 234) → splits into 4 sub-poses (body9, lhand21, rhand21,
    mouth8+face18) → encodes each with its own SubPoseEncoder → concats to
    (B, T, 4*out_ch).

    Note: RESONA-77 layout is (joint, [x, y, conf]) interleaved for first
    231 dims, then 3 part-validity tail. We use only the joint block.
    """

    def __init__(self, out_ch: int = 256):
        super().__init__()
        from .graph import body9_adj, hand21_adj, mouth8_adj, face18_adj
        self.out_ch = out_ch
        # Adjacency matrices as buffers
        self.body = SubPoseEncoder(9, body9_adj(), in_ch=3, out_ch=out_ch)
        self.lhand = SubPoseEncoder(21, hand21_adj(), in_ch=3, out_ch=out_ch)
        self.rhand = SubPoseEncoder(21, hand21_adj(), in_ch=3, out_ch=out_ch)
        # mface = mouth8 + face18 = 26 joints; build adjacency by block-diag
        self.mface = SubPoseEncoder(26, self._mface_adj(), in_ch=3, out_ch=out_ch)
        # Joint slices in the 231-dim joint block (per-joint 3-channel layout)
        # body 0-8, lhand 9-29, rhand 30-50, mouth 51-58, face 59-76
        self.slices = {
            "body":  slice(0, 9),
            "lhand": slice(9, 30),
            "rhand": slice(30, 51),
            "mface": slice(51, 77),
        }

    @staticmethod
    def _mface_adj() -> torch.Tensor:
        """Block-diagonal of mouth8 + face18 (no cross edges)."""
        from .graph import mouth8_adj, face18_adj
        a_m = mouth8_adj()
        a_f = face18_adj()
        n = 26
        adj = torch.zeros(n, n)
        adj[:8, :8] = a_m
        adj[8:, 8:] = a_f
        return adj

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, 234) — first 231 = (77, 3) joint block, last 3 = part validity (ignored here)
        B, T, D = x.shape
        joints = x[:, :, : 77 * 3].view(B, T, 77, 3)         # (B, T, 77, 3)

        feats = []
        for name, enc in [("body", self.body), ("lhand", self.lhand),
                          ("rhand", self.rhand), ("mface", self.mface)]:
            sl = self.slices[name]
            xs = joints[:, :, sl, :]                          # (B, T, n_g, 3)
            feats.append(enc(xs))                              # (B, T, out_ch)
        return torch.cat(feats, dim=-1)                       # (B, T, 4*out_ch)
