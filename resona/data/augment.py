"""Skeleton augmentation for RESONA-77 (T, 234) tensors.

Schema layout per frame (234 floats):
    [0   : 154]   77 joints × (x, y)         = 154
    [154 : 231]   77 joints × confidence     = 77
    [231 : 234]   3 part validity flags      = 3

NOTE: Joint layout is NOT (joint, [x, y, conf]) interleaved. (x, y) for all 77
joints come first, then 77 confidences, then 3 part flags. We split per-segment
when applying spatial transforms; only (x, y) block is mutated.
"""
from __future__ import annotations

import torch

N_JOINTS = 77
XY_END = N_JOINTS * 2          # 154
CONF_END = XY_END + N_JOINTS   # 231
TAIL_END = CONF_END + 3        # 234 (= part validity flags)


# ---------- low-level: split / merge ----------------------------------------

def _split(x: torch.Tensor):
    """(T, 234) -> (xy_view: T,77,2, conf: T,77, tail: T,3)."""
    T, D = x.shape
    if D != TAIL_END:
        raise ValueError(f"expected D={TAIL_END}, got {D}")
    xy = x[:, :XY_END].view(T, N_JOINTS, 2)
    conf = x[:, XY_END:CONF_END]
    tail = x[:, CONF_END:TAIL_END]
    return xy, conf, tail


def _merge(xy: torch.Tensor, conf: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """(T,77,2)+(T,77)+(T,3) -> (T, 234)."""
    T = xy.shape[0]
    flat_xy = xy.reshape(T, XY_END)
    return torch.cat([flat_xy, conf, tail], dim=-1)


# ---------- per-op transforms (operate on xy only) --------------------------

def spatial_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Gaussian noise on (x, y) only."""
    if sigma <= 0:
        return x
    xy, conf, tail = _split(x)
    xy = xy + torch.randn_like(xy) * sigma
    return _merge(xy, conf, tail)


def horizontal_flip(x: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
    """Mirror x-coordinate. Does NOT swap L/R joint indices (mild aug)."""
    if prob <= 0 or torch.rand(()) > prob:
        return x
    xy, conf, tail = _split(x)
    xy = xy.clone()
    xy[..., 0] = -xy[..., 0]
    return _merge(xy, conf, tail)


def temporal_crop(x: torch.Tensor, crop_len: int) -> torch.Tensor:
    """Random temporal crop. If T < crop_len, pad with zero frames at end."""
    T = x.shape[0]
    if T == crop_len:
        return x
    if T > crop_len:
        s = int(torch.randint(0, T - crop_len + 1, ()).item())
        return x[s : s + crop_len].contiguous()
    pad = x.new_zeros(crop_len - T, x.shape[1])
    return torch.cat([x, pad], dim=0)


def temporal_speed(x: torch.Tensor, ratio_range: tuple[float, float] = (0.8, 1.25)) -> torch.Tensor:
    """Linearly resample along time. xy and conf interpolated; tail nearest."""
    lo, hi = ratio_range
    if lo == hi == 1.0:
        return x
    r = float(torch.empty(()).uniform_(lo, hi).item())
    T = x.shape[0]
    new_T = max(8, int(round(T * r)))
    if new_T == T:
        return x

    xy, conf, tail = _split(x)
    # xy: (T, 77, 2) -> (1, 154, T) for 1D interpolate
    xy_t = xy.reshape(T, XY_END).permute(1, 0).unsqueeze(0)  # (1, 154, T)
    xy_new = torch.nn.functional.interpolate(xy_t, size=new_T, mode="linear", align_corners=False)
    xy_new = xy_new.squeeze(0).permute(1, 0).reshape(new_T, N_JOINTS, 2)

    conf_t = conf.permute(1, 0).unsqueeze(0)                 # (1, 77, T)
    conf_new = torch.nn.functional.interpolate(conf_t, size=new_T, mode="linear", align_corners=False)
    conf_new = conf_new.squeeze(0).permute(1, 0)             # (new_T, 77)

    tail_t = tail.permute(1, 0).unsqueeze(0)                 # (1, 3, T)
    tail_new = torch.nn.functional.interpolate(tail_t, size=new_T, mode="nearest")
    tail_new = tail_new.squeeze(0).permute(1, 0)             # (new_T, 3)

    return _merge(xy_new, conf_new, tail_new)


# ---------- compositional augmentor -----------------------------------------

class SkeletonAugment:
    """Default RESONA augmentation pipeline.

    Operates on (T, 234) tensors, mutating only the (x, y) block. Confidence
    and part-validity tail are interpolated only by speed perturbation
    (linear/nearest), preserved otherwise.
    """

    def __init__(
        self,
        spatial_sigma: float = 0.02,
        flip_prob: float = 0.5,
        speed_range: tuple[float, float] = (0.85, 1.15),
    ):
        self.sigma = spatial_sigma
        self.flip_prob = flip_prob
        self.speed_range = speed_range

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.speed_range != (1.0, 1.0):
            x = temporal_speed(x, self.speed_range)
        if self.flip_prob > 0:
            x = horizontal_flip(x, self.flip_prob)
        if self.sigma > 0:
            x = spatial_jitter(x, self.sigma)
        return x
