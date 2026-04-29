"""Skeleton augmentation for RESONA-77 (T, 234) tensors.

Schema layout per frame (234 floats):
    [0   : 231]   77 joints × (x, y, conf) interleaved   (T, 77, 3)
    [231 : 234]   3 part validity flags                   (T, 3)

Only the joint block (T, 77, 3) is interpreted as per-joint, with last channel
being confidence. Augmentation:
- spatial_jitter: noise on (x, y) only, conf untouched
- horizontal_flip: flip x sign (no L/R swap), conf untouched
- temporal_speed: interpolate joints linearly, tail nearest
"""
from __future__ import annotations

import torch

N_JOINTS = 77
JOINT_BLOCK = N_JOINTS * 3   # 231
TAIL = 3
TOTAL = JOINT_BLOCK + TAIL   # 234


# ---------- low-level: split / merge ----------------------------------------

def _split(x: torch.Tensor):
    """(T, 234) -> (j: T,77,3, tail: T,3)."""
    T, D = x.shape
    if D != TOTAL:
        raise ValueError(f"expected D={TOTAL}, got {D}")
    j = x[:, :JOINT_BLOCK].view(T, N_JOINTS, 3)
    tail = x[:, JOINT_BLOCK:]
    return j, tail


def _merge(j: torch.Tensor, tail: torch.Tensor) -> torch.Tensor:
    """(T,77,3) + (T,3) -> (T, 234)."""
    T, J, _ = j.shape
    return torch.cat([j.reshape(T, J * 3), tail], dim=-1)


# ---------- per-op transforms (operate on (x,y) inside joint block) --------

def spatial_jitter(x: torch.Tensor, sigma: float = 0.02) -> torch.Tensor:
    """Gaussian noise on (x, y); confidence and tail untouched."""
    if sigma <= 0:
        return x
    j, tail = _split(x)
    j = j.clone()
    j[..., :2] = j[..., :2] + torch.randn_like(j[..., :2]) * sigma
    return _merge(j, tail)


def horizontal_flip(x: torch.Tensor, prob: float = 0.5) -> torch.Tensor:
    """Mirror x-coordinate. Does NOT swap L/R joint indices (mild aug)."""
    if prob <= 0 or torch.rand(()) > prob:
        return x
    j, tail = _split(x)
    j = j.clone()
    j[..., 0] = -j[..., 0]
    return _merge(j, tail)


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
    """Linearly resample along time. (xy, conf) interp linear; tail nearest."""
    lo, hi = ratio_range
    if lo == hi == 1.0:
        return x
    r = float(torch.empty(()).uniform_(lo, hi).item())
    T = x.shape[0]
    new_T = max(8, int(round(T * r)))
    if new_T == T:
        return x

    j, tail = _split(x)
    # j: (T, 77, 3) -> (1, 77*3=231, T) for 1D interpolate
    j_t = j.reshape(T, JOINT_BLOCK).permute(1, 0).unsqueeze(0)
    j_new = torch.nn.functional.interpolate(j_t, size=new_T, mode="linear", align_corners=False)
    j_new = j_new.squeeze(0).permute(1, 0).reshape(new_T, N_JOINTS, 3)

    tail_t = tail.permute(1, 0).unsqueeze(0)                 # (1, 3, T)
    tail_new = torch.nn.functional.interpolate(tail_t, size=new_T, mode="nearest")
    tail_new = tail_new.squeeze(0).permute(1, 0)             # (new_T, 3)

    return _merge(j_new, tail_new)


# ---------- compositional augmentor -----------------------------------------

class SkeletonAugment:
    """Default RESONA augmentation pipeline.

    Operates on (T, 234) tensors. Joint (x, y) is mutated by jitter/flip;
    confidence is preserved; tail validity is preserved (only resampled by
    speed perturb via nearest-neighbor).
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
