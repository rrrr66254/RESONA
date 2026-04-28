"""Skeleton augmentation for RESONA-77 (T, 234) tensors.

Each sample is a flat (T, 77*3) tensor: [x_0, y_0, v_0, x_1, y_1, v_1, ...].
Validity flags (v_i) are NEVER mutated by augmentation — only (x, y).
"""
from __future__ import annotations

import torch


# ---------- low-level ops on (T, J, 3) view ---------------------------------

def _to_jview(x: torch.Tensor, n_joints: int = 77) -> torch.Tensor:
    """(T, 234) -> (T, J, 3)"""
    T, D = x.shape
    return x.view(T, n_joints, 3)


def _from_jview(x: torch.Tensor) -> torch.Tensor:
    """(T, J, 3) -> (T, J*3)"""
    T, J, _ = x.shape
    return x.reshape(T, J * 3)


def spatial_jitter(x: torch.Tensor, sigma: float = 0.02, n_joints: int = 77) -> torch.Tensor:
    """Add Gaussian noise to (x, y), leaving validity untouched."""
    if sigma <= 0:
        return x
    v = _to_jview(x, n_joints).clone()
    noise = torch.randn_like(v[..., :2]) * sigma
    v[..., :2] = v[..., :2] + noise
    return _from_jview(v)


def horizontal_flip(x: torch.Tensor, n_joints: int = 77, prob: float = 0.5) -> torch.Tensor:
    """Mirror x-coordinate around 0 (assumes coords already centered).

    Note: this does NOT swap left/right joint indices. For RESONA-77 schema,
    L-hand and R-hand occupy distinct slots, so a true left-right flip would
    require an index permutation. Until that LUT is provided, we only flip
    the x sign — which is a milder augmentation but still useful.
    """
    if prob <= 0 or torch.rand(()) > prob:
        return x
    v = _to_jview(x, n_joints).clone()
    v[..., 0] = -v[..., 0]
    return _from_jview(v)


def temporal_crop(x: torch.Tensor, crop_len: int) -> torch.Tensor:
    """Random temporal crop. If T < crop_len, pad with zero frames at the end."""
    T = x.shape[0]
    if T == crop_len:
        return x
    if T > crop_len:
        s = int(torch.randint(0, T - crop_len + 1, ()).item())
        return x[s : s + crop_len].contiguous()
    pad = x.new_zeros(crop_len - T, x.shape[1])
    return torch.cat([x, pad], dim=0)


def temporal_speed(x: torch.Tensor, ratio_range: tuple[float, float] = (0.8, 1.25)) -> torch.Tensor:
    """Linearly resample along time by a random factor — equivalent to speed
    perturbation. Validity flags are nearest-neighbor sampled.
    """
    lo, hi = ratio_range
    if lo == hi == 1.0:
        return x
    r = float(torch.empty(()).uniform_(lo, hi).item())
    T = x.shape[0]
    new_T = max(8, int(round(T * r)))
    # interpolate xy linearly, validity nearest
    v = _to_jview(x).clone()
    xy = v[..., :2].permute(1, 2, 0).unsqueeze(0)  # (1, J, 2, T)
    xy_new = torch.nn.functional.interpolate(xy, size=new_T, mode="linear", align_corners=False)
    xy_new = xy_new.squeeze(0).permute(2, 0, 1)    # (new_T, J, 2)
    val = v[..., 2:3].permute(1, 2, 0).unsqueeze(0)
    val_new = torch.nn.functional.interpolate(val, size=new_T, mode="nearest")
    val_new = val_new.squeeze(0).permute(2, 0, 1)
    out = torch.cat([xy_new, val_new], dim=-1)
    return _from_jview(out)


# ---------- compositional augmentor -----------------------------------------

class SkeletonAugment:
    """Default RESONA augmentation pipeline.

    Designed to be cheap (no allocs beyond the cloned tensor) and to preserve
    skeleton semantics: validity flags are not mutated (except by speed perturbation,
    where they are nearest-neighbor resampled).
    """

    def __init__(
        self,
        spatial_sigma: float = 0.02,
        flip_prob: float = 0.5,
        speed_range: tuple[float, float] = (0.85, 1.15),
        n_joints: int = 77,
    ):
        self.sigma = spatial_sigma
        self.flip_prob = flip_prob
        self.speed_range = speed_range
        self.n_joints = n_joints

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if self.speed_range != (1.0, 1.0):
            x = temporal_speed(x, self.speed_range)
        if self.flip_prob > 0:
            x = horizontal_flip(x, self.n_joints, self.flip_prob)
        if self.sigma > 0:
            x = spatial_jitter(x, self.sigma, self.n_joints)
        return x
