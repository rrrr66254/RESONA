"""Skeleton graph adjacency for RESONA-77 sub-poses.

RESONA-77 layout (joint indices, contiguous per group):
    body9    : 0-8     (nose, l/r shoulder, l/r elbow, l/r wrist, l/r hip)
    lhand21  : 9-29    (wrist + 5 fingers × 4 joints, MediaPipe hand)
    rhand21  : 30-50
    mouth8   : 51-58
    face18   : 59-76

For each sub-pose we define a symmetric adjacency (incl. self-loops).
Identity-init topology is "anatomical chain" — minimal viable; refine later.
"""
from __future__ import annotations

import torch


def _chain_adj(n: int, edges: list[tuple[int, int]]) -> torch.Tensor:
    """Build symmetric adjacency from edge list, with self-loops.
    Returns normalized A_hat = D^-1/2 (A + I) D^-1/2 of shape (n, n).
    """
    A = torch.zeros(n, n)
    for i, j in edges:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A = A + torch.eye(n)            # self-loop
    d = A.sum(dim=1).clamp(min=1)
    D_inv_sqrt = torch.diag(d.pow(-0.5))
    return D_inv_sqrt @ A @ D_inv_sqrt


def body9_adj() -> torch.Tensor:
    """Body9: 0=nose, 1=l_shoulder, 2=r_shoulder, 3=l_elbow, 4=r_elbow,
    5=l_wrist, 6=r_wrist, 7=l_hip, 8=r_hip.
    """
    edges = [
        (0, 1), (0, 2),     # nose-shoulders
        (1, 2),             # l_shoulder-r_shoulder
        (1, 3), (3, 5),     # l arm chain
        (2, 4), (4, 6),     # r arm chain
        (1, 7), (2, 8),     # shoulders-hips
        (7, 8),             # hips
    ]
    return _chain_adj(9, edges)


def hand21_adj() -> torch.Tensor:
    """MediaPipe hand: 0=wrist, then 5 fingers × 4 joints each.
        thumb : 1-2-3-4
        index : 5-6-7-8
        middle: 9-10-11-12
        ring  : 13-14-15-16
        pinky : 17-18-19-20
    """
    edges = []
    # wrist to base of each finger
    for finger_base in [1, 5, 9, 13, 17]:
        edges.append((0, finger_base))
    # finger chains
    for finger_base in [1, 5, 9, 13, 17]:
        for k in range(3):
            edges.append((finger_base + k, finger_base + k + 1))
    # adjacent finger bases (palm)
    for a, b in [(1, 5), (5, 9), (9, 13), (13, 17)]:
        edges.append((a, b))
    return _chain_adj(21, edges)


def mouth8_adj() -> torch.Tensor:
    """Mouth8 — assume 8 points around mouth, ring topology."""
    edges = [(i, (i + 1) % 8) for i in range(8)]
    return _chain_adj(8, edges)


def face18_adj() -> torch.Tensor:
    """Face18 — 18 face landmarks. Use anatomical groups + chain.
    Without precise landmark IDs, we use a fully-connected sparse approximation:
    each point connected to 2 nearest in the index space (chain).
    """
    edges = [(i, i + 1) for i in range(17)]
    return _chain_adj(18, edges)


SUBPOSE_INFO: dict[str, dict] = {
    "body":   {"start": 0,  "n": 9,  "adj_fn": body9_adj},
    "lhand":  {"start": 9,  "n": 21, "adj_fn": hand21_adj},
    "rhand":  {"start": 30, "n": 21, "adj_fn": hand21_adj},
    "mouth":  {"start": 51, "n": 8,  "adj_fn": mouth8_adj},
    "face":   {"start": 59, "n": 18, "adj_fn": face18_adj},
}


def get_subpose_indices(name: str) -> tuple[int, int]:
    """Return (start, end) joint indices for sub-pose."""
    info = SUBPOSE_INFO[name]
    return info["start"], info["start"] + info["n"]


def get_adj(name: str) -> torch.Tensor:
    return SUBPOSE_INFO[name]["adj_fn"]()
