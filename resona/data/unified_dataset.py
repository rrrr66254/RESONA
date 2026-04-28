"""UnifiedSignDataset — concat-style dataset over multiple RESONA-77 H5 files.

H5 layout assumed:
    {h5_root}/{split}/{video_id} -> (T, 234) float32
or:
    {h5_root}/{group}/{split}/{video_id} -> (T, 234) float32   (KSL103 case)

Each H5 file is opened lazily per worker (avoids fork-after-open issues).

Sample modes:
  - "single": returns one augmented clip per index
  - "two_view": returns two independently-augmented temporal crops of the same
                clip — used for BYOL-style temporal contrastive learning.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from .augment import SkeletonAugment, temporal_crop


@dataclass
class ClipIndex:
    file_idx: int
    h5_path_in_file: str
    n_frames: int


class UnifiedSignDataset(Dataset):
    """Union of RESONA-77 H5 files as one flat dataset.

    Args:
        h5_paths: list of *.h5 absolute paths.
        clip_len: number of frames per sample (random crop / pad).
        flat_dim: 234 for canonical RESONA-77 (77 joints × (x,y,validity)).
        split: 'train' | 'dev' | 'test' | None (None = all).
        min_frames: drop clips shorter than this.
        sample_mode: "single" or "two_view".
        augment: SkeletonAugment | None — applied per view.
    """

    def __init__(
        self,
        h5_paths: Sequence[str],
        clip_len: int = 256,
        flat_dim: int = 234,
        split: str | None = "train",
        min_frames: int = 8,
        sample_mode: str = "single",
        augment: SkeletonAugment | None = None,
    ):
        assert sample_mode in ("single", "two_view")
        self.h5_paths = [str(p) for p in h5_paths]
        self.clip_len = clip_len
        self.flat_dim = flat_dim
        self.split = split
        self.min_frames = min_frames
        self.sample_mode = sample_mode
        self.augment = augment

        self.index: list[ClipIndex] = []
        self._build_index()
        self._handles: dict[int, h5py.File] = {}

    # ----- index ------------------------------------------------------------

    def _build_index(self) -> None:
        for fi, p in enumerate(self.h5_paths):
            if not os.path.exists(p):
                print(f"[warn] missing H5: {p}")
                continue
            with h5py.File(p, "r") as f:
                self._scan(f, fi, "")
        print(f"[UnifiedSignDataset] {len(self.index)} clips across {len(self.h5_paths)} H5")

    def _scan(self, group: h5py.Group, file_idx: int, prefix: str) -> None:
        for k, v in group.items():
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Group):
                if self.split is not None and k in {"train", "dev", "val", "test", "validation"}:
                    if not self._split_match(k):
                        continue
                self._scan(v, file_idx, path)
            elif isinstance(v, h5py.Dataset):
                if v.ndim == 2 and v.shape[-1] == self.flat_dim and v.shape[0] >= self.min_frames:
                    self.index.append(
                        ClipIndex(file_idx=file_idx, h5_path_in_file=path, n_frames=v.shape[0])
                    )

    def _split_match(self, k: str) -> bool:
        if self.split is None:
            return True
        if self.split == "dev":
            return k in {"dev", "val", "validation"}
        return k == self.split

    # ----- handles ----------------------------------------------------------

    def _get(self, file_idx: int) -> h5py.File:
        h = self._handles.get(file_idx)
        if h is None:
            h = h5py.File(self.h5_paths[file_idx], "r", swmr=True)
            self._handles[file_idx] = h
        return h

    # ----- sampling helpers -------------------------------------------------

    def _load_full(self, ci: ClipIndex) -> torch.Tensor:
        """Load the entire clip as a torch tensor (T, 234)."""
        ds = self._get(ci.file_idx)[ci.h5_path_in_file]
        arr = ds[:]  # full read; H5 chunked, OK
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))

    def _make_view(self, full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Independent temporal crop (with implicit pad) + augmentation.
        Returns (x: (clip_len, 234), valid_mask: (clip_len,) bool).
        """
        T = full.shape[0]
        cropped = temporal_crop(full, self.clip_len)
        valid = torch.zeros(self.clip_len, dtype=torch.bool)
        valid[: min(T, self.clip_len)] = True
        if self.augment is not None:
            cropped = self.augment(cropped)
        return cropped, valid

    # ----- API --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ci = self.index[idx]
        full = self._load_full(ci)

        if self.sample_mode == "single":
            x, m = self._make_view(full)
            return {"x": x, "valid_mask": m, "src_file": ci.file_idx}

        # two_view: two independent crops + augmentations
        x1, m1 = self._make_view(full)
        x2, m2 = self._make_view(full)
        return {
            "x1": x1, "valid_mask1": m1,
            "x2": x2, "valid_mask2": m2,
            "src_file": ci.file_idx,
        }


def discover_h5(data_dir: str | Path, datasets: Sequence[str] | str = "all") -> list[str]:
    """Find skeleton_resona77_*.h5 in `data_dir`. `datasets` filters by name suffix."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("skeleton_resona77_*.h5"))
    if datasets == "all":
        return [str(p) for p in files]
    keep = set(datasets)
    return [str(p) for p in files if any(k in p.stem for k in keep)]
