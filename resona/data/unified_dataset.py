"""UnifiedSignDataset — concat-style dataset over multiple RESONA-77 H5 files.

H5 layout assumed:
    {h5_root}/{split}/{video_id} -> (T, 234) float32
or:
    {h5_root}/{group}/{split}/{video_id} -> (T, 234) float32   (KSL103 case)

Each H5 file is opened lazily per worker (avoids fork-after-open issues).
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


@dataclass
class ClipIndex:
    file_idx: int
    h5_path_in_file: str  # absolute path inside the H5
    n_frames: int


class UnifiedSignDataset(Dataset):
    """Union of RESONA-77 H5 files as one flat dataset.

    Args:
        h5_paths: list of *.h5 absolute paths.
        clip_len: number of frames per sample (random crop / pad).
        flat_dim: 234 for canonical RESONA-77 (77 joints × (x,y,validity)).
        split: 'train' | 'dev' | 'test' | None (None = all).
        min_frames: drop clips shorter than this.
    """

    def __init__(
        self,
        h5_paths: Sequence[str],
        clip_len: int = 256,
        flat_dim: int = 234,
        split: str | None = "train",
        min_frames: int = 8,
    ):
        self.h5_paths = [str(p) for p in h5_paths]
        self.clip_len = clip_len
        self.flat_dim = flat_dim
        self.split = split
        self.min_frames = min_frames

        self.index: list[ClipIndex] = []
        self._build_index()

        # per-worker handles (lazy)
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
                # filter by split if last path segment looks like split name
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

    # ----- API --------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        ci = self.index[idx]
        ds = self._get(ci.file_idx)[ci.h5_path_in_file]
        T = ds.shape[0]

        # random crop / pad to clip_len
        if T >= self.clip_len:
            start = np.random.randint(0, T - self.clip_len + 1)
            arr = ds[start : start + self.clip_len]
            mask = np.ones(self.clip_len, dtype=np.bool_)
        else:
            arr = np.zeros((self.clip_len, self.flat_dim), dtype=np.float32)
            arr[:T] = ds[:]
            mask = np.zeros(self.clip_len, dtype=np.bool_)
            mask[:T] = True

        x = torch.from_numpy(np.ascontiguousarray(arr, dtype=np.float32))  # (T, 234)
        m = torch.from_numpy(mask)  # (T,)
        return {"x": x, "valid_mask": m, "src_file": ci.file_idx}


def discover_h5(data_dir: str | Path, datasets: Sequence[str] | str = "all") -> list[str]:
    """Find skeleton_resona77_*.h5 in `data_dir`. `datasets` filters by name suffix."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("skeleton_resona77_*.h5"))
    if datasets == "all":
        return [str(p) for p in files]
    keep = set(datasets)
    return [str(p) for p in files if any(k in p.stem for k in keep)]
