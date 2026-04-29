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

Index caching:
  Walking yt_asl alone (390K clips) takes ~35 min on Lustre. The index is
  cached to disk after first build, keyed by (h5 paths + mtimes + split).
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import time
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
        cache_dir: directory to store/load pickled index. None disables caching.
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
        cache_dir: str | Path | None = None,
    ):
        assert sample_mode in ("single", "two_view")
        self.h5_paths = [str(p) for p in h5_paths]
        self.clip_len = clip_len
        self.flat_dim = flat_dim
        self.split = split
        self.min_frames = min_frames
        self.sample_mode = sample_mode
        self.augment = augment
        self.cache_dir = Path(cache_dir) if cache_dir else None

        self.index: list[ClipIndex] = []
        self._build_index()
        self._handles: dict[int, h5py.File] = {}

    # ----- index ------------------------------------------------------------

    def _cache_key(self) -> str:
        """Hash of h5 path + mtime + size + split + flat_dim + min_frames.

        Invalidates cache automatically when any H5 file changes.
        """
        parts = []
        for p in self.h5_paths:
            try:
                st = os.stat(p)
                parts.append(f"{p}:{st.st_mtime_ns}:{st.st_size}")
            except OSError:
                parts.append(f"{p}:MISSING")
        parts.append(f"split={self.split}")
        parts.append(f"flat_dim={self.flat_dim}")
        parts.append(f"min_frames={self.min_frames}")
        h = hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]
        return h

    def _cache_path(self) -> Path | None:
        if self.cache_dir is None:
            return None
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return self.cache_dir / f"resona_index_{self._cache_key()}.pkl"

    def _try_load_cache(self) -> bool:
        cp = self._cache_path()
        if cp is None or not cp.exists():
            return False
        try:
            with open(cp, "rb") as f:
                payload = pickle.load(f)
            if payload.get("version") != 1:
                return False
            self.index = payload["index"]
            print(f"[UnifiedSignDataset] loaded index from cache: "
                  f"{len(self.index)} clips ({cp.name})")
            return True
        except Exception as e:
            print(f"[UnifiedSignDataset] cache load failed: {e}")
            return False

    def _save_cache(self) -> None:
        cp = self._cache_path()
        if cp is None:
            return
        try:
            with open(cp, "wb") as f:
                pickle.dump({"version": 1, "index": self.index}, f, protocol=4)
            print(f"[UnifiedSignDataset] saved index cache: {cp.name}")
        except Exception as e:
            print(f"[UnifiedSignDataset] cache save failed: {e}")

    def _build_index(self) -> None:
        if self._try_load_cache():
            return
        t0 = time.time()
        for fi, p in enumerate(self.h5_paths):
            if not os.path.exists(p):
                print(f"[warn] missing H5: {p}")
                continue
            t1 = time.time()
            try:
                with h5py.File(p, "r") as f:
                    self._scan(f, fi, "")
                print(f"  scanned {os.path.basename(p)}: total now "
                      f"{len(self.index)} clips ({time.time()-t1:.1f}s)")
            except Exception as e:
                print(f"  [WARN] scan failed for {p}: {type(e).__name__}: {e}")
        print(f"[UnifiedSignDataset] {len(self.index)} clips across "
              f"{len(self.h5_paths)} H5 (build took {time.time()-t0:.1f}s)")
        self._save_cache()

    def _scan(self, group: h5py.Group, file_idx: int, prefix: str) -> None:
        try:
            keys = list(group.keys())
        except Exception:
            return  # corrupt group — skip silently
        for k in keys:
            try:
                v = group[k]
            except Exception:
                continue
            if v is None:
                continue
            path = f"{prefix}/{k}" if prefix else k
            if isinstance(v, h5py.Group):
                if self.split is not None and k in {"train", "dev", "val", "test", "validation"}:
                    if not self._split_match(k):
                        continue
                self._scan(v, file_idx, path)
            elif isinstance(v, h5py.Dataset):
                try:
                    shape = v.shape
                except Exception:
                    continue
                if len(shape) == 2 and shape[-1] == self.flat_dim and shape[0] >= self.min_frames:
                    self.index.append(
                        ClipIndex(file_idx=file_idx, h5_path_in_file=path, n_frames=shape[0])
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

        Order: pre_crop aug (temporal_speed) → temporal_crop → post_crop aug
        (flip, jitter). This keeps clip_len invariant for batch stacking.
        """
        if self.augment is not None:
            full = self.augment.pre_crop(full)     # may change T
        T = full.shape[0]
        cropped = temporal_crop(full, self.clip_len)
        valid = torch.zeros(self.clip_len, dtype=torch.bool)
        valid[: min(T, self.clip_len)] = True
        if self.augment is not None:
            cropped = self.augment.post_crop(cropped)
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
    """Find skeleton_resona77_*.h5 in `data_dir`.

    `datasets`:
      - "all": include every `skeleton_resona77_*.h5` in dir
      - list of dataset names: include only files with stem == `skeleton_resona77_{name}`
        (exact match — protects against accidental matches like `autsl_corrupt`)
    """
    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("skeleton_resona77_*.h5"))
    if datasets == "all":
        return [str(p) for p in files]
    keep = set(datasets)
    expected = {f"skeleton_resona77_{k}" for k in keep}
    return [str(p) for p in files if p.stem in expected]
