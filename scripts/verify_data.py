"""Verify RESONA-77 H5 corpus on the cluster.

Usage:
    python scripts/verify_data.py --data-dir /scratch/$USER/resona/data
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py


def walk(g: h5py.Group, prefix: str = "") -> list[tuple[str, tuple]]:
    out = []
    for k, v in g.items():
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Group):
            out.extend(walk(v, p))
        else:
            out.append((p, v.shape))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("skeleton_resona77_*.h5"))
    print(f"[verify_data] {len(files)} H5 files in {data_dir}")

    total_clips = 0
    total_frames = 0
    bad = 0
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                ds = walk(h)
                clips = [s for s in ds if len(s[1]) == 2 and s[1][1] == 234]
                fr = sum(s[1][0] for s in clips)
                total_clips += len(clips)
                total_frames += fr
                print(f"  {f.name:50s} clips={len(clips):>7d} frames={fr:>10d} size={f.stat().st_size/1e9:.2f}GB")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {e}")
            bad += 1

    print(f"\nTOTAL: {total_clips:,} clips / {total_frames:,} frames / {bad} errors")


if __name__ == "__main__":
    main()
