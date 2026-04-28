"""Verify RESONA-77 H5 corpus on the cluster.

Usage:
    python scripts/verify_data.py --data-dir /scratch/$USER/resona/data
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py


def walk(g: h5py.Group, prefix: str = "") -> tuple[list[tuple[str, tuple]], int]:
    """Returns (entries, n_skipped). Robust to broken links and weird types."""
    out: list[tuple[str, tuple]] = []
    skipped = 0
    for k in g.keys():
        try:
            v = g[k]
        except (KeyError, OSError) as e:
            skipped += 1
            continue
        if v is None:
            skipped += 1
            continue
        p = f"{prefix}/{k}" if prefix else k
        if isinstance(v, h5py.Group):
            sub, sk = walk(v, p)
            out.extend(sub)
            skipped += sk
        elif isinstance(v, h5py.Dataset):
            try:
                out.append((p, v.shape))
            except Exception:
                skipped += 1
        else:
            skipped += 1
    return out, skipped


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    files = sorted(data_dir.glob("skeleton_resona77_*.h5"))
    print(f"[verify_data] {len(files)} H5 files in {data_dir}")

    total_clips = 0
    total_frames = 0
    bad_files = 0
    total_skipped = 0
    for f in files:
        try:
            with h5py.File(f, "r") as h:
                ds, sk = walk(h)
                clips = [s for s in ds if len(s[1]) == 2 and s[1][1] == 234]
                fr = sum(s[1][0] for s in clips)
                total_clips += len(clips)
                total_frames += fr
                total_skipped += sk
                tag = f"  [skipped {sk}]" if sk else ""
                print(f"  {f.name:50s} clips={len(clips):>7d} frames={fr:>10d} size={f.stat().st_size/1e9:.2f}GB{tag}")
        except Exception as e:
            print(f"  [ERROR] {f.name}: {type(e).__name__}: {e}")
            bad_files += 1

    print(f"\nTOTAL: {total_clips:,} clips / {total_frames:,} frames"
          f" / {bad_files} unreadable file(s) / {total_skipped} skipped entries")


if __name__ == "__main__":
    main()
