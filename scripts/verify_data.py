"""Verify RESONA-77 H5 corpus on the cluster.

Usage:
    python scripts/verify_data.py --data-dir /scratch/$USER/resona/data
    python scripts/verify_data.py --data-dir ... --quick   # fast (sample only)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import h5py


def walk(g: h5py.Group, prefix: str = "") -> tuple[list[tuple[str, tuple]], int]:
    out: list[tuple[str, tuple]] = []
    skipped = 0
    try:
        keys = list(g.keys())
    except Exception:
        return out, 1
    for k in keys:
        try:
            v = g[k]
        except Exception:
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


def quick_probe(h: h5py.File) -> tuple[int, int]:
    """Estimate clips/frames by sampling top groups, no full traversal.

    Returns (estimated_clips, estimated_frames).
    """
    try:
        roots = list(h.keys())
    except Exception:
        return 0, 0
    total_clips = 0
    total_frames = 0
    for r in roots:
        try:
            grp = h[r]
        except Exception:
            continue
        if not isinstance(grp, h5py.Group):
            continue
        # try splits inside
        try:
            sub = list(grp.keys())
        except Exception:
            continue
        for s in sub:
            try:
                node = grp[s]
            except Exception:
                continue
            if isinstance(node, h5py.Group):
                # split-level: count via len() (O(1) for h5py group)
                try:
                    n = len(node)
                except Exception:
                    n = 0
                total_clips += n
                # sample 5 clips for avg frames
                try:
                    sample_keys = list(node.keys())[:5]
                except Exception:
                    sample_keys = []
                if sample_keys:
                    avg_t = 0
                    cnt = 0
                    for k in sample_keys:
                        try:
                            arr = node[k]
                            if isinstance(arr, h5py.Dataset) and arr.ndim == 2:
                                avg_t += arr.shape[0]
                                cnt += 1
                        except Exception:
                            pass
                    if cnt:
                        avg_t //= cnt
                        total_frames += n * avg_t
            elif isinstance(node, h5py.Dataset) and node.ndim == 2:
                total_clips += 1
                try:
                    total_frames += node.shape[0]
                except Exception:
                    pass
    return total_clips, total_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--quick", action="store_true",
                    help="Sample-based estimation (~1s/file). For large H5.")
    args = ap.parse_args()

    files = sorted(Path(args.data_dir).glob("skeleton_resona77_*.h5"))
    mode = "QUICK" if args.quick else "FULL"
    print(f"[verify_data {mode}] {len(files)} H5 files in {args.data_dir}")

    total_clips = total_frames = bad = total_skipped = 0
    for f in files:
        sys.stdout.write(f"  {f.name:50s} ... ")
        sys.stdout.flush()
        t0 = time.time()
        try:
            with h5py.File(f, "r") as h:
                if args.quick:
                    clips, fr = quick_probe(h)
                    sk = 0
                    star = "~"
                else:
                    ds, sk = walk(h)
                    clips_list = [s for s in ds if len(s[1]) == 2 and s[1][1] == 234]
                    clips = len(clips_list)
                    fr = sum(s[1][0] for s in clips_list)
                    star = ""
                total_clips += clips
                total_frames += fr
                total_skipped += sk
                tag = f" [skipped {sk}]" if sk else ""
                size = f.stat().st_size / 1e9
                dt = time.time() - t0
                print(f"clips={star}{clips:>7d} frames={star}{fr:>10d} size={size:.2f}GB{tag}  ({dt:.1f}s)")
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")
            bad += 1

    star = "~" if args.quick else ""
    print(f"\nTOTAL: {star}{total_clips:,} clips / {star}{total_frames:,} frames"
          f" / {bad} bad / {total_skipped} skipped")


if __name__ == "__main__":
    main()
