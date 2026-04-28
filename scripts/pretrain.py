"""RESONA pretrain entry — DDP-ready via resona.trainer.Trainer.

Single-GPU sanity:
    python scripts/pretrain.py --config configs/sanity.yaml --steps 10

Multi-GPU (Slurm-launched):
    srun python scripts/pretrain.py --config configs/pretrain_300m.yaml \\
        --resume /scratch/$USER/resona/ckpt/last.pt \\
        --ckpt-dir /scratch/$USER/resona/ckpt
"""
from __future__ import annotations

import argparse
import sys

import yaml

from resona.trainer import Trainer


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None,
                    help="override train.steps / train.total_steps for quick runs")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        # respect either key
        if "total_steps" in cfg["train"]:
            cfg["train"]["total_steps"] = args.steps
        cfg["train"]["steps"] = args.steps

    trainer = Trainer(cfg, ckpt_dir=args.ckpt_dir, run_dir=args.run_dir)
    if args.resume:
        trainer.load_ckpt(args.resume)
    trainer.fit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
