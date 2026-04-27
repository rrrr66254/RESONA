"""RESONA pretrain entry — sanity-runnable; full DDP/AMP TBD.

This is a *minimal* runner so `slurm/sanity.sh` can succeed end-to-end.
The full distributed trainer will live in resona/trainer.py later.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch
import yaml
from torch.utils.data import DataLoader

from resona.data.unified_dataset import UnifiedSignDataset, discover_h5
from resona.models.transformer import TransformerEncoder
from resona.models.msm_head import MSMHead, make_joint_mask, apply_mask, msm_loss
from resona.models.contrastive_head import TemporalContrastive


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", type=int, default=None, help="override train.steps for sanity")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--run-dir", default=None)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[pretrain] device={device}")

    # ----- data -----
    data_cfg = cfg["data"]
    h5_paths = discover_h5(data_cfg["data_dir"], data_cfg.get("datasets", "all"))
    print(f"[pretrain] H5 files: {len(h5_paths)}")
    for p in h5_paths:
        print(f"  - {os.path.basename(p)}")

    ds = UnifiedSignDataset(
        h5_paths=h5_paths,
        clip_len=data_cfg["clip_len"],
        flat_dim=data_cfg["flat_dim"],
        split="train",
    )
    dl = DataLoader(
        ds,
        batch_size=data_cfg["batch_size"],
        shuffle=True,
        num_workers=data_cfg.get("num_workers", 2),
        pin_memory=True,
        drop_last=True,
    )

    # ----- model -----
    m_cfg = cfg["model"]
    encoder = TransformerEncoder(
        in_dim=data_cfg["flat_dim"],
        d_model=m_cfg["d_model"],
        n_heads=m_cfg["n_heads"],
        n_layers=m_cfg["n_layers"],
        ffn_dim=m_cfg["ffn_dim"],
        dropout=m_cfg.get("dropout", 0.1),
        max_len=m_cfg["max_len"],
    ).to(device)

    msm_head = MSMHead(d_model=m_cfg["d_model"]).to(device)

    o_cfg = cfg["objective"]
    use_msm = o_cfg["msm"]["enabled"]
    use_con = o_cfg["contrastive"]["enabled"]

    contrastive = None
    if use_con:
        contrastive = TemporalContrastive(
            encoder=encoder,
            d_model=m_cfg["d_model"],
            proj_dim=o_cfg["contrastive"]["proj_dim"],
            ema_decay=o_cfg["contrastive"]["ema_decay"],
        ).to(device)

    params = list(encoder.parameters()) + list(msm_head.parameters())
    if contrastive is not None:
        # online + heads only (target is EMA, no grad)
        params += list(contrastive.proj_online.parameters())
        params += list(contrastive.predict.parameters())

    t_cfg = cfg["train"]
    opt = torch.optim.AdamW(params, lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])

    # ----- train loop -----
    n_params = sum(p.numel() for p in encoder.parameters())
    print(f"[pretrain] encoder params: {n_params/1e6:.1f}M")

    target_steps = t_cfg.get("steps", t_cfg.get("total_steps", 100))
    step = 0
    t0 = time.time()
    encoder.train()
    msm_head.train()

    while step < target_steps:
        for batch in dl:
            x = batch["x"].to(device, non_blocking=True)
            valid = batch["valid_mask"].to(device, non_blocking=True)

            loss = x.new_zeros(())
            if use_msm:
                m = make_joint_mask(x, n_joints=77, mask_ratio=o_cfg["msm"]["mask_ratio"])
                x_masked = apply_mask(x, m)
                h = encoder(x_masked, valid)
                pred = msm_head(h)
                l_msm = msm_loss(pred, x, m, valid)
                loss = loss + o_cfg["msm"].get("loss_weight", 1.0) * l_msm
            if use_con and contrastive is not None:
                # cheap two-view: same clip, different mask noise (placeholder)
                l_con = contrastive(x, x, valid, valid)
                loss = loss + o_cfg["contrastive"].get("loss_weight", 0.5) * l_con

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, t_cfg.get("grad_clip", 1.0))
            opt.step()
            if contrastive is not None:
                contrastive.update_target()

            if step % t_cfg.get("log_interval", 10) == 0:
                dt = time.time() - t0
                print(f"step {step:>6d} loss={loss.item():.4f} ({dt:.1f}s)")
            step += 1
            if step >= target_steps:
                break

    print(f"[pretrain] done. {target_steps} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    sys.exit(main() or 0)
