"""DDP-aware RESONA pretrain trainer.

Single-GPU and multi-GPU paths share this Trainer. DDP is initialized lazily
from environment (`torchrun` or Slurm `srun` populates LOCAL_RANK / RANK / WORLD_SIZE).
"""
from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from .data.augment import SkeletonAugment
from .data.unified_dataset import UnifiedSignDataset, discover_h5
from .models.contrastive_head import TemporalContrastive
from .models.msm_head import MSMHead, apply_mask, make_joint_mask, msm_loss
from .models.transformer import TransformerEncoder


# ---------- distributed bootstrap -------------------------------------------

@dataclass
class DistInfo:
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    is_dist: bool = False
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistInfo:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        torch.cuda.set_device(local)
        return DistInfo(rank=rank, world_size=world, local_rank=local,
                        is_dist=True, device=torch.device(f"cuda:{local}"))
    return DistInfo()


# ---------- LR schedule ------------------------------------------------------

def cosine_with_warmup(step: int, total: int, warmup: int, base_lr: float, min_ratio: float = 0.1) -> float:
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    if total <= warmup:
        return base_lr
    p = (step - warmup) / (total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


# ---------- trainer ----------------------------------------------------------

class Trainer:
    def __init__(self, cfg: dict[str, Any], ckpt_dir: str | None = None, run_dir: str | None = None):
        self.cfg = cfg
        self.dist = init_distributed()
        self.ckpt_dir = Path(ckpt_dir) if ckpt_dir else None
        self.run_dir = Path(run_dir) if run_dir else None
        if self.dist.is_main and self.ckpt_dir:
            self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        if self.dist.is_main and self.run_dir:
            self.run_dir.mkdir(parents=True, exist_ok=True)

        self._build_data()
        self._build_model()
        self._build_optim()
        self.step = 0

    # ---- build ----

    def _build_data(self) -> None:
        d = self.cfg["data"]
        h5 = discover_h5(d["data_dir"], d.get("datasets", "all"))
        if self.dist.is_main:
            print(f"[Trainer] H5 files: {len(h5)}")
        aug_cfg = d.get("augment", {}) or {}
        aug = SkeletonAugment(
            spatial_sigma=aug_cfg.get("spatial_jitter", 0.02),
            flip_prob=aug_cfg.get("horizontal_flip_prob", 0.5),
            speed_range=aug_cfg.get("speed_range", (0.85, 1.15)),
        )
        sample_mode = "two_view" if self.cfg["objective"]["contrastive"]["enabled"] else "single"
        self.train_set = UnifiedSignDataset(
            h5_paths=h5,
            clip_len=d["clip_len"],
            flat_dim=d["flat_dim"],
            split="train",
            sample_mode=sample_mode,
            augment=aug,
            cache_dir=d.get("index_cache_dir"),
        )
        sampler = DistributedSampler(self.train_set, shuffle=True, drop_last=True) if self.dist.is_dist else None
        self.train_loader = DataLoader(
            self.train_set,
            batch_size=d["batch_size"],
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=d.get("num_workers", 4),
            pin_memory=True,
            drop_last=True,
            persistent_workers=d.get("num_workers", 4) > 0,
        )
        self.train_sampler = sampler

    def _build_model(self) -> None:
        m = self.cfg["model"]
        d = self.cfg["data"]
        self.encoder = TransformerEncoder(
            in_dim=d["flat_dim"],
            d_model=m["d_model"],
            n_heads=m["n_heads"],
            n_layers=m["n_layers"],
            ffn_dim=m["ffn_dim"],
            dropout=m.get("dropout", 0.1),
            max_len=m["max_len"],
        ).to(self.dist.device)
        self.msm_head = MSMHead(d_model=m["d_model"]).to(self.dist.device)

        o = self.cfg["objective"]
        self.use_msm = o["msm"]["enabled"]
        self.use_con = o["contrastive"]["enabled"]
        self.contrastive: TemporalContrastive | None = None
        if self.use_con:
            self.contrastive = TemporalContrastive(
                encoder=self.encoder,
                d_model=m["d_model"],
                proj_dim=o["contrastive"]["proj_dim"],
                ema_decay=o["contrastive"]["ema_decay"],
            ).to(self.dist.device)

        if self.dist.is_dist:
            self.encoder = DDP(self.encoder, device_ids=[self.dist.local_rank],
                               find_unused_parameters=False)
            self.msm_head = DDP(self.msm_head, device_ids=[self.dist.local_rank])
            if self.contrastive is not None:
                # online wrapped via encoder (already DDP'd); only proj_online + predict need DDP
                self.contrastive.proj_online = DDP(self.contrastive.proj_online, device_ids=[self.dist.local_rank])
                self.contrastive.predict = DDP(self.contrastive.predict, device_ids=[self.dist.local_rank])

        if self.dist.is_main:
            n = sum(p.numel() for p in self._encoder_inner().parameters())
            print(f"[Trainer] encoder params: {n/1e6:.1f}M")

    def _encoder_inner(self) -> nn.Module:
        e = self.encoder
        return e.module if isinstance(e, DDP) else e

    def _build_optim(self) -> None:
        t = self.cfg["train"]
        params: list[nn.Parameter] = []
        params += list(self._encoder_inner().parameters())
        params += list((self.msm_head.module if isinstance(self.msm_head, DDP) else self.msm_head).parameters())
        if self.contrastive is not None:
            for m in (self.contrastive.proj_online, self.contrastive.predict):
                params += list((m.module if isinstance(m, DDP) else m).parameters())
        self.params = [p for p in params if p.requires_grad]
        self.opt = torch.optim.AdamW(
            self.params,
            lr=t["lr"],
            weight_decay=t.get("weight_decay", 0.05),
            betas=(0.9, 0.95),
        )
        self.amp = t.get("amp", "bf16")
        self.scaler = torch.amp.GradScaler("cuda") if self.amp == "fp16" else None

    # ---- train step ----

    def _amp_dtype(self):
        if self.amp == "bf16":
            return torch.bfloat16
        if self.amp == "fp16":
            return torch.float16
        return torch.float32

    def _step_batch(self, batch: dict) -> dict[str, float]:
        t = self.cfg["train"]
        o = self.cfg["objective"]
        dev = self.dist.device

        loss = torch.zeros((), device=dev)
        log: dict[str, float] = {}

        with torch.amp.autocast("cuda", enabled=(self.amp != "fp32"), dtype=self._amp_dtype()):
            if self.use_con and self.contrastive is not None:
                x1 = batch["x1"].to(dev, non_blocking=True)
                x2 = batch["x2"].to(dev, non_blocking=True)
                m1 = batch["valid_mask1"].to(dev, non_blocking=True)
                m2 = batch["valid_mask2"].to(dev, non_blocking=True)

                # MSM uses view 1
                if self.use_msm:
                    mask = make_joint_mask(x1, n_joints=77, mask_ratio=o["msm"]["mask_ratio"])
                    x_masked = apply_mask(x1, mask)
                    h = self.encoder(x_masked, m1)
                    pred = self.msm_head(h)
                    l_msm = msm_loss(pred, x1, mask, m1)
                    loss = loss + o["msm"].get("loss_weight", 1.0) * l_msm
                    log["msm"] = float(l_msm.detach())

                l_con = self.contrastive(x1, x2, m1, m2)
                loss = loss + o["contrastive"].get("loss_weight", 0.5) * l_con
                log["con"] = float(l_con.detach())
            else:
                x = batch["x"].to(dev, non_blocking=True)
                m = batch["valid_mask"].to(dev, non_blocking=True)
                if self.use_msm:
                    mask = make_joint_mask(x, n_joints=77, mask_ratio=o["msm"]["mask_ratio"])
                    x_masked = apply_mask(x, mask)
                    h = self.encoder(x_masked, m)
                    pred = self.msm_head(h)
                    l_msm = msm_loss(pred, x, mask, m)
                    loss = loss + o["msm"].get("loss_weight", 1.0) * l_msm
                    log["msm"] = float(l_msm.detach())

        self.opt.zero_grad(set_to_none=True)
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.opt)
            torch.nn.utils.clip_grad_norm_(self.params, t.get("grad_clip", 1.0))
            self.scaler.step(self.opt)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.params, t.get("grad_clip", 1.0))
            self.opt.step()

        if self.contrastive is not None:
            self.contrastive.update_target()

        # LR schedule (per-step cosine)
        total = t.get("total_steps", t.get("steps", 100))
        warm = t.get("warmup_steps", 0)
        new_lr = cosine_with_warmup(self.step, total, warm, t["lr"])
        for g in self.opt.param_groups:
            g["lr"] = new_lr

        log["loss"] = float(loss.detach())
        log["lr"] = new_lr
        return log

    # ---- save ----

    def save_ckpt(self, name: str = "last.pt") -> None:
        if not self.dist.is_main or self.ckpt_dir is None:
            return
        state = {
            "step": self.step,
            "encoder": self._encoder_inner().state_dict(),
            "msm_head": (self.msm_head.module if isinstance(self.msm_head, DDP) else self.msm_head).state_dict(),
        }
        if self.contrastive is not None:
            state["contrastive_proj_online"] = (
                self.contrastive.proj_online.module if isinstance(self.contrastive.proj_online, DDP) else self.contrastive.proj_online
            ).state_dict()
            state["contrastive_predict"] = (
                self.contrastive.predict.module if isinstance(self.contrastive.predict, DDP) else self.contrastive.predict
            ).state_dict()
            state["contrastive_target"] = self.contrastive.target.state_dict()
            state["contrastive_proj_target"] = self.contrastive.proj_target.state_dict()
        state["opt"] = self.opt.state_dict()
        state["cfg"] = self.cfg
        torch.save(state, self.ckpt_dir / name)

    def load_ckpt(self, path: str) -> None:
        if not Path(path).exists():
            if self.dist.is_main:
                print(f"[Trainer] no resume ckpt at {path}, fresh start")
            return
        st = torch.load(path, map_location="cpu", weights_only=False)
        self._encoder_inner().load_state_dict(st["encoder"])
        (self.msm_head.module if isinstance(self.msm_head, DDP) else self.msm_head).load_state_dict(st["msm_head"])
        self.opt.load_state_dict(st["opt"])
        self.step = st["step"]
        if self.contrastive is not None and "contrastive_target" in st:
            (self.contrastive.proj_online.module if isinstance(self.contrastive.proj_online, DDP) else self.contrastive.proj_online).load_state_dict(st["contrastive_proj_online"])
            (self.contrastive.predict.module if isinstance(self.contrastive.predict, DDP) else self.contrastive.predict).load_state_dict(st["contrastive_predict"])
            self.contrastive.target.load_state_dict(st["contrastive_target"])
            self.contrastive.proj_target.load_state_dict(st["contrastive_proj_target"])
        if self.dist.is_main:
            print(f"[Trainer] resumed from step {self.step}")

    # ---- loop ----

    def fit(self) -> None:
        t = self.cfg["train"]
        total = t.get("total_steps", t.get("steps", 100))
        log_int = t.get("log_interval", 50)
        ckpt_int = t.get("ckpt_interval", 2000)

        t0 = time.time()
        epoch = 0
        while self.step < total:
            if self.train_sampler is not None:
                self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                log = self._step_batch(batch)
                if self.dist.is_main and self.step % log_int == 0:
                    dt = time.time() - t0
                    msg = " ".join(f"{k}={v:.4g}" for k, v in log.items())
                    print(f"[step {self.step:>7d}/{total}] {msg}  ({dt:.1f}s)")
                if self.dist.is_main and self.step > 0 and self.step % ckpt_int == 0:
                    self.save_ckpt("last.pt")
                self.step += 1
                if self.step >= total:
                    break
            epoch += 1
        if self.dist.is_main:
            self.save_ckpt("last.pt")
            print(f"[Trainer] done. {total} steps in {time.time()-t0:.1f}s")
