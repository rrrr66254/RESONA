# RESONA

**RE**presentation for **S**keleton-based **O**mnilingual **N**eural **A**rchitecture

Multilingual sign language foundation model pretrained on a unified
RESONA-77 skeleton corpus (14+ languages, ~700M frames target).

> *For some, sound travels through the air.*
> *For others, it blooms from the fingertips.*
> *The world where that sound reaches is not yet wide enough.*
> ***RESONA*** — *where both resonances meet.*

## Status

🚧 Pretraining infrastructure setup in progress (KISTI Neuron, A100×8).

## Layout

```
resona/                 # core package
  data/                 # UnifiedSignDataset, augmentation, normalization
  models/               # Transformer encoder, MSM/contrastive heads
  losses.py
  trainer.py
configs/                # YAML configs (sanity / pretrain_300m / ...)
scripts/                # Entry points (pretrain.py, verify_data.py, ...)
slurm/                  # KISTI Neuron Slurm job templates
```

## Pretraining objective

Dual self-supervised:

1. **Masked Skeleton Modeling (MSM)** — joint-level mask + L2 reconstruction
2. **Temporal Contrastive** — EMA target network, two-view clip alignment (BYOL-style)

## Target

- **Model**: 24-layer Transformer, d=1024, 16 heads, ~300M params
- **Data**: 13 H5 files (RESONA-77 schema), ~121M frames currently, ~700M target
- **Compute**: KISTI Neuron `amd_a100nv_8` (A100 80GB×8), chained Slurm jobs

## License

MIT (planned, on public release)
