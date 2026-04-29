#!/bin/bash
# RESONA-100M pretrain on sltrain 5090 (single GPU, bash launcher — not slurm).
# Usage:
#   bash slurm/pretrain_100m_5090.sh [GPU_ID]
# Default GPU_ID=2 (assuming GPU 0/1/3 used by other users).

set -e
GPU_ID=${1:-2}
RESONA_ROOT=/home/sltrain/RESONA          # code (fast read)
CKPT_ROOT=/mnt/synology_nas_00/junkim/RESONA_ckpt  # ckpt (large, NAS)
RUN_TS=$(date +%s)
LOG=/home/sltrain/log_resona_100m_${RUN_TS}.log

cd "$RESONA_ROOT"
source /home/sltrain/miniconda3/etc/profile.d/conda.sh
conda activate hmc_vlm                     # reuse existing env (sltrain disk tight)

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RESONA_ROOT:$PYTHONPATH"

mkdir -p "$CKPT_ROOT/100m"

echo "=== RESONA-100M pretrain on GPU ${GPU_ID} ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG"

exec python -u scripts/pretrain.py \
    --config configs/pretrain_100m.yaml \
    --ckpt-dir "$CKPT_ROOT/100m" \
    --run-dir "$CKPT_ROOT/runs/100m_${RUN_TS}" \
    --resume "$CKPT_ROOT/100m/last.pt" \
    2>&1 | tee -a "$LOG"
