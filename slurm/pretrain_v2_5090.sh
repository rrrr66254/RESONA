#!/bin/bash
# RESONA-v2 pretrain on sltrain 5090 (single GPU bash launcher).
# Sub-pose ST-GCN + RoPE Transformer + multi-objective (MSM + ForwardPred).
# Usage: bash slurm/pretrain_v2_5090.sh [GPU_ID]   (default GPU_ID=2)

set -e
GPU_ID=${1:-2}
RESONA_ROOT=/home/sltrain/RESONA
CKPT_ROOT=/mnt/synology_nas_00/junkim/RESONA_ckpt
RUN_TS=$(date +%s)
LOG=/home/sltrain/log_resona_v2_${RUN_TS}.log

cd "$RESONA_ROOT"
source /home/sltrain/miniconda3/etc/profile.d/conda.sh
conda activate hmc_vlm

export CUDA_VISIBLE_DEVICES=${GPU_ID}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export PYTHONPATH="$RESONA_ROOT:$PYTHONPATH"

mkdir -p "$CKPT_ROOT/v2"

echo "=== RESONA-v2 pretrain on GPU ${GPU_ID} ===" | tee -a "$LOG"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader 2>&1 | tee -a "$LOG"

exec python -u scripts/pretrain.py \
    --config configs/pretrain_v2_5090.yaml \
    --ckpt-dir "$CKPT_ROOT/v2" \
    --run-dir "$CKPT_ROOT/runs/v2_${RUN_TS}" \
    --resume "$CKPT_ROOT/v2/last.pt" \
    2>&1 | tee -a "$LOG"
