#!/bin/bash
#SBATCH -J resona_pretrain
#SBATCH -p amd_a100nv_8
#SBATCH --comment pytorch
#SBATCH --gres=gpu:8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=8
#SBATCH -t 48:00:00
#SBATCH -o /scratch/x3397a12/resona/logs/pretrain_%j.log

set -e
cd /scratch/x3397a12/resona/code

source /scratch/x3397a12/miniforge/etc/profile.d/conda.sh
conda activate resona

export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export HF_HOME=/scratch/$USER/.cache/hf
export PYTHONNOUSERSITE=1

# DDP / NCCL
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS=8

echo "=== job $SLURM_JOB_ID on $(hostname) ==="
nvidia-smi
echo "MASTER_ADDR=$MASTER_ADDR"

CONFIG=${CONFIG:-configs/pretrain_300m.yaml}
RESUME=${RESUME:-/scratch/x3397a12/resona/ckpt/last.pt}

srun python scripts/pretrain.py \
    --config $CONFIG \
    --resume $RESUME \
    --ckpt-dir /scratch/x3397a12/resona/ckpt \
    --run-dir /scratch/x3397a12/resona/runs/$SLURM_JOB_ID

echo "=== DONE ==="
