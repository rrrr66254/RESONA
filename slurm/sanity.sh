#!/bin/bash
#SBATCH -J resona_sanity
#SBATCH -p amd_a100nv_8
#SBATCH --comment pytorch
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 00:30:00
#SBATCH -o /scratch/x3397a12/resona/logs/sanity_%j.log

set -e
cd /scratch/x3397a12/resona/code

source /scratch/x3397a12/miniforge/etc/profile.d/conda.sh
conda activate resona

# Cache redirects (in case .bashrc not sourced)
export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export HF_HOME=/scratch/$USER/.cache/hf
export PYTHONNOUSERSITE=1

echo "=== env ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); print('device', torch.cuda.get_device_name(0))"
nvidia-smi

echo "=== verify_data ==="
python scripts/verify_data.py --data-dir /scratch/x3397a12/resona/data

echo "=== mini forward pass ==="
python scripts/pretrain.py --config configs/sanity.yaml --steps 10

echo "=== DONE ==="
