#!/bin/bash
#SBATCH -J fa_build
#SBATCH -p amd_a100nv_8
#SBATCH --comment pytorch
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH -o /scratch/x3397a12/resona/logs/fa_build_%j.log

set -e
source /scratch/x3397a12/miniforge/etc/profile.d/conda.sh
conda activate resona

export PIP_CACHE_DIR=/scratch/$USER/.cache/pip
export TORCH_CUDA_ARCH_LIST="8.0"          # A100 only — faster build
export MAX_JOBS=4                            # avoid OOM during nvcc
export FLASH_ATTENTION_FORCE_BUILD=TRUE

echo "=== nvcc / torch ==="
nvcc --version | tail -3
python -c "import torch; print(torch.__version__, torch.version.cuda)"

echo "=== install flash-attn (build from source) ==="
pip install packaging ninja
pip install flash-attn --no-build-isolation -v

echo "=== verify ==="
python -c "import flash_attn; print('flash_attn', flash_attn.__version__)"
echo "=== DONE ==="
