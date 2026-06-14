#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --job-name=varc_mamba_smoke
#SBATCH --time=0-00:15:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/varc_mamba_smoke_%A.out
set -eo pipefail
source ~/miniforge3/etc/profile.d/conda.sh; conda activate nvsubq
export PYTHONPATH="."
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
cd /home/dwessel/code/VARC
echo "node $SLURM_NODELIST $(date)"
python scripts/smoke_mamba.py
echo "done $(date)"
