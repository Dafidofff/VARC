#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=varc_pretrain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=8
#SBATCH --time=1-00:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/varc_pretrain_%A.out

echo "This job is running on node: $SLURM_NODELIST"

set -eo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export WANDB_DIR="${PWD}/wandb"
export WANDB_DATA_DIR="${PWD}/wandb"

# CUDA (needed by torch.compile / inductor)
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

# Memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /home/dwessels2/code/VARC
mkdir -p logs saves/offline_train_ViT wandb

# Original paper uses 8 GPUs with batch-size 32 (effective batch 256).
# We use 4 GPUs with batch-size 64 to keep the same effective batch size.
torchrun --nproc_per_node=4 offline_train_ARC.py \
  --epochs 100 \
  --depth 10 \
  --batch-size 64 \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --embed-dim 512 \
  --num-heads 8 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_pretrain_baseline" \
  --save-path "saves/offline_train_ViT/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_ViT/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "vit" \
  --vis-every 50 \
  --distributed \
  --use-wandb
