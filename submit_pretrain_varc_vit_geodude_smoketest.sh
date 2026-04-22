#!/bin/bash
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=2
#SBATCH --job-name=varc_pretrain_smoke
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=0-04:00:00
#SBATCH --mem=96G
#SBATCH --output=slurm/varc_pretrain_smoke_%A.out

echo "This job is running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

# ─── Environment ─────────────────────────────────────────────────────────────
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export WANDB_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb
export WANDB_DATA_DIR=/ivi/zfs/s0/original_homes/dwessel/wandb

# CUDA (needed by torch.compile / inductor)
export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

# Memory
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

# ─── Run ─────────────────────────────────────────────────────────────────────
cd /home/dwessel/code/VARC
mkdir -p slurm saves/offline_train_ViT wandb

# Smoketest: 3 epochs on 2 GPUs to verify the training pipeline runs end-to-end.
# Paper uses 8×H200 batch 32 (effective 256); we adapt to 2 GPUs batch 64 (effective 128).
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
torchrun --nproc_per_node=2 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 3 \
  --depth 10 \
  --batch-size 16 \
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
  --wandb-run-name "varc_pretrain_smoketest_geodude" \
  --save-path "saves/offline_train_ViT/checkpoint_final_smoke.pt" \
  --best-save-path "saves/offline_train_ViT/checkpoint_best_smoke.pt" \
  --lr-scheduler "cosine" \
  --architecture "vit" \
  --vis-every 1 \
  --distributed \
  --use-wandb

echo "Smoketest done: $(date)"
