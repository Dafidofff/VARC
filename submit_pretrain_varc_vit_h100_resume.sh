#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --job-name=varc_pretrain_resume
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

# ─── Config ──────────────────────────────────────────────────────────────────
cd /home/dwessels2/code/VARC
mkdir -p logs saves/offline_train_ViT wandb

TARGET_EPOCHS=200
WANDB_RUN_ID="cwkfvy5p"
LATEST_CKPT="saves/offline_train_ViT/checkpoint_latest.pt"
BEST_CKPT="saves/offline_train_ViT/checkpoint_best.pt"

# Pick the most recent checkpoint available
if [ -f "$LATEST_CKPT" ]; then
    RESUME_CKPT="$LATEST_CKPT"
    echo "Resuming from latest checkpoint: $RESUME_CKPT"
elif [ -f "$BEST_CKPT" ]; then
    RESUME_CKPT="$BEST_CKPT"
    echo "No latest checkpoint found; resuming from best checkpoint: $RESUME_CKPT"
else
    echo "ERROR: No checkpoint found to resume from." >&2
    exit 1
fi

# ─── Run ─────────────────────────────────────────────────────────────────────
torchrun --nproc_per_node=4 offline_train_ARC.py \
  --epochs ${TARGET_EPOCHS} \
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
  --wandb-id "${WANDB_RUN_ID}" \
  --save-path "saves/offline_train_ViT/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_ViT/checkpoint_best.pt" \
  --latest-save-path "${LATEST_CKPT}" \
  --lr-scheduler "cosine" \
  --architecture "vit" \
  --vis-every 50 \
  --distributed \
  --use-wandb \
  --resume-checkpoint "${RESUME_CKPT}" || true

# ─── Autoresume ──────────────────────────────────────────────────────────────
# Check how many epochs were completed; resubmit if target not yet reached.
if [ -f "$LATEST_CKPT" ]; then
    COMPLETED_EPOCH=$(python3 -c "
import torch, sys
ckpt = torch.load('${LATEST_CKPT}', map_location='cpu', weights_only=False)
print(ckpt.get('epoch', 0))
" 2>/dev/null || echo "0")

    echo "Completed epoch: ${COMPLETED_EPOCH} / ${TARGET_EPOCHS}"

    if [ "${COMPLETED_EPOCH}" -lt "${TARGET_EPOCHS}" ]; then
        echo "Target not reached — resubmitting job..."
        sbatch "$(realpath "$0")"
    else
        echo "Training complete at epoch ${COMPLETED_EPOCH}."
    fi
fi
