#!/bin/bash
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=4
#SBATCH --job-name=varc_hyena
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=3-00:00:00
#SBATCH --mem=192G
#SBATCH --output=slurm/varc_hyena_%A.out

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
mkdir -p slurm saves/offline_train_Hyena wandb

HYENA_CFG="/home/dwessel/code/nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py"
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# Hyena with circular FFT does not support torch.compile (complex64 limitation).
# Batch 16 × 2 GPUs = effective 32; fits within 24GB A5000 VRAM.
torchrun --nproc_per_node=4 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 500 \
  --batch-size 16 \
  --image-size 32 \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_hyena_geodude" \
  --save-path "saves/offline_train_Hyena/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_Hyena/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "${HYENA_CFG}" \
  --no-compile \
  --vis-every 50 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
