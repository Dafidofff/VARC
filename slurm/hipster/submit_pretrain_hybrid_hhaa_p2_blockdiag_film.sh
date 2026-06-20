#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --job-name=varc_hybrid_hhaa
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=5-00:00:00
#SBATCH --mem=192G
#SBATCH --output=logs/varc_hybrid_hhaa_%A.out

echo "This job is running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export WANDB_DIR="${PWD}/wandb"
export WANDB_DATA_DIR="${PWD}/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/dwessel/code/VARC
mkdir -p logs saves/offline_train_Hyena_hybrid_hhaa_p2_blockdiag_film wandb

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# HYBRID Hyena/Attention pretraining — HHAA (paired) interleaving.
# 12 blocks: H H A A H H A A H H A A (6 Hyena BlockDiag+FiLM SOTA blocks, 6 self-attention blocks).
# 4× RTX 6000 Ada (48 GB). patch=2 -> 32x32=1024 tokens; effective batch 4x64=256 (matches all runs).
# Attention blocks: 8 heads, 2D RoPE (32x32), cosine QK-norm. Hyena blocks: BlockDiag-w0 + FiLM SIREN.
# --no-compile required: Hyena circular FFT uses complex64, incompatible with torch.compile/inductor.
torchrun --nproc_per_node=4 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 100 \
  --batch-size 64 \
  --grad-accum-steps 1 \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_hybrid_hhaa_p2_blockdiag_film" \
  --save-path "saves/offline_train_Hyena_hybrid_hhaa_p2_blockdiag_film/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_Hyena_hybrid_hhaa_p2_blockdiag_film/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hybrid_hhaa_p2_blockdiag_film.py" \
  --no-compile \
  --vis-every 10 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
