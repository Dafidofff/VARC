#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --job-name=smoke_p2_blkd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:30:00
#SBATCH --mem=32G
#SBATCH --output=logs/smoketest_hyena_patch2_blockdiag_%A.out

echo "This job is running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/dwessel/code/VARC
mkdir -p logs saves/smoketest_Hyena_patch2_blockdiag

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# 1× GPU smoke test — verifies the full train+val loop runs without error.
# Effective batch size: 1 GPU × batch 16 × grad-accum 1 = 16 (intentionally small for speed).
# The production run uses 4 GPUs × batch 64 × grad-accum 1 = 256.
# No --use-wandb: keep smoke runs out of WandB.
torchrun --nproc_per_node=1 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 2 \
  --batch-size 16 \
  --grad-accum-steps 1 \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --save-path "saves/smoketest_Hyena_patch2_blockdiag/checkpoint_final.pt" \
  --best-save-path "saves/smoketest_Hyena_patch2_blockdiag/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag.py" \
  --no-compile \
  --vis-every 1 \
  --distributed

echo "Done: $(date)"
