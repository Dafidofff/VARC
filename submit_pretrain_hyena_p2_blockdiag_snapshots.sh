#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --job-name=varc_hyena_p2_snap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/varc_hyena_p2_blockdiag_snap_%A.out

# Re-pretrain of Hyena BlockDiag p=2 (reproduces job 268829, val 82.93%) but
# saving SNAPSHOT checkpoints at epochs 20/40/60/80 in addition to best/final.
# Purpose: the autoresearch loop has exhausted every TTT-recipe scalar (LR,
# schedule, epochs, num-attempts all <= baseline 42% screen / 36% full-400),
# strongly implying the bottleneck is the OVER-SPECIALIZED pretrained
# representation, not the TTT recipe. These early checkpoints let us TTT from
# under-trained (lower val_acc) filters to test whether they TTT *better* — the
# decisive test of the over-specialization hypothesis.
#
# IMPORTANT: saves into a NEW dir (..._snap) so the canonical checkpoint_best.pt
# (the fixed TTT start point) is never overwritten. Uses the varc_configs/ path
# (the examples/arc/ copy gets wiped by nvSubq branch switches and is MISSING).
# Runs on the `performance` GPU partition, separate from the `capacity` L4 pool
# the TTT screens use — so it does not block continued TTT experiments.

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
SNAP_DIR="saves/offline_train_Hyena_patch2_blockdiag_lr1e3_snap"
mkdir -p logs "${SNAP_DIR}" wandb

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# Same recipe as the original BlockDiag p=2 run (eff. BS 256, cosine, lr1e3,
# --no-compile for complex64 FFT). --periodic-save-dir + --save-epochs add the
# intermediate snapshots.
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
  --wandb-run-name "varc_hyena_patch2_blockdiag_lr1e3_snap" \
  --save-path "${SNAP_DIR}/checkpoint_final.pt" \
  --best-save-path "${SNAP_DIR}/checkpoint_best.pt" \
  --periodic-save-dir "${SNAP_DIR}" \
  --save-epochs "20,40,60,80" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag.py" \
  --no-compile \
  --vis-every 10 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
