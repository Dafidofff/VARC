#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:8
#SBATCH --job-name=varc_hyena_p1_blkd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --time=5-00:00:00
#SBATCH --mem=256G
#SBATCH --output=logs/varc_hyena_patch1_blockdiag_%A.out

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
mkdir -p logs saves/offline_train_Hyena_patch1_blockdiag_lr1e3 wandb

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# 8× RTX 6000 Ada (48 GB each).
# patch=1 → seq_len=4096; batch=32 OOMs, batch=16 fits at ~27 GB/GPU.
# Effective batch size: 8 GPUs × batch 16 × grad-accum 2 = 256  ← matches all other runs.
# Kernel: BlockDiagonalLearnableOmegaSIRENKernelND (num_blocks=8, ω₀∈[1,12] linear, learnable scale).
# Mask: BlockAlignedGaussianModulationND.
# --no-compile required: circular FFT uses complex64, incompatible with torch.compile/inductor.
torchrun --nproc_per_node=8 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 100 \
  --batch-size 16 \
  --grad-accum-steps 2 \
  --image-size 64 \
  --patch-size 1 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_hyena_patch1_blockdiag_lr1e3" \
  --save-path "saves/offline_train_Hyena_patch1_blockdiag_lr1e3/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_Hyena_patch1_blockdiag_lr1e3/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch1_circular_adaln_blockdiag.py" \
  --no-compile \
  --vis-every 10 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
