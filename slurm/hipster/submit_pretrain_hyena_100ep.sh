#!/bin/bash
#SBATCH --partition=gpu_h100
#SBATCH --gpus=4
#SBATCH --gpus-per-task=1
#SBATCH --job-name=varc_hyena_100ep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=1-02:00:00
#SBATCH --mem=192G
#SBATCH --output=logs/varc_hyena_%A.out

echo "This job is running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."

export WANDB_DIR="${PWD}/wandb"
export WANDB_DATA_DIR="${PWD}/wandb"

export PATH="/usr/local/cuda-13.0/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-13.0/lib64:${LD_LIBRARY_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/dwessels2/code/VARC
mkdir -p logs saves/offline_train_Hyena_100ep wandb

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# gpu_h100 partition: 4×H100.
# Effective batch size: 4 GPUs × 16 batch × 4 grad-accum = 256.
torchrun --nproc_per_node=4 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 100 \
  --batch-size 16 \
  --grad-accum-steps 4 \
  --image-size 64 \
  --learning-rate 3e-4 \
  --weight-decay 0 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_hyena_100ep_lr3e4" \
  --save-path "saves/offline_train_Hyena_100ep/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_Hyena_100ep/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/gpfs/home2/dwessels2/code/nvSubquadratic-private/examples/arc/cfg_hyena_varc_replica_adaln_patch1.py" \
  --compile-mode max-autotune-no-cudagraphs \
  --vis-every 50 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
