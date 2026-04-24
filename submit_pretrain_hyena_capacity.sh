#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:8
#SBATCH --job-name=varc_hyena
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=3-00:00:00
#SBATCH --mem=192G
#SBATCH --output=logs/varc_hyena_%A.out

echo "This job is running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source /home/dwessel/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."

export WANDB_DIR="${PWD}/wandb"
export WANDB_DATA_DIR="${PWD}/wandb"

export PATH="/usr/local/cuda-12.3/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.3/lib64:${LD_LIBRARY_PATH:-}"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1

cd /home/dwessel/code/VARC
mkdir -p logs saves/offline_train_Hyena wandb

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

torchrun --nproc_per_node=8 --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs 500 \
  --batch-size 16 \
  --image-size 64 \
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
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py" \
  --no-compile \
  --vis-every 50 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
