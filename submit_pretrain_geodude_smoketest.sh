#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:8
#SBATCH --job-name=varc_pretrain_smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=80G
#SBATCH --output=logs/varc_pretrain_smoke_%A.out

echo "Pretrain smoketest running on node: $SLURM_NODELIST"
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
mkdir -p logs saves/offline_train_ViT wandb

# Smoketest: 2 epochs on 2 GPUs to verify the pipeline runs end-to-end.
# capacity GPUs are ~22 GiB (vs 80 GiB H100); batch 16 fits the 1024-token ViT attention.
torchrun --nproc_per_node=8 offline_train_ARC.py \
  --epochs 2 \
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
  --save-path "saves/offline_train_ViT/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_ViT/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "vit" \
  --vis-every 1 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
