#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --job-name=varc_hyena_p2_film
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/varc_hyena_patch2_film_%A.out

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
mkdir -p logs saves/offline_train_Hyena_patch2_film_lr1e3 wandb

MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))

# 4× RTX 6000 Ada (48 GB each).
# Effective batch size: 4 GPUs × batch 64 × grad-accum 1 = 256  ← matches winning run 22255864.
# Kernel: FiLM-conditioned SIRENKernelND (task-tok modulates SIREN hidden layers via AdaLN→FiLM).
# patch_size=2 set via Patchify in the config; --image-size 64 → 32×32 = 1024 tokens.
# --no-compile required: circular FFT uses complex64, incompatible with torch.compile/inductor.
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
  --wandb-run-name "varc_hyena_patch2_film_lr1e3" \
  --save-path "saves/offline_train_Hyena_patch2_film_lr1e3/checkpoint_final.pt" \
  --best-save-path "saves/offline_train_Hyena_patch2_film_lr1e3/checkpoint_best.pt" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_film.py" \
  --no-compile \
  --vis-every 10 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
