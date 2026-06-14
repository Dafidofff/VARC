#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --job-name=varc_mamba
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=5-00:00:00
#SBATCH --mem=192G
#SBATCH --output=logs/varc_mamba_%A.out
#SBATCH --exclude=hipster-cn011  # broken capacity node (harmless on performance)

# Pure-full Mamba2 pretrain — the mamba arm of the VARC <-> Hyena comparison.
# patch=2 + AdaLN-Zero + bidirectional Mamba2 mixer (cfg_mamba_p2_bidir.py), embed_dim=384,
# 12 blocks, eff. batch 4x64=256 (matches every other run). Snapshots {20,40,60,80,100}
# so the whole epoch curve is TTT-screenable (over-spec inverted-U lever).
#
# Requires mamba_ssm + causal_conv1d built from source in nvsubq (RHEL8/glibc 2.28 — prebuilt
# wheels need glibc 2.32 and do NOT work here). --no-compile: Mamba2 selective-scan kernels
# are not torch.compile/inductor friendly.
#
# Smoke test (verify it trains + snapshots, ~minutes): add --export with EPOCHS=2,SAVE_EPOCHS=1,2,NPROC=1.

echo "This job is running on node: $SLURM_NODELIST"; echo "Start: $(date)"
set -eo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."
export WANDB_DIR="${PWD}/wandb"; export WANDB_DATA_DIR="${PWD}/wandb"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR=/tmp/triton_nocache_${SLURM_JOB_ID}
export OMP_NUM_THREADS=1
cd /home/dwessel/code/VARC

EPOCHS=${EPOCHS:-100}; SAVE_EPOCHS=${SAVE_EPOCHS:-20,40,60,80,100}; NPROC=${NPROC:-4}
BATCH=${BATCH:-64}; GRAD_ACCUM=${GRAD_ACCUM:-1}
SNAP_DIR="${RUN_DIR:-saves/offline_train_Mamba_p2}"
mkdir -p logs "${SNAP_DIR}" wandb
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
echo "Mamba pretrain: EPOCHS=$EPOCHS SAVE_EPOCHS=$SAVE_EPOCHS NPROC=$NPROC -> $SNAP_DIR"

torchrun --nproc_per_node=${NPROC} --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --grad-accum-steps ${GRAD_ACCUM} \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_mamba_p2_bidir" \
  --save-path "${SNAP_DIR}/checkpoint_final.pt" \
  --best-save-path "${SNAP_DIR}/checkpoint_best.pt" \
  --periodic-save-dir "${SNAP_DIR}" \
  --save-epochs "${SAVE_EPOCHS}" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_mamba_p2_bidir.py" \
  --no-compile \
  --vis-every 10 \
  --distributed \
  --use-wandb

echo "Done: $(date)"
