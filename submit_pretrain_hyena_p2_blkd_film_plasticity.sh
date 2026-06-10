#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:4
#SBATCH --job-name=varc_plast
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=5-00:00:00
#SBATCH --mem=192G
#SBATCH --output=logs/varc_plast_%A.out
#SBATCH --exclude=hipster-cn011  # broken capacity node: launches fail there (harmless on performance)

# Phase 5 (plasticity-preserving pretraining): re-pretrain the SOTA-base FiLM+BlockDiag p=2
# checkpoint changing ONLY the plasticity regularizer, to test whether resisting
# over-specialization at pretrain time raises the TTT ceiling (the ep40>ep100 over-spec
# inverted-U says full convergence destroys TTT-adaptability). Snapshots at {20,40,60,80,100}
# let each setting's whole epoch curve be TTT-screened cheaply afterward.
#
# Identical to submit_pretrain_hyena_patch2_blockdiag_film_lr1e3_4gpu.sh EXCEPT:
#   --weight-decay <WD>  (baseline used 0) and snapshot dumping.
# Launch via --export, e.g.:
#   sbatch -J varc_plast_wd05 -o logs/varc_plast_wd05_%A.out \
#     --export=ALL,WD=0.05,TAG=wd05 submit_pretrain_hyena_p2_blkd_film_plasticity.sh
# Smoke test (verify it trains + snapshots, ~minutes): add EPOCHS=2,SAVE_EPOCHS=1,2 NPROC=2.

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

: "${WD:?must set WD via --export}"; : "${TAG:?must set TAG via --export}"
EPOCHS=${EPOCHS:-100}; SAVE_EPOCHS=${SAVE_EPOCHS:-20,40,60,80,100}; NPROC=${NPROC:-4}
# batch/grad-accum default to the rtx_6000 (48GB) recipe; override for L4 (24GB): BATCH=32 GRAD_ACCUM=2
# keeps effective batch = NPROC * BATCH * GRAD_ACCUM = 256.
BATCH=${BATCH:-64}; GRAD_ACCUM=${GRAD_ACCUM:-1}
SNAP_DIR="saves/offline_train_Hyena_p2_blkd_film_${TAG}_snap"
mkdir -p logs "${SNAP_DIR}" wandb
MASTER_PORT=$(( 29500 + (SLURM_JOB_ID % 1000) ))
echo "Plasticity pretrain: TAG=$TAG WD=$WD EPOCHS=$EPOCHS SAVE_EPOCHS=$SAVE_EPOCHS NPROC=$NPROC -> $SNAP_DIR"

# Optional resume (e.g. to migrate a capacity/L4 run onto performance/rtx_6000_ada without
# losing progress): RESUME_CKPT=<path> restores model+optimizer+scheduler+epoch and continues
# the cosine schedule from where it left off. Unset -> fresh run (default behaviour).
RESUME_ARGS=()
if [ -n "${RESUME_CKPT:-}" ]; then
  echo "Resuming from checkpoint: ${RESUME_CKPT}"
  RESUME_ARGS+=(--resume-checkpoint "${RESUME_CKPT}")
fi

torchrun --nproc_per_node=${NPROC} --master-port=$MASTER_PORT offline_train_ARC.py \
  --epochs ${EPOCHS} \
  --batch-size ${BATCH} \
  --grad-accum-steps ${GRAD_ACCUM} \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 1e-3 \
  --weight-decay ${WD} \
  --include-rearc \
  --num-colors 12 \
  --data-root "raw_data/ARC-AGI" \
  --train-split "training" \
  --wandb-project "VisionARC" \
  --wandb-run-name "varc_hyena_p2_blkd_film_${TAG}" \
  --save-path "${SNAP_DIR}/checkpoint_final.pt" \
  --best-save-path "${SNAP_DIR}/checkpoint_best.pt" \
  --periodic-save-dir "${SNAP_DIR}" \
  --save-epochs "${SAVE_EPOCHS}" \
  --lr-scheduler "cosine" \
  --architecture "hyena" \
  --hyena-config "/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag_film.py" \
  --no-compile \
  --vis-every 10 \
  --distributed \
  --use-wandb \
  "${RESUME_ARGS[@]}"

echo "Done: $(date)"
