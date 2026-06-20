#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --array=0-39
#SBATCH --exclude=hipster-cn011  # broken node: /local_scratch container ns missing -> launch failures

# SNAPSHOT-checkpoint TTT harness (backlog #6: earlier pretrain checkpoint).
# Tests the over-specialization hypothesis directly: TTT an UNDER-trained BlockDiag p=2
# checkpoint and see if it TTTs BETTER than the fully-trained one (baseline 36.25%).
# Recipe is held at the BlockDiag baseline (unfrozen, lr1e3, const) so the ONLY variable
# is the pretrain epoch of the checkpoint. NOTE: plain BlockDiag config (NO FiLM here).
#
# Launch via sbatch --export, one snapshot epoch per launch, e.g.:
#   sbatch -J ttt_snap_ep20 -o logs/ttt_autores_snap_ep20_%A_%a.out \
#          --export=ALL,SWEEP_NAME=ep20,CKPT=saves/offline_train_Hyena_patch2_blockdiag_lr1e3_snap/checkpoint_epoch20.pt \
#          submit_ttt_autores_snap.sh
# Score with: python scripts/score_one.py ttt_autores/snap_ep20
#
# Tunable env vars (defaults = BlockDiag baseline recipe + epoch20 snapshot):
#   SWEEP_NAME (required)  CKPT=...checkpoint_epoch20.pt  LR=1e-3  SCHED=none  WD=0
#   NUM_ATTEMPTS=10  EPOCHS=100  TTT_NUM_EACH=2  FREEZE=0  BATCH=8

echo "TTT autores snapshot 400-confirm '${SWEEP_NAME}' array ${SLURM_ARRAY_TASK_ID}/39 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

: "${SWEEP_NAME:?must set SWEEP_NAME via --export}"
CKPT=${CKPT:-saves/offline_train_Hyena_patch2_blockdiag_lr1e3_snap/checkpoint_epoch20.pt}
LR=${LR:-1e-3}; SCHED=${SCHED:-none}; WD=${WD:-0}; NUM_ATTEMPTS=${NUM_ATTEMPTS:-10}
EPOCHS=${EPOCHS:-100}; TTT_NUM_EACH=${TTT_NUM_EACH:-2}; FREEZE=${FREEZE:-0}; BATCH=${BATCH:-8}

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag.py"
EVAL_SAVE_NAME="ttt_autores/snap_${SWEEP_NAME}"
STRIDE=40
JOB_IDX=${SLURM_ARRAY_TASK_ID}

FREEZE_ARG=""; [ "$FREEZE" = "1" ] && FREEZE_ARG="--freeze-hyena-filters"
echo "Recipe: CKPT=$CKPT LR=$LR SCHED=$SCHED WD=$WD NUM_ATTEMPTS=$NUM_ATTEMPTS EPOCHS=$EPOCHS TTT_NUM_EACH=$TTT_NUM_EACH FREEZE=$FREEZE BATCH=$BATCH"

# All 400 ARC-1 evaluation tasks, sorted (deterministic). The 100 screened tasks are
# skipped via skip-if-done so only the remaining ~300 run.
mapfile -t file_names < <(cd raw_data/ARC-AGI/data/evaluation && ls *.json | sed 's/\.json$//' | sort)
echo "Total tasks: ${#file_names[@]}"

i=0
for file_name in "${file_names[@]}"; do
  if (( i % STRIDE == JOB_IDX )); then
    if [ -f "outputs/${EVAL_SAVE_NAME}_attempt_0/${file_name}_predictions.json" ]; then
      echo "[${JOB_IDX}] skipping ${file_name} (already done)"
    else
      echo "[${JOB_IDX}] task ${file_name} (idx ${i})"
      python test_time_train_ARC.py \
        --epochs "${EPOCHS}" \
        --batch-size "${BATCH}" \
        --image-size 64 \
        --learning-rate "${LR}" \
        --weight-decay "${WD}" \
        --num-colors 12 \
        --resume-checkpoint "${CKPT}" \
        --patch-size 2 \
        --lr-scheduler "${SCHED}" \
        --train-split "eval_color_permute_ttt_9/${file_name}" \
        --data-root "raw_data/ARC-AGI" \
        --eval-split "eval_color_permute_ttt_9/${file_name}" \
        --resume-skip-task-token \
        --architecture hyena \
        --hyena-config "${HYENA_CONFIG}" \
        --eval-save-name "${EVAL_SAVE_NAME}" \
        --num-attempts "${NUM_ATTEMPTS}" \
        --ttt-num-each "${TTT_NUM_EACH}" \
        ${FREEZE_ARG} \
        --no-compile
    fi
  fi
  i=$((i + 1))
done

echo "[${JOB_IDX}] Done: $(date)"
