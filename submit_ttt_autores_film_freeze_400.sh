#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --job-name=ttt_autores_filmfrz400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/ttt_autores_film_freeze_400_%A_%a.out
#SBATCH --array=0-39

# 400-task CONFIRMATION of experiment #14 (film_freeze).
# The 100-task screen scored 49.0% Pass@1 / 56.0% Pass@2 -- the best screen by +6pp,
# well above the established 43% noise band -- so the loop requires a full-400 confirm.
# Recipe is IDENTICAL to submit_ttt_autores_film_freeze.sh (FiLM+AdaLN p=2 ckpt + freeze
# -filters + baseline lr1e3_const); only the task set (all 400) and array layout change.
#
# Reuses EVAL_SAVE_NAME=ttt_autores/film_freeze: the 100 screen predictions already on disk
# are skipped (skip-if-done), so only the remaining ~300 tasks run. After it finishes,
#   python scripts/score_one.py ttt_autores/film_freeze
# scores all 400 (score_one derives the task set from the attempt_0 dir).
#
# 40-way array, STRIDE=40 -> 10 tasks/job; ~14 min/task (freeze is cheap) and ~7.5 fresh
# tasks/job after skips => ~105 min, well within the 4h wall. Skip-if-done makes requeue safe.

echo "TTT autores film_freeze 400-confirm array ${SLURM_ARRAY_TASK_ID}/39 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag_film.py"
EVAL_SAVE_NAME="ttt_autores/film_freeze"
STRIDE=40
JOB_IDX=${SLURM_ARRAY_TASK_ID}

# All 400 ARC-1 evaluation tasks, sorted (deterministic). The 100 screened tasks are a
# subset and are skipped via skip-if-done.
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
        --epochs 100 \
        --batch-size 8 \
        --image-size 64 \
        --learning-rate 1e-3 \
        --weight-decay 0 \
        --num-colors 12 \
        --resume-checkpoint "saves/offline_train_Hyena_patch2_blockdiag_film_lr1e3/checkpoint_best.pt" \
        --patch-size 2 \
        --lr-scheduler none \
        --train-split "eval_color_permute_ttt_9/${file_name}" \
        --data-root "raw_data/ARC-AGI" \
        --eval-split "eval_color_permute_ttt_9/${file_name}" \
        --resume-skip-task-token \
        --architecture hyena \
        --hyena-config "${HYENA_CONFIG}" \
        --eval-save-name "${EVAL_SAVE_NAME}" \
        --num-attempts 10 \
        --ttt-num-each 2 \
        --freeze-hyena-filters \
        --no-compile
    fi
  fi
  i=$((i + 1))
done

echo "[${JOB_IDX}] Done: $(date)"
