#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --job-name=ttt_autores_filmunfrz400
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/ttt_autores_film_unfrozen_400_%A_%a.out
#SBATCH --array=0-39
#SBATCH --exclude=hipster-cn011  # broken node: /local_scratch container ns missing -> launch failures

# 400-task CONFIRMATION of experiment #17 (film_unfrozen) -- the new best screen at
# 51.0% Pass@1 / 57.0% Pass@2, above film_freeze (49%/56%). Recipe IDENTICAL to
# submit_ttt_autores_film_unfrozen.sh (FiLM+AdaLN p=2 ckpt, UNFROZEN, baseline lr1e3_const);
# only the task set (all 400) and array layout change.
# Reuses EVAL_SAVE_NAME=ttt_autores/film_unfrozen: the 100 screen predictions are skipped
# (skip-if-done), so only the remaining ~300 run. Score after with:
#   python scripts/score_one.py ttt_autores/film_unfrozen
# Bars to beat: confirmed-400 best = film_freeze 43.25% P@1 / 48.0% P@2; baseline 36.25%/41.5%.
# 40-way array, STRIDE=40 -> 10 tasks/job; unfrozen is ~50 min/task, ~7.5 fresh/job after
# skips => ~6h of work spread over requeue windows (skip-if-done makes requeue safe).

echo "TTT autores film_unfrozen 400-confirm array ${SLURM_ARRAY_TASK_ID}/39 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag_film.py"
EVAL_SAVE_NAME="ttt_autores/film_unfrozen"
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
        --no-compile
    fi
  fi
  i=$((i + 1))
done

echo "[${JOB_IDX}] Done: $(date)"
