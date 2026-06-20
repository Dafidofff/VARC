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

# 400-task CONFIRMATION of a LoRA screen win (Phase 3). Recipe IDENTICAL to the LoRA screen
# (submit_ttt_autores_lora.sh: FiLM+BlockDiag p2 ckpt, base qkv/out proj FROZEN, rank-r LoRA
# delta, baseline lr1e3_const); only the task set (all 400) and array layout change.
# Reuses EVAL_SAVE_NAME=ttt_autores/film_<SWEEP_NAME>: the 100 screen predictions are skipped
# (skip-if-done), so only the remaining ~300 run. Score after with:
#   python scripts/score_one.py ttt_autores/film_<SWEEP_NAME>
# Bar to beat: confirmed-400 best = film_freeze 43.25% P@1 / 48.0% P@2.
#
# Launch via --export, e.g.:
#   sbatch -J ttt_lora_r4_400 -o logs/ttt_autores_lora_r4_400_%A_%a.out \
#          --export=ALL,SWEEP_NAME=lora_r4,LORA_RANK=4 submit_ttt_autores_lora_400.sh

echo "TTT autores LoRA 400-confirm '${SWEEP_NAME}' array ${SLURM_ARRAY_TASK_ID}/39 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

: "${SWEEP_NAME:?must set SWEEP_NAME via --export}"
LORA_RANK=${LORA_RANK:-8}; LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}

# CKPT/CONFIG overridable via --export (must match the screen that produced the reused preds).
HYENA_CONFIG="${HYENA_CONFIG:-/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag_film.py}"
CKPT="${CKPT:-saves/offline_train_Hyena_patch2_blockdiag_film_lr1e3/checkpoint_best.pt}"
EVAL_SAVE_NAME="ttt_autores/film_${SWEEP_NAME}"
STRIDE=40
JOB_IDX=${SLURM_ARRAY_TASK_ID}
echo "Recipe: LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA (base proj frozen, lr1e3_const)"

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
        --resume-checkpoint "${CKPT}" \
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
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        --no-compile
    fi
  fi
  i=$((i + 1))
done

echo "[${JOB_IDX}] Done: $(date)"
