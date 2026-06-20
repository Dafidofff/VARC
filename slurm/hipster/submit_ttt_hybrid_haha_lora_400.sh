#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --array=0-39
#SBATCH --exclude=hipster-cn011  # broken node

# Full-400 TTT for Hybrid HAHA (alternating H A H A H A...) checkpoint.
# Recipe: LoRA r4 on qkv_proj/out_proj (base frozen), lr1e3_const — same as film_lora_r4.
# Requires rope.py fix (inplace ops on gradient views crash TTT backward).
# Output dir: ttt_autores/film_haha_r4  (reuses prior partial outputs via skip-if-done)

echo "TTT hybrid HAHA LoRA r4 400 array ${SLURM_ARRAY_TASK_ID}/39 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

LORA_RANK=4
LORA_ALPHA=4
HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hybrid_haha_p2_blockdiag_film.py"
CKPT="saves/offline_train_Hyena_hybrid_haha_p2_blockdiag_film/checkpoint_best.pt"
EVAL_SAVE_NAME="ttt_autores/film_haha_r4"
STRIDE=40
JOB_IDX=${SLURM_ARRAY_TASK_ID}

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
