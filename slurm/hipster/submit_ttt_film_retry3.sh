#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --job-name=varc_ttt_film_retry
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0:30:00
#SBATCH --mem=64G
#SBATCH --output=logs/varc_ttt_film_retry3_%A_%a.out
#SBATCH --array=0-2
#SBATCH --exclude=hipster-cn011

echo "FiLM TTT retry job ${SLURM_ARRAY_TASK_ID} on $SLURM_NODELIST — Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_film.py"
CHECKPOINT="saves/offline_train_Hyena_patch2_film_lr1e3/checkpoint_best.pt"
EVAL_SAVE_NAME="ARC_1_eval_Hyena_patch2_film_attempt_0"

# The 3 tasks missing from attempt_1
file_names=("310f3251" "50aad11f" "e345f17b")
file_name="${file_names[${SLURM_ARRAY_TASK_ID}]}"

# Skip if attempt_1 already exists for this task
if [ -f "outputs/${EVAL_SAVE_NAME}_attempt_1/${file_name}_predictions.json" ]; then
  echo "skipping ${file_name} (attempt_1 already done)"
  exit 0
fi

echo "processing task ${file_name}"
python test_time_train_ARC.py \
  --epochs 100 \
  --batch-size 8 \
  --image-size 64 \
  --patch-size 2 \
  --learning-rate 1e-3 \
  --weight-decay 0 \
  --num-colors 12 \
  --resume-checkpoint "${CHECKPOINT}" \
  --lr-scheduler none \
  --train-split "eval_color_permute_ttt_9/${file_name}" \
  --data-root "raw_data/ARC-AGI" \
  --eval-split "eval_color_permute_ttt_9/${file_name}" \
  --resume-skip-task-token \
  --architecture "hyena" \
  --hyena-config "${HYENA_CONFIG}" \
  --eval-save-name "${EVAL_SAVE_NAME}" \
  --num-attempts 10 \
  --ttt-num-each 2 \
  --no-compile

echo "Done: $(date)"
