#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --job-name=ttt_abl_lr5e4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-04:00:00
#SBATCH --mem=64G
#SBATCH --output=logs/ttt_abl_lr5e4_%A_%a.out
#SBATCH --array=0-7

echo "TTT ablation baseline array ${SLURM_ARRAY_TASK_ID}/7 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/examples/arc/cfg_hyena_rearc_subq_ops_patch1_circular_adaln.py"
EVAL_SAVE_NAME="ttt_ablation/lr_5e4"
STRIDE=8
JOB_IDX=${SLURM_ARRAY_TASK_ID}

file_names=("af24b4cc" "e1d2900e" "903d1b4a" "4e469f39" "b1fc8b8e" "2c737e39" "992798f6" "00576224" "48131b3c" "60a26a3e" "59341089" "31d5ba1a" "e633a9e5" "62ab2642" "73c3b0d8" "c663677b" "c48954c1" "08573cc6" "136b0064" "929ab4e9" "5b526a93" "ef26cbf6" "fafd9572" "67c52801" "ad7e01d0" "506d28a5" "27a77e38" "d492a647" "72a961c9" "fd4b2b02" "bf89d739" "f5aa3634" "b942fd60" "d282b262" "9772c176" "ed74f2f2" "184a9768" "94133066" "256b0a75" "e681b708" "ce8d95cc" "817e6c09" "7d18a6fb" "1da012fc" "310f3251" "bf699163" "917bccba" "551d5bf1" "b457fec5" "50a16a69")

i=0
for file_name in "${file_names[@]}"; do
  if (( i % STRIDE == JOB_IDX )); then
    echo "[${JOB_IDX}] task ${file_name} (idx ${i})"
    python test_time_train_ARC.py \
      --epochs 100 \
      --batch-size 8 \
      --image-size 64 \
      --patch-size 1 \
      --learning-rate 5e-4 \
      --weight-decay 0 \
      --num-colors 12 \
      --resume-checkpoint "saves/offline_train_Hyena_100ep_lr1e3/checkpoint_best.pt" \
      --lr-scheduler cosine \
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
  i=$((i + 1))
done

echo "[${JOB_IDX}] Done: $(date)"
