#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --array=0-24
#SBATCH --exclude=hipster-cn011  # broken node: /local_scratch container ns missing -> launch failures

# Phase 3 — LoRA adapters on the Hyena mixer projections, on the BEST checkpoint
# (FiLM+BlockDiag p=2). The under-fit diagnosis implicates the spatial mixer: full
# projection training is inert/noisy (freeze-mixer == baseline), so this freezes the base
# qkv/out projections and adapts them only via a rank-r LoRA delta -- constrained test-time
# fitting capacity on exactly the part the dx implicates. Base recipe = film_unfrozen
# (lr1e3_const), confirmed 42.75%; we vary LORA_RANK per launch.
#
# Launch via sbatch --export, overriding job-name/output on the CLI, e.g.:
#   sbatch -J ttt_lora_r8 -o logs/ttt_autores_lora_r8_%A_%a.out \
#          --export=ALL,SWEEP_NAME=lora_r8,LORA_RANK=8 submit_ttt_autores_lora.sh
# Each launch writes to outputs/ttt_autores/film_<SWEEP_NAME>_attempt_0 (skip-if-done safe).
# Score with: python scripts/score_one.py ttt_autores/film_<SWEEP_NAME>
#
# Tunable env vars (defaults = the film_unfrozen base recipe + LoRA rank 8):
#   SWEEP_NAME (required)  LORA_RANK=8  LORA_ALPHA=<rank>  LR=1e-3  SCHED=none  WD=0
#   NUM_ATTEMPTS=10  EPOCHS=100  TTT_NUM_EACH=2  FREEZE=0  BATCH=8

echo "TTT autores LoRA sweep '${SWEEP_NAME}' array ${SLURM_ARRAY_TASK_ID}/24 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

: "${SWEEP_NAME:?must set SWEEP_NAME via --export}"
LR=${LR:-1e-3}; SCHED=${SCHED:-none}; WD=${WD:-0}; NUM_ATTEMPTS=${NUM_ATTEMPTS:-10}
EPOCHS=${EPOCHS:-100}; TTT_NUM_EACH=${TTT_NUM_EACH:-2}; FREEZE=${FREEZE:-0}; BATCH=${BATCH:-8}
LORA_RANK=${LORA_RANK:-8}; LORA_ALPHA=${LORA_ALPHA:-${LORA_RANK}}

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag_film.py"
CKPT="saves/offline_train_Hyena_patch2_blockdiag_film_lr1e3/checkpoint_best.pt"
EVAL_SAVE_NAME="ttt_autores/film_${SWEEP_NAME}"
STRIDE=25
JOB_IDX=${SLURM_ARRAY_TASK_ID}

FREEZE_ARG=""; [ "$FREEZE" = "1" ] && FREEZE_ARG="--freeze-hyena-filters"
echo "Recipe: LORA_RANK=$LORA_RANK LORA_ALPHA=$LORA_ALPHA LR=$LR SCHED=$SCHED WD=$WD NUM_ATTEMPTS=$NUM_ATTEMPTS EPOCHS=$EPOCHS TTT_NUM_EACH=$TTT_NUM_EACH FREEZE=$FREEZE BATCH=$BATCH"

# Fixed 100-task subset (first 50 = original ablation subset; next 50 = sorted ARC-1 eval).
file_names=("af24b4cc" "e1d2900e" "903d1b4a" "4e469f39" "b1fc8b8e" "2c737e39" "992798f6" "00576224" "48131b3c" "60a26a3e" "59341089" "31d5ba1a" "e633a9e5" "62ab2642" "73c3b0d8" "c663677b" "c48954c1" "08573cc6" "136b0064" "929ab4e9" "5b526a93" "ef26cbf6" "fafd9572" "67c52801" "ad7e01d0" "506d28a5" "27a77e38" "d492a647" "72a961c9" "fd4b2b02" "bf89d739" "f5aa3634" "b942fd60" "d282b262" "9772c176" "ed74f2f2" "184a9768" "94133066" "256b0a75" "e681b708" "ce8d95cc" "817e6c09" "7d18a6fb" "1da012fc" "310f3251" "bf699163" "917bccba" "551d5bf1" "b457fec5" "50a16a69" "009d5c81" "00dbd492" "03560426" "05a7bcf2" "0607ce86" "0692e18c" "070dd51e" "0934a4d8" "09c534e7" "0a1d4ef5" "0a2355a6" "0b17323b" "0bb8deee" "0becf7df" "0c786b71" "0c9aba6e" "0d87d2a6" "0e671a1a" "0f63c0b9" "103eff5b" "11e1fe23" "12422b43" "12997ef3" "12eac192" "13713586" "137f0df0" "140c817e" "14754a24" "15113be4" "15663ba9" "15696249" "16b78196" "17b80ad2" "17cae0c1" "18419cfa" "195ba7dc" "1990f7a8" "19bb5feb" "1a2e2828" "1a6449f1" "1acc24af" "1c02dbbe" "1c0d0a4b" "1c56ad9f" "1d0a4b61" "1d398264" "1e81d6f9" "1e97544e" "2037f2c7" "2072aba6")

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
        --lora-rank "${LORA_RANK}" \
        --lora-alpha "${LORA_ALPHA}" \
        ${FREEZE_ARG} \
        --no-compile
    fi
  fi
  i=$((i + 1))
done

echo "[${JOB_IDX}] Done: $(date)"
