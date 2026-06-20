#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:l4:1
#SBATCH --job-name=ttt_autores_g_lr15
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=0-04:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/ttt_autores_guard_lr1p5e3_%A_%a.out
#SBATCH --array=0-24
#SBATCH --exclude=hipster-cn011  # broken node: /local_scratch container ns missing -> launch failures

# Autoresearch experiment #16 — GLOBAL LR 1.5e-3 with the standing NaN-guard.
# Motivated by the under-fitting diagnosis (scripts/diag_supportfit.py): WRONG tasks
# mostly fail to even fit the support set, so MORE adaptation (bigger step) might help.
# The old LR=2e-3 collapsed to 26% UNGUARDED (NaN corruption); the guard (always-on in
# test_time_train_ARC.py since exp#12) now neutralizes that failure mode, so a clean
# re-test of a higher LR is finally possible.
# CAVEAT / prior evidence: exp#15 raised the LIGHT-path LR 1.5x (guard on) -> 43%/46%,
# no gain. Raising the GLOBAL LR also raises the (freeze-mixer-proven inert) mixer LR.
# This screen exists to nail the LR axis shut with the guard in place; expectation is
# neutral-to-negative, but the mixer+light combined at 1.5e-3 is an untested point.
# 25-way array, STRIDE=25 -> 4 tasks/job, fits the 4h wall. Skip-if-done makes requeue safe.
# Score with: python scripts/score_one.py ttt_autores/guard_lr1p5e3

echo "TTT autores guard_lr1p5e3 array ${SLURM_ARRAY_TASK_ID}/24 on ${SLURM_NODELIST}"
echo "Start: $(date)"
set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

HYENA_CONFIG="/home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag.py"
EVAL_SAVE_NAME="ttt_autores/guard_lr1p5e3"
STRIDE=25
JOB_IDX=${SLURM_ARRAY_TASK_ID}

# Fixed 100-task subset: first 50 = original ablation subset (numbers stay comparable),
# next 50 = sorted ARC-1 eval tasks not already in the 50.
file_names=("af24b4cc" "e1d2900e" "903d1b4a" "4e469f39" "b1fc8b8e" "2c737e39" "992798f6" "00576224" "48131b3c" "60a26a3e" "59341089" "31d5ba1a" "e633a9e5" "62ab2642" "73c3b0d8" "c663677b" "c48954c1" "08573cc6" "136b0064" "929ab4e9" "5b526a93" "ef26cbf6" "fafd9572" "67c52801" "ad7e01d0" "506d28a5" "27a77e38" "d492a647" "72a961c9" "fd4b2b02" "bf89d739" "f5aa3634" "b942fd60" "d282b262" "9772c176" "ed74f2f2" "184a9768" "94133066" "256b0a75" "e681b708" "ce8d95cc" "817e6c09" "7d18a6fb" "1da012fc" "310f3251" "bf699163" "917bccba" "551d5bf1" "b457fec5" "50a16a69" "009d5c81" "00dbd492" "03560426" "05a7bcf2" "0607ce86" "0692e18c" "070dd51e" "0934a4d8" "09c534e7" "0a1d4ef5" "0a2355a6" "0b17323b" "0bb8deee" "0becf7df" "0c786b71" "0c9aba6e" "0d87d2a6" "0e671a1a" "0f63c0b9" "103eff5b" "11e1fe23" "12422b43" "12997ef3" "12eac192" "13713586" "137f0df0" "140c817e" "14754a24" "15113be4" "15663ba9" "15696249" "16b78196" "17b80ad2" "17cae0c1" "18419cfa" "195ba7dc" "1990f7a8" "19bb5feb" "1a2e2828" "1a6449f1" "1acc24af" "1c02dbbe" "1c0d0a4b" "1c56ad9f" "1d0a4b61" "1d398264" "1e81d6f9" "1e97544e" "2037f2c7" "2072aba6")

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
        --learning-rate 1.5e-3 \
        --weight-decay 0 \
        --num-colors 12 \
        --resume-checkpoint "saves/offline_train_Hyena_patch2_blockdiag_lr1e3/checkpoint_best.pt" \
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
