#!/bin/bash
#SBATCH --partition=performance
#SBATCH --gres=gpu:rtx_6000_ada:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0-00:30:00
#SBATCH --mem=32G
#SBATCH --job-name=lora_smoke
#SBATCH --output=logs/lora_smoke_%j.out
#SBATCH --exclude=hipster-cn011

# One-task, 2-epoch LoRA smoke on the real FiLM+BlockDiag p2 model: confirms inject_lora
# finds the nested mixer qkv/out projections, the FiLM ckpt loads, and TTT runs end-to-end.
set -eo pipefail
source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq
cd /home/dwessel/code/VARC

python test_time_train_ARC.py \
  --epochs 2 --batch-size 8 --image-size 64 --learning-rate 1e-3 --weight-decay 0 --num-colors 12 \
  --resume-checkpoint saves/offline_train_Hyena_patch2_blockdiag_film_lr1e3/checkpoint_best.pt \
  --patch-size 2 --lr-scheduler none \
  --train-split eval_color_permute_ttt_9/af24b4cc --data-root raw_data/ARC-AGI \
  --eval-split eval_color_permute_ttt_9/af24b4cc --resume-skip-task-token \
  --architecture hyena \
  --hyena-config /home/dwessel/code/nvSubquadratic-private/varc_configs/cfg_hyena_rearc_subq_ops_patch2_circular_adaln_blockdiag_film.py \
  --eval-save-name ttt_autores/_lora_smoke --num-attempts 1 --ttt-num-each 1 \
  --lora-rank 8 --lora-alpha 8 --no-compile
echo "SMOKE_DONE"
