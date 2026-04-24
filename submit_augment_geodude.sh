#!/bin/bash
#SBATCH --partition=capacity
#SBATCH --gres=gpu:1
#SBATCH --job-name=varc_augment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/varc_augment_%A.out

echo "Augmentation job running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source /home/dwessel/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

export PYTHONPATH="."

cd /home/dwessel/code/VARC

python augment_data.py

echo "Done: $(date)"
