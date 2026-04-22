#!/bin/bash
#SBATCH --partition=rome
#SBATCH --job-name=varc_augment
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=0-02:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/varc_augment_%A.out

echo "Augmentation job running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dwessels2/code/VARC

python augment_data.py

echo "Done: $(date)"
