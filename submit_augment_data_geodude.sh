#!/bin/bash
#SBATCH --account=geodudeusers
#SBATCH --partition=geodude
#SBATCH --gpus=1
#SBATCH --job-name=varc_augment
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=0-01:00:00
#SBATCH --mem=24G
#SBATCH --output=slurm/varc_augment_%A.out

echo "Augmentation job running on node: $SLURM_NODELIST"
echo "Start: $(date)"

set -eo pipefail

source ~/miniforge3/etc/profile.d/conda.sh
conda activate nvsubq

cd /home/dwessel/code/VARC

python augment_data.py

echo "Done: $(date)"
