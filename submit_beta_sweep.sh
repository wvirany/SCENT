#!/bin/bash
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=scenta_beta
#SBATCH --output=logs/beta_%A_%a.out
#SBATCH --error=logs/beta_%A_%a.err
#SBATCH --array=1-8

# Load modules
module load python/3.11 scipy-stack rdkit/2023.09.5 cuda/12.6

# Activate virtual environment
source ~/scratch/SCENT/scent_env/bin/activate

BETA_VALUES=(1 2 4 8 16 32 64 128)
BETA=${BETA_VALUES[$SLURM_ARRAY_TASK_ID-1]}

# Run experiment
cd ~/scratch/SCENT
python train.py --cfg configs/experiments/beta-experiments/beta_${BETA}.gin
