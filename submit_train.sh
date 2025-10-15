#!/bin/bash
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:l40s:1
#SBATCH --job-name=scenta_train_medium
#SBATCH --output=logs/train_medium_%A_%a.out
#SBATCH --error=logs/train_medium_%A_%a.err
#SBATCH --array=1-8
#SBATCH --requeue

# Load modules
module load python/3.11 scipy-stack rdkit/2023.09.5 cuda/12.6

# Activate virtual environment
source ~/scratch/SCENT/scent_env/bin/activate

# Run experiment
cd ~/scratch/SCENT
python train.py --cfg configs/experiments/train_medium.gin
