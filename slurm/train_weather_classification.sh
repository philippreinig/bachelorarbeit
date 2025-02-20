#!/bin/bash
#SBATCH --job-name=weather_classification
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=philipp.reinig@gmail.com
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=64
#SBATCH --mem=150G
#SBATCH --output=../logs/slurm/%x-%j.out

cd ~/development/bachelorarbeit/

source .venv_ants/bin/activate

srun python3 train_weather_classification.py
