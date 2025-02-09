#!/bin/bash
#SBATCH --job-name=semantic_image_segmentation
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=philipp.reinig@gmail.com
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=100G
#SBATCH --output=../logs/slurm/%x-%j.out

cd ~/development/bachelorarbeit/

source .venv_ants/bin/activate

srun python3 train_semantic_image_segmentation.py
