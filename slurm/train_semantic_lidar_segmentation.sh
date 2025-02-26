#!/bin/bash
#SBATCH --job-name=semantic_lidar_segmentation
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=reinig@ovgu.de
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=100G
#SBATCH --output=../logs/slurm/%x-%j.out

cd ~/development/bachelorarbeit/

source .venv_ants/bin/activate

#export CUDA_LAUNCH_BLOCKING=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#echo $PYTORCH_CUDA_ALLOC_CONF

srun python3 train_semantic_lidar_segmentation.py
