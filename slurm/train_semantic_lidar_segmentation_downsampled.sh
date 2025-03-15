#!/bin/bash
#SBATCH --job-name=lid_down_sem_seg
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=reinig@ovgu.de
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
##SBATCH --ntasks-per-node=1
##SBATCH --gpus-per-task=h100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=150G
#SBATCH --output=../logs/slurm/%x-%j.out

cd ~/development/bachelorarbeit/

source .venv_ants/bin/activate

#export CUDA_LAUNCH_BLOCKING=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments
#echo $PYTORCH_CUDA_ALLOC_CONF

srun python3 train_semantic_lidar_segmentation_downsampled.py $SLURM_JOB_ID
