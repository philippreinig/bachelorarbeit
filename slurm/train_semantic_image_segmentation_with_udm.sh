#!/bin/bash
#SBATCH --job-name=img_udm_sem_seg
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=philipp.reinig@gmail.com
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=32
#SBATCH --mem=500G
#SBATCH --output=../logs/slurm/%x-%j.out

export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

cd ~/development/bachelorarbeit/

source .venv_ants/bin/activate

srun python3 train_semantic_image_segmentation_with_udm.py $SLURM_JOB_ID
