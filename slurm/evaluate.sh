#!/bin/bash
#SBATCH --job-name=eval
#SBATCH --partition=gpu
#SBATCH --mail-type=END,FAIL    
#SBATCH --mail-user=reinig@ovgu.de
#SBATCH --nodes=1
#SBATCH --gpus=h100:1
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=150G
#SBATCH --output=../logs/slurm/%x-%j.out

cd ~/development/bachelorarbeit/

source .venv_ants/bin/activate

#export CUDA_LAUNCH_BLOCKING=1
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments
export TF_ENABLE_ONEDNN_OPTS=0
export TF_CPP_MIN_LOG_LEVEL=1

echo $CUDA_VISIBLE_DEVICES
#echo $PYTORCH_CUDA_ALLOC_CONF

srun python3 evaluate.py $SLURM_JOB_ID
