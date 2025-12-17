#!/bin/bash
#SBATCH --job-name=04_datagen
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --array=0-99
#SBATCH --gpus-per-node=8
#SBATCH --account=transformer2
#SBATCH --qos=h200_comm_shared
#SBATCH --output=experiments/04_datagen/logs/%x_%j.out
#SBATCH --mem=400GB
#SBATCH --time=48:00:00

echo "Starting run of job $SLURM_JOB_NAME with SLURM array id $SLURM_ARRAY_TASK_ID"
cd /home/rishabhtiwari/adaptive_reasoning/experiments/04_datagen
python3 -u main.py --split-id $SLURM_ARRAY_TASK_ID
