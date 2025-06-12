#!/bin/bash

EXPERIMENT=$1
N_TASKS=$2
GPUS_PER_TASK=$3

if [ -z "$N_TASKS" ]; then
    N_TASKS=1
fi

if [ -z "$GPUS_PER_TASK" ]; then
    GPUS_PER_TASK=1
fi

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment>"
    exit 1
fi

chmod +x experiments/$EXPERIMENT/run.sh
sbatch --account=genai_interns --qos=genai_interns --job-name=$EXPERIMENT --ntasks=$N_TASKS --gpus-per-task=$GPUS_PER_TASK --output=experiments/$EXPERIMENT/%x_%j.out experiments/$EXPERIMENT/run.sh $GPUS_PER_TASK

echo "Job submitted with job name: $EXPERIMENT, ntasks: $N_TASKS, gpus per task: $GPUS_PER_TASK"