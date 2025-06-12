#!/bin/bash

EXPERIMENT=$1
N_TASKS=$2
JOB_NAME=$3
GPUS_PER_TASK=$4

if [ -z "$N_TASKS" ]; then
    N_TASKS=1
fi

if [ -z "$GPUS_PER_TASK" ]; then
    GPUS_PER_TASK=1
fi

if [ -z "$JOB_NAME" ]; then
    JOB_NAME=$EXPERIMENT
fi

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment>"
    exit 1
fi

chmod +x experiments/$EXPERIMENT.sh
sbatch --account=genai_interns --qos=genai_interns --job-name=$JOB_NAME --ntasks=$N_TASKS --gpus-per-task=$GPUS_PER_TASK experiments/$EXPERIMENT.sh $GPUS_PER_TASK

echo "Job submitted with job name: $JOB_NAME, ntasks: $N_TASKS, gpus per task: $GPUS_PER_TASK"