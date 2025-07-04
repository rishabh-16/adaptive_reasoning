#!/bin/bash

EXPERIMENT=$1
GPUS=$2

if [ -z "$GPUS" ]; then
    GPUS=1
fi

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment>"
    exit 1
fi

chmod +x experiments/$EXPERIMENT/run.sh
sbatch --account=genai_interns --qos=genai_interns --job-name=$EXPERIMENT --gpus=$GPUS experiments/$EXPERIMENT/run.sh

echo "Job submitted with job name: $EXPERIMENT, gpus: $GPUS"