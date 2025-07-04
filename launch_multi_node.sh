#!/bin/bash

EXPERIMENT=$1
NODES=$2
GPUS_PER_NODE=$3

if [ -z "$NODES" ]; then
    NODES=2
fi

if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=8
fi

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment> [nodes] [gpus_per_node]"
    echo "Example: $0 01_training 4 8  # Run on 4 nodes with 8 GPUs each"
    exit 1
fi

# Calculate total GPUs
TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

echo "Launching multi-node training:"
echo "Experiment: $EXPERIMENT"
echo "Nodes: $NODES"
echo "GPUs per node: $GPUS_PER_NODE"
echo "Total GPUs: $TOTAL_GPUS"

# Make the script executable
chmod +x experiments/$EXPERIMENT/run_multi_node.sh

# Submit the job with multi-node configuration
sbatch \
    --account=genai_interns \
    --qos=lowest \
    --job-name=${EXPERIMENT}_multi \
    --nodes=$NODES \
    --ntasks-per-node=1 \
    --gpus-per-node=$GPUS_PER_NODE \
    experiments/$EXPERIMENT/run_multi_node.sh

echo "Multi-node job submitted!"
echo "Job name: ${EXPERIMENT}_multi"
echo "Nodes: $NODES"
echo "Total GPUs: $TOTAL_GPUS" 