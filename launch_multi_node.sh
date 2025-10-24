#!/bin/bash

EXPERIMENT=$1
NODES=$2
GPUS_PER_NODE=$3
NUM_EXPERTS_PER_TOK=$4

if [ -z "$NODES" ]; then
    NODES=2
fi

if [ -z "$GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=8
fi

if [ -z "$EXPERIMENT" ]; then
    echo "Usage: $0 <experiment> [nodes] [gpus_per_node] [num_experts_per_tok]"
    echo "Example: $0 01_training 4 8 4  # Run on 4 nodes with 8 GPUs each, using 4 experts per token"
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
# Pass NUM_EXPERTS_PER_TOK as environment variable to the job
if [ -n "$NUM_EXPERTS_PER_TOK" ]; then
    echo "Experts per token: $NUM_EXPERTS_PER_TOK"
    sbatch \
        --job-name=${EXPERIMENT}_multi \
        --nodes=$NODES \
        --ntasks-per-node=1 \
        --gpus-per-node=$GPUS_PER_NODE \
        --export=ALL,NUM_EXPERTS_PER_TOK=${NUM_EXPERTS_PER_TOK} \
        experiments/$EXPERIMENT/run_multi_node.sh
else
    sbatch \
        --job-name=${EXPERIMENT}_multi \
        --nodes=$NODES \
        --ntasks-per-node=1 \
        --gpus-per-node=$GPUS_PER_NODE \
        experiments/$EXPERIMENT/run_multi_node.sh
fi

echo "Multi-node job submitted!"
echo "Job name: ${EXPERIMENT}_multi"
echo "Nodes: $NODES"
echo "Total GPUs: $TOTAL_GPUS" 