#!/bin/bash

# Batch script to download common datasets using download_dataset.py
# Make sure to install dependencies first: pip install -r requirements_dataset_downloader.txt

set -e  # Exit on any error

echo "Starting batch download of common datasets..."
echo "=============================================="

# Create data directory
mkdir -p ./Quantized-Reasoning-Models/datasets

# Common NLP datasets
echo "Downloading aime 90 dataset..."
python download_dataset.py --dataset_id "xiaoyuanliu/AIME90" --target_dir "./Quantized-Reasoning-Models/datasets/AIME90"
python download_dataset.py --dataset_id "yentinglin/aime_2025" --target_dir "./Quantized-Reasoning-Models/datasets/aime_2025"
