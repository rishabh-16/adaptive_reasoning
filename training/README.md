# Qwen Reasoning Trajectory Training

This repository contains scripts for generating reasoning trajectories using a large Qwen model and then fine-tuning a smaller Qwen model using those trajectories.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Weights & Biases account (optional but recommended for training monitoring):
```bash
wandb login
```

3. Configure the training parameters:
   - Edit `config.sh` to set your desired model configurations, training parameters, and environment variables
   - Replace placeholder values like `your_wandb_api_key` and `your_username` with your actual values

## Usage

### Running the Training Pipeline

You can run the training pipeline either locally or using SLURM:

#### Local Execution

```bash
# Make the script executable
chmod +x launch_local.sh

# Run the training pipeline
./launch_local.sh
```

#### SLURM Execution

```bash
# Make the script executable
chmod +x launch_slurm.sh

# Submit the job to SLURM
sbatch launch_slurm.sh
```

Note: Before running with SLURM, make sure to:
1. Update the SLURM configuration in `launch_slurm.sh` (partition, resources, etc.)
2. Set up the correct module loads for your cluster
3. Update the conda environment name

### Manual Execution

If you prefer to run the steps manually:

1. Generate reasoning trajectories:
```bash
python generate_trajectories.py
```

2. Fine-tune the smaller model:
```bash
python train_sft.py
```

## Configuration

You can modify the following parameters in `config.sh`:

- Model configurations (model names)
- Training hyperparameters (epochs, batch size, learning rate)
- LoRA configurations (rank, alpha, dropout)
- Generation parameters (temperature, top_p)
- Environment configurations (W&B settings)
- Directory configurations

## Output

The training process will:
1. Generate reasoning trajectories in `trajectories/reasoning_trajectories.json`
2. Save model checkpoints in `qwen-sft-checkpoints`
3. Save the final fine-tuned model in `qwen-sft-final`
4. Store logs in the `logs` directory

## Monitoring

Training progress can be monitored through:
- Weights & Biases dashboard (if enabled)
- Log files in the `logs` directory
- SLURM output files (when using SLURM)
- GPU usage statistics (when running locally) 