# Hugging Face Dataset Downloader

A comprehensive script to download Hugging Face datasets to a specified directory with various configuration options.

## Features

- Download any Hugging Face dataset to a local directory
- Support for dataset configurations and specific splits
- Multiple output formats (Arrow, Parquet, CSV, JSON)
- Streaming mode for large datasets
- Parallel processing support
- Private dataset support with tokens
- Configuration listing functionality
- Automatic directory structure creation

## Installation

Install the required dependencies:

```bash
pip install datasets huggingface_hub
```

## Usage

### Basic Usage

```bash
# Download entire dataset
python download_dataset.py --dataset_id "squad" --target_dir "./data/squad"

# Download specific split
python download_dataset.py --dataset_id "imdb" --split "train" --target_dir "./data/imdb_train"

# Download with specific configuration
python download_dataset.py --dataset_id "glue" --config "sst2" --target_dir "./data/glue_sst2"
```

### Advanced Usage

```bash
# Download multiple splits
python download_dataset.py --dataset_id "squad" --split "train,validation" --target_dir "./data/squad"

# Use streaming mode for large datasets
python download_dataset.py --dataset_id "c4" --streaming --target_dir "./data/c4"

# Save in different formats
python download_dataset.py --dataset_id "imdb" --save_format "parquet" --target_dir "./data/imdb"

# Use custom cache directory
python download_dataset.py --dataset_id "squad" --cache_dir "/tmp/hf_cache" --target_dir "./data/squad"

# Download private dataset with token
python download_dataset.py --dataset_id "private/dataset" --token "your_hf_token" --target_dir "./data/private"

# Use parallel processing
python download_dataset.py --dataset_id "squad" --num_proc 4 --target_dir "./data/squad"
```

### List Available Configurations

```bash
# List all available configurations for a dataset
python download_dataset.py --dataset_id "glue" --list_configs
```

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--dataset_id` | Hugging Face dataset repository ID (required) | - |
| `--target_dir` | Directory to save the dataset | `./data/{dataset_slug}` |
| `--config` | Dataset configuration name | `None` |
| `--split` | Dataset split(s) to download (comma-separated) | `None` (all splits) |
| `--cache_dir` | Custom cache directory for temporary files | `None` |
| `--token` | Hugging Face token for private datasets | `None` |
| `--streaming` | Use streaming mode for large datasets | `False` |
| `--trust_remote_code` | Trust remote code for datasets that require it | `False` |
| `--revision` | Specific revision/branch to download | `None` |
| `--num_proc` | Number of processes for parallel processing | `None` |
| `--save_format` | Format to save dataset (arrow, parquet, csv, json) | `arrow` |
| `--list_configs` | List available configurations and exit | `False` |

## Output Structure

The script creates the following structure:

```
target_dir/
├── dataset_info.json          # Dataset metadata
├── train/                     # Training split (if available)
│   ├── data.arrow            # Dataset files
│   └── ...
├── validation/               # Validation split (if available)
│   ├── data.arrow
│   └── ...
└── test/                     # Test split (if available)
    ├── data.arrow
    └── ...
```

For single splits, the data is saved directly in the target directory.

## Examples

### Common Datasets

```bash
# NLP datasets
python download_dataset.py --dataset_id "squad" --target_dir "./data/squad"
python download_dataset.py --dataset_id "imdb" --target_dir "./data/imdb"
python download_dataset.py --dataset_id "glue" --config "sst2" --target_dir "./data/glue_sst2"

# Computer Vision datasets
python download_dataset.py --dataset_id "cifar10" --target_dir "./data/cifar10"
python download_dataset.py --dataset_id "imagenet-1k" --target_dir "./data/imagenet"

# Large language model datasets
python download_dataset.py --dataset_id "roneneldan/TinyStories" --target_dir "./data/tinystories"
python download_dataset.py --dataset_id "openwebtext" --streaming --target_dir "./data/openwebtext"
```

### Batch Processing

Use the provided `example_dataset_downloads.py` script to see interactive examples:

```bash
python example_dataset_downloads.py
```

## Dataset Information

The script automatically saves dataset metadata to `dataset_info.json` in the target directory:

```json
{
  "dataset_id": "squad",
  "config": null,
  "split": null,
  "revision": null,
  "save_format": "arrow",
  "features": "{'id': Value(dtype='string'), 'title': Value(dtype='string'), ...}",
  "num_rows": 87599
}
```

## Error Handling

The script includes comprehensive error handling:

- Validates dataset existence and accessibility
- Handles network interruptions gracefully
- Provides clear error messages for common issues
- Supports resumable downloads where possible

## Performance Tips

1. **Use streaming mode** for very large datasets to avoid memory issues
2. **Enable parallel processing** with `--num_proc` for faster downloads
3. **Choose appropriate formats**: Arrow for speed, Parquet for compression
4. **Use custom cache directories** on fast storage for better performance

## Troubleshooting

### Common Issues

1. **ImportError**: Install required packages with `pip install datasets huggingface_hub`
2. **Authentication errors**: Use `--token` for private datasets
3. **Memory issues**: Use `--streaming` for large datasets
4. **Network timeouts**: The script will retry automatically

### Getting Help

```bash
python download_dataset.py --help
```

## License

This script is provided as-is for educational and research purposes. 