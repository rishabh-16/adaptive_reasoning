#!/usr/bin/env python3
"""
Script to download Hugging Face datasets to a specified directory.
Supports various dataset configurations, splits, and download options.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Optional, List, Union
import json

try:
    from datasets import load_dataset, Dataset, DatasetDict
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    print("Required packages not found. Please install with:")
    print("pip install datasets huggingface_hub")
    sys.exit(1)


def create_dataset_slug(dataset_id: str, config: Optional[str] = None) -> str:
    """Create a filesystem-safe slug from dataset ID and config."""
    slug = dataset_id.replace('/', '--')
    if config:
        slug += f"--{config}"
    return slug


def download_dataset(
    dataset_id: str,
    target_dir: str,
    config: Optional[str] = None,
    split: Optional[Union[str, List[str]]] = None,
    streaming: bool = False,
    cache_dir: Optional[str] = None,
    token: Optional[str] = None,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    num_proc: Optional[int] = None,
    save_format: str = "arrow"
) -> None:
    """
    Download a Hugging Face dataset to the specified directory.
    
    Args:
        dataset_id: Hugging Face dataset repository ID
        target_dir: Directory to save the dataset
        config: Dataset configuration name (if applicable)
        split: Dataset split(s) to download
        streaming: Whether to use streaming mode
        cache_dir: Custom cache directory
        token: Hugging Face token for private datasets
        trust_remote_code: Whether to trust remote code
        revision: Specific revision/branch to download
        num_proc: Number of processes for parallel processing
        save_format: Format to save dataset ('arrow', 'parquet', 'csv', 'json')
    """
    
    print(f"Downloading dataset: {dataset_id}")
    if config:
        print(f"Configuration: {config}")
    if split:
        print(f"Split(s): {split}")
    print(f"Target directory: {target_dir}")
    print(f"Save format: {save_format}")
    
    try:
        # Load the dataset
        dataset = load_dataset(
            dataset_id,
            name=config,
            split=split,
            streaming=streaming,
            cache_dir=cache_dir,
            token=token,
            trust_remote_code=trust_remote_code,
            revision=revision,
            num_proc=num_proc
        )
        
        if streaming:
            print("Dataset loaded in streaming mode. Converting to local format...")
            # Convert streaming dataset to regular dataset
            if isinstance(dataset, dict):
                dataset = {k: Dataset.from_generator(lambda: iter(v)) for k, v in dataset.items()}
            else:
                dataset = Dataset.from_generator(lambda: iter(dataset))
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Save dataset in the specified format
        if isinstance(dataset, DatasetDict):
            # Handle multiple splits
            for split_name, split_dataset in dataset.items():
                split_path = Path(target_dir) / split_name
                save_dataset_split(split_dataset, split_path, save_format)
        else:
            # Handle single split
            save_dataset_split(dataset, Path(target_dir), save_format)
        
        # Save dataset info
        info_path = Path(target_dir) / "dataset_info.json"
        dataset_info = {
            "dataset_id": dataset_id,
            "config": config,
            "split": split,
            "revision": revision,
            "save_format": save_format,
            "features": str(dataset.features) if hasattr(dataset, 'features') else None,
            "num_rows": len(dataset) if hasattr(dataset, '__len__') else None
        }
        
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"Dataset downloaded successfully to {target_dir}")
        
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise


def save_dataset_split(dataset: Dataset, path: Path, format: str) -> None:
    """Save a dataset split in the specified format."""
    path.mkdir(parents=True, exist_ok=True)
    
    if format == "arrow":
        dataset.save_to_disk(str(path))
    elif format == "parquet":
        dataset.to_parquet(str(path / "data.parquet"))
    elif format == "csv":
        dataset.to_csv(str(path / "data.csv"))
    elif format == "json":
        dataset.to_json(str(path / "data.json"))
    else:
        raise ValueError(f"Unsupported format: {format}")


def list_dataset_configs(dataset_id: str, token: Optional[str] = None) -> List[str]:
    """List available configurations for a dataset."""
    try:
        api = HfApi(token=token)
        dataset_info = api.dataset_info(dataset_id)
        
        if hasattr(dataset_info, 'config_names') and dataset_info.config_names:
            return dataset_info.config_names
        else:
            # Try to load dataset to get configs
            from datasets import get_dataset_config_names
            return get_dataset_config_names(dataset_id, token=token)
    except Exception as e:
        print(f"Could not retrieve configurations: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(
        description="Download Hugging Face datasets to a specified directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download entire dataset
  python download_dataset.py --dataset_id "squad" --target_dir "./data/squad"
  
  # Download specific split
  python download_dataset.py --dataset_id "imdb" --split "train" --target_dir "./data/imdb_train"
  
  # Download with specific configuration
  python download_dataset.py --dataset_id "glue" --config "sst2" --target_dir "./data/glue_sst2"
  
  # Download multiple splits
  python download_dataset.py --dataset_id "squad" --split "train,validation" --target_dir "./data/squad"
  
  # List available configurations
  python download_dataset.py --dataset_id "glue" --list_configs
        """
    )
    
    parser.add_argument(
        "--dataset_id", 
        type=str, 
        required=True,
        help="Hugging Face dataset repository ID (e.g., 'squad', 'imdb', 'glue')"
    )
    
    parser.add_argument(
        "--target_dir", 
        type=str,
        help="Directory to save the dataset. If not provided, uses dataset_id as directory name"
    )
    
    parser.add_argument(
        "--config", 
        type=str, 
        default=None,
        help="Dataset configuration name (if applicable)"
    )
    
    parser.add_argument(
        "--split", 
        type=str, 
        default=None,
        help="Dataset split(s) to download. Use comma-separated for multiple splits (e.g., 'train,validation')"
    )
    
    parser.add_argument(
        "--cache_dir", 
        type=str, 
        default=None,
        help="Custom cache directory for temporary files"
    )
    
    parser.add_argument(
        "--token", 
        type=str, 
        default=None,
        help="Hugging Face token for private datasets"
    )
    
    parser.add_argument(
        "--streaming", 
        action="store_true",
        help="Use streaming mode for large datasets"
    )
    
    parser.add_argument(
        "--trust_remote_code", 
        action="store_true",
        help="Trust remote code for datasets that require it"
    )
    
    parser.add_argument(
        "--revision", 
        type=str, 
        default=None,
        help="Specific revision/branch to download"
    )
    
    parser.add_argument(
        "--num_proc", 
        type=int, 
        default=None,
        help="Number of processes for parallel processing"
    )
    
    parser.add_argument(
        "--save_format", 
        type=str, 
        choices=["arrow", "parquet", "csv", "json"],
        default="arrow",
        help="Format to save the dataset (default: arrow)"
    )
    
    parser.add_argument(
        "--list_configs", 
        action="store_true",
        help="List available configurations for the dataset and exit"
    )
    
    args = parser.parse_args()
    
    # Handle listing configurations
    if args.list_configs:
        print(f"Available configurations for {args.dataset_id}:")
        configs = list_dataset_configs(args.dataset_id, args.token)
        if configs:
            for config in configs:
                print(f"  - {config}")
        else:
            print("  No configurations found or dataset uses default configuration")
        return
    
    # Set target directory if not provided
    if not args.target_dir:
        dataset_slug = create_dataset_slug(args.dataset_id, args.config)
        args.target_dir = f"./data/{dataset_slug}"
    
    # Parse split argument
    split = None
    if args.split:
        splits = [s.strip() for s in args.split.split(',')]
        split = splits if len(splits) > 1 else splits[0]
    
    # Download the dataset
    download_dataset(
        dataset_id=args.dataset_id,
        target_dir=args.target_dir,
        config=args.config,
        split=split,
        streaming=args.streaming,
        cache_dir=args.cache_dir,
        token=args.token,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
        num_proc=args.num_proc,
        save_format=args.save_format
    )


if __name__ == "__main__":
    main() 