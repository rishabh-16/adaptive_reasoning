#!/usr/bin/env python3
"""
Example script demonstrating how to use download_dataset.py
for various common datasets and use cases.
"""

import subprocess
import sys
from pathlib import Path

def run_command(cmd):
    """Run a command and print the output."""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
        print("STDOUT:", e.stdout)
        print("STDERR:", e.stderr)
        return False

def main():
    """Run example dataset downloads."""
    
    # Create data directory
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)
    
    examples = [
        {
            "name": "Download IMDB movie reviews dataset (train split only)",
            "cmd": ["python", "download_dataset.py", 
                   "--dataset_id", "imdb", 
                   "--split", "train",
                   "--target_dir", "./data/imdb_train"]
        },
        {
            "name": "Download SQuAD dataset (all splits)",
            "cmd": ["python", "download_dataset.py", 
                   "--dataset_id", "squad", 
                   "--target_dir", "./data/squad"]
        },
        {
            "name": "Download GLUE SST-2 dataset",
            "cmd": ["python", "download_dataset.py", 
                   "--dataset_id", "glue", 
                   "--config", "sst2",
                   "--target_dir", "./data/glue_sst2"]
        },
        {
            "name": "Download Common Crawl Stories dataset in Parquet format",
            "cmd": ["python", "download_dataset.py", 
                   "--dataset_id", "roneneldan/TinyStories", 
                   "--save_format", "parquet",
                   "--target_dir", "./data/tinystories"]
        },
        {
            "name": "List available configurations for GLUE dataset",
            "cmd": ["python", "download_dataset.py", 
                   "--dataset_id", "glue", 
                   "--list_configs"]
        }
    ]
    
    print("Dataset Download Examples")
    print("=" * 50)
    
    for i, example in enumerate(examples, 1):
        print(f"\n{i}. {example['name']}")
        print("-" * 40)
        
        if input("Run this example? (y/n): ").lower().strip() == 'y':
            success = run_command(example['cmd'])
            if success:
                print("✓ Success!")
            else:
                print("✗ Failed!")
                if input("Continue with next example? (y/n): ").lower().strip() != 'y':
                    break
        else:
            print("Skipped.")
    
    print("\nAll examples completed!")
    print("\nTo run any of these manually, use the commands shown above.")
    print("For more options, run: python download_dataset.py --help")

if __name__ == "__main__":
    main() 