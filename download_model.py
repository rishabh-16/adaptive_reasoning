import argparse
import os
from huggingface_hub import snapshot_download
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Download a Hugging Face model.")
    parser.add_argument("-m", "--model_repo_id", type=str, required=True, 
                        help="Hugging Face model repository ID (e.g., 'meta-llama/Llama-2-7b-hf').")
    parser.add_argument("--cache_dir_base", type=str, default="/home/rishabhtiwari/hf_cache", 
                        help="Base directory for storing downloaded models. A subdirectory will be created under this based on the model name.")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Optional Hugging Face token for downloading gated models.")
    
    args = parser.parse_args()

    # Create a slug from the model path for the local directory name
    # e.g., 'meta-llama/Llama-2-7b-hf' becomes 'meta-llama--Llama-2-7b-hf'
    model_path_slug = '--'.join(args.model_repo_id.split('/'))
    local_model_dir = Path(args.cache_dir_base) / model_path_slug

    print(f"Downloading Hugging Face model: {args.model_repo_id}")
    print(f"Target local directory: {local_model_dir}")

    try:
        snapshot_download(
            repo_id=args.model_repo_id,
            local_dir=str(local_model_dir),
            local_dir_use_symlinks=False,  # Avoid symlinks for better compatibility on shared systems
            token=args.hf_token,
            # You can add other options like allow_patterns, ignore_patterns, resume_download, etc.
            # For example, to download only safetensors:
            # allow_patterns="*.safetensors"
        )
        print(f"Model downloaded successfully to {local_model_dir}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        # Consider re-raising the exception if you want the Slurm job to fail explicitly
        # raise

if __name__ == "__main__":
    main() 