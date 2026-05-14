#!/usr/bin/env python3
"""
Helper script to push an existing dataset to HuggingFace Hub.
Loads a dataset from disk and pushes it to the hub.
"""

import os
from pathlib import Path
from datasets import load_from_disk
from dotenv import load_dotenv


def push_dataset_to_hub(dataset_path: str, repo_id: str):
    """
    Load a dataset from disk and push it to HuggingFace Hub.
    
    Args:
        dataset_path: Path to the saved dataset
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
    """
    # Load dataset from disk
    print(f"Loading dataset from: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    print(f"\nDataset loaded successfully!")
    print(f"Splits: {list(dataset.keys())}")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} samples")
    
    # Load HF_TOKEN from .env file
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token")
    
    # Push to hub
    print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
    dataset.push_to_hub(repo_id, token=hf_token)
    print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Push a saved dataset to HuggingFace Hub")
    parser.add_argument("dataset_path", type=str, help="Path to the saved dataset directory")
    parser.add_argument("repo_id", type=str, help="HuggingFace repository ID (e.g., username/dataset-name)")
    
    args = parser.parse_args()
    
    push_dataset_to_hub(args.dataset_path, args.repo_id)
