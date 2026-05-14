#!/usr/bin/env python3
"""
Script to process images in easy and hard folders and create a HuggingFace dataset
with image, number of nodes, and tree structure.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict, concatenate_datasets
from typing import List, Tuple, Dict, Optional
import json
import os
from dotenv import load_dotenv
import random


def build_contour_tree(hierarchy):
    """Build a tree-like data structure from OpenCV hierarchy."""
    tree_nodes = []
    hierarchy = hierarchy[0]  # Get rid of the extra dimension

    for i, h_entry in enumerate(hierarchy):
        node = {
            "contour_idx": i,
            "next": h_entry[0],
            "previous": h_entry[1],
            "first_child": h_entry[2],
            "parent": h_entry[3],
            "children": []
        }
        tree_nodes.append(node)

    # Populate children lists
    for i, node in enumerate(tree_nodes):
        if node["parent"] != -1:
            tree_nodes[node["parent"]]["children"].append(i)

    # Calculate depths
    contour_depths = {}
    def calculate_depth(node_idx, current_depth):
        contour_depths[node_idx] = current_depth
        for child_idx in tree_nodes[node_idx]["children"]:
            calculate_depth(child_idx, current_depth + 1)

    root_contours = [i for i, node in enumerate(tree_nodes) if node["parent"] == -1]
    for root_idx in root_contours:
        calculate_depth(root_idx, 0)

    return tree_nodes, contour_depths


def merge_contour_pairs(tree_nodes, contour_depths):
    """Merge outer and inner contours into logical regions."""
    merged_nodes = []
    contour_to_merged = {}
    visited = set()

    # Step 1: merge outer + inner pairs
    for idx, node in enumerate(tree_nodes):
        if idx in visited:
            continue

        depth = contour_depths[idx]

        if depth % 2 == 0:
            children = node["children"]

            if len(children) == 1 and contour_depths[children[0]] == depth + 1:
                inner_idx = children[0]
                visited.add(inner_idx)
            else:
                inner_idx = None

            merged_idx = len(merged_nodes)
            merged_nodes.append({
                "outer_contour": idx,
                "inner_contour": inner_idx,
                "children": []
            })

            contour_to_merged[idx] = merged_idx
            if inner_idx is not None:
                contour_to_merged[inner_idx] = merged_idx

            visited.add(idx)

    # Step 2: reconnect children between merged nodes
    for region in merged_nodes:
        source = (
            region["inner_contour"]
            if region["inner_contour"] is not None
            else region["outer_contour"]
        )

        for child in tree_nodes[source]["children"]:
            if child in contour_to_merged:
                child_region = contour_to_merged[child]
                if child_region != contour_to_merged[source]:
                    region["children"].append(child_region)

    # Step 3: inject virtual root
    all_children = set()
    for node in merged_nodes:
        all_children.update(node["children"])

    root_regions = [
        i for i in range(len(merged_nodes))
        if i not in all_children
    ]

    virtual_root = {
        "outer_contour": None,
        "inner_contour": None,
        "children": root_regions
    }

    # shift indices by +1
    for node in merged_nodes:
        node["children"] = [c + 1 for c in node["children"]]

    merged_tree = [virtual_root] + merged_nodes
    merged_tree[0]["children"] = [r + 1 for r in root_regions]

    return merged_tree


def tree_to_edges(merged_tree):
    """Convert a rooted tree into (parent, child) edges."""
    edges = []
    for parent_idx, node in enumerate(merged_tree):
        for child_idx in node["children"]:
            edges.append((parent_idx, child_idx))
    return edges


def process_image(image_path: Path) -> Dict:
    """
    Process a single image and extract tree structure.
    
    Returns:
        Dictionary with 'image', 'num_nodes', and 'tree' keys
    """
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Preprocess
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded_image = cv2.erode(thresh_image, kernel, iterations=1)
    
    # Find contours
    contours, hierarchy = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Build tree
    tree_nodes, contour_depths = build_contour_tree(hierarchy)
    merged_tree = merge_contour_pairs(tree_nodes, contour_depths)
    
    # Get number of nodes (excluding virtual root at index 0)
    num_nodes = len(merged_tree) - 1
    
    # Convert tree to edges
    edges = tree_to_edges(merged_tree)
    
    # Convert image to PIL Image for HuggingFace dataset
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb)
    
    return {
        "image": pil_image,
        "num_nodes": num_nodes,
        "tree": edges  # List of (parent, child) tuples
    }


def process_folder(image_folder: str) -> Dataset:
    """
    Process all images in a folder and create a HuggingFace dataset.
    
    Args:
        image_folder: Path to folder containing images
        
    Returns:
        HuggingFace Dataset
    """
    folder_path = Path(image_folder)
    if not folder_path.exists():
        raise ValueError(f"Folder does not exist: {image_folder}")
    
    # Supported image extensions
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    
    # Find all images
    image_files = [
        f for f in folder_path.iterdir()
        if f.suffix.lower() in image_extensions
    ]
    
    if not image_files:
        raise ValueError(f"No images found in folder: {image_folder}")
    
    print(f"Found {len(image_files)} images. Processing...")
    
    # Process each image
    data = {
        "image": [],
        "num_nodes": [],
        "tree": []
    }
    
    for i, image_file in enumerate(image_files):
        try:
            print(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
            result = process_image(image_file)
            data["image"].append(result["image"])
            data["num_nodes"].append(result["num_nodes"])
            data["tree"].append(result["tree"])
        except Exception as e:
            print(f"Error processing {image_file.name}: {e}")
            continue
    
    if not data["image"]:
        raise ValueError("No images were successfully processed")
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict(data)
    
    print(f"\nDataset created with {len(dataset)} samples")
    print(f"Sample: num_nodes={data['num_nodes'][0]}, tree_edges={len(data['tree'][0])}")
    
    return dataset


def create_dataset(easy_folder: str, hard_folder: str, output_path: str = None, 
                   repo_id: str = None, push_to_hub: bool = False):
    """
    Process images in easy and hard folders and create a HuggingFace dataset with splits.
    
    Args:
        easy_folder: Path to folder containing easy images
        hard_folder: Path to folder containing hard images
        output_path: Optional path to save the dataset locally
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        push_to_hub: Whether to push the dataset to HuggingFace Hub
    """
    # Process easy folder
    print("=" * 60)
    print("Processing EASY folder...")
    print("=" * 60)
    easy_dataset = process_folder(easy_folder)
    
    # Process hard folder
    print("\n" + "=" * 60)
    print("Processing HARD folder...")
    print("=" * 60)
    hard_dataset = process_folder(hard_folder)
    
    # Create DatasetDict with splits
    dataset_dict = DatasetDict({
        "easy": easy_dataset,
        "hard": hard_dataset
    })
    
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Easy split: {len(easy_dataset)} samples")
    print(f"Hard split: {len(hard_dataset)} samples")
    print(f"Total: {len(easy_dataset) + len(hard_dataset)} samples")
    
    # Save locally if output path provided
    if output_path:
        dataset_dict.save_to_disk(output_path)
        print(f"\nDataset saved to: {output_path}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        
        # Load HF_TOKEN from .env file
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token")
        
        print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
        dataset_dict.push_to_hub(repo_id, token=hf_token)
        print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")
    
    return dataset_dict


def merge_datasets(datasets: List[Dataset]) -> Dataset:
    """
    Merge multiple HuggingFace datasets into one.
    
    Args:
        datasets: List of Dataset objects to merge
        
    Returns:
        Merged Dataset
    """
    if not datasets:
        raise ValueError("No datasets provided to merge")
    
    if len(datasets) == 1:
        return datasets[0]
    
    # Concatenate all datasets using concatenate_datasets
    merged = concatenate_datasets(datasets)
    
    return merged


def create_train_val_test_split(dataset: Dataset, train_ratio: float = 0.7, 
                                val_ratio: float = 0.15, test_ratio: float = 0.15,
                                seed: int = 42) -> DatasetDict:
    """
    Split a dataset into train/val/test splits.
    
    Args:
        dataset: Dataset to split
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with 'train', 'validation', and 'test' splits
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle dataset indices
    num_samples = len(dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    # Calculate split sizes
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    # test_size is the remainder
    
    # Split indices
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create splits
    train_dataset = dataset.select(train_indices)
    val_dataset = dataset.select(val_indices)
    test_dataset = dataset.select(test_indices)
    
    print(f"\nSplit Summary:")
    print(f"Train: {len(train_dataset)} samples ({len(train_dataset)/num_samples*100:.1f}%)")
    print(f"Validation: {len(val_dataset)} samples ({len(val_dataset)/num_samples*100:.1f}%)")
    print(f"Test: {len(test_dataset)} samples ({len(test_dataset)/num_samples*100:.1f}%)")
    
    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })


def create_easy_mode_dataset(easy_base_folder: str, output_path: str = None,
                             repo_id: str = None, push_to_hub: bool = False,
                             train_ratio: float = 0.7, val_ratio: float = 0.15, 
                             test_ratio: float = 0.15, seed: int = 42):
    """
    Create an easy mode dataset by merging images from Easy/, Easy/Easy new set/, and Easy/Fair/.
    Creates both a total merged dataset and train/val/test splits.
    
    Args:
        easy_base_folder: Path to the Easy folder containing subfolders
        output_path: Optional path to save the dataset locally
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with 'total' and 'splits' (train/val/test) keys
    """
    base_path = Path(easy_base_folder)
    
    # Define folder paths
    easy_folder = base_path
    easy_new_folder = base_path / "Easy new set"
    fair_folder = base_path / "Fair"
    
    # Process each folder
    datasets = []
    
    print("=" * 60)
    print("Processing EASY folder...")
    print("=" * 60)
    if easy_folder.exists():
        easy_dataset = process_folder(str(easy_folder))
        datasets.append(easy_dataset)
    else:
        print(f"Warning: {easy_folder} does not exist, skipping...")
    
    print("\n" + "=" * 60)
    print("Processing EASY NEW SET folder...")
    print("=" * 60)
    if easy_new_folder.exists():
        easy_new_dataset = process_folder(str(easy_new_folder))
        datasets.append(easy_new_dataset)
    else:
        print(f"Warning: {easy_new_folder} does not exist, skipping...")
    
    print("\n" + "=" * 60)
    print("Processing FAIR folder...")
    print("=" * 60)
    if fair_folder.exists():
        fair_dataset = process_folder(str(fair_folder))
        datasets.append(fair_dataset)
    else:
        print(f"Warning: {fair_folder} does not exist, skipping...")
    
    if not datasets:
        raise ValueError("No datasets were successfully processed")
    
    # Merge all datasets
    print("\n" + "=" * 60)
    print("Merging all datasets...")
    print("=" * 60)
    total_dataset = merge_datasets(datasets)
    
    print(f"\nTotal merged dataset: {len(total_dataset)} samples")
    
    # Create train/val/test splits
    print("\n" + "=" * 60)
    print("Creating train/val/test splits...")
    print("=" * 60)
    splits = create_train_val_test_split(
        total_dataset, 
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create final DatasetDict
    dataset_dict = DatasetDict({
        "total": total_dataset,
        "train": splits["train"],
        "validation": splits["validation"],
        "test": splits["test"]
    })
    
    print("\n" + "=" * 60)
    print("Dataset Summary")
    print("=" * 60)
    print(f"Total: {len(total_dataset)} samples")
    print(f"Train: {len(splits['train'])} samples")
    print(f"Validation: {len(splits['validation'])} samples")
    print(f"Test: {len(splits['test'])} samples")
    
    # Save locally if output path provided
    if output_path:
        dataset_dict.save_to_disk(output_path)
        print(f"\nDataset saved to: {output_path}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        
        # Load HF_TOKEN from .env file
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token")
        
        print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
        dataset_dict.push_to_hub(repo_id, token=hf_token)
        print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")
    
    return dataset_dict


def create_difficulty_levels_dataset(base_folder: str, output_path: str = None,
                                     repo_id: str = None, push_to_hub: bool = False,
                                     train_ratio: float = 0.7, val_ratio: float = 0.15, 
                                     test_ratio: float = 0.15, seed: int = 42):
    """
    Create datasets organized by difficulty levels from 'Easy by difficulty' folder.
    Creates a dataset for each level (with train/val/test splits) and a total merged dataset.
    
    Args:
        base_folder: Path to the "Easy by difficulty" folder containing level subfolders (1, 2, 3)
        output_path: Optional path to save the dataset locally
        repo_id: HuggingFace repository ID (e.g., "username/dataset-name")
        push_to_hub: Whether to push the dataset to HuggingFace Hub
        train_ratio: Proportion of data for training (default: 0.7)
        val_ratio: Proportion of data for validation (default: 0.15)
        test_ratio: Proportion of data for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        DatasetDict with level datasets and total dataset, each with train/val/test splits
    """
    base_path = Path(base_folder)
    
    # Find all level folders (1, 2, 3, etc.)
    level_folders = sorted([d for d in base_path.iterdir() if d.is_dir() and d.name.isdigit()], 
                          key=lambda x: int(x.name))
    
    if not level_folders:
        raise ValueError(f"No level folders found in {base_folder}. Expected folders named 1, 2, 3, etc.")
    
    print("=" * 60)
    print(f"Found {len(level_folders)} difficulty levels: {[f.name for f in level_folders]}")
    print("=" * 60)
    
    level_datasets = {}
    all_datasets = []
    
    # Process each level
    for level_folder in level_folders:
        level_name = level_folder.name
        print("\n" + "=" * 60)
        print(f"Processing LEVEL {level_name}...")
        print("=" * 60)
        
        # Process the level folder
        level_dataset = process_folder(str(level_folder))
        all_datasets.append(level_dataset)
        
        # Create train/val/test splits for this level
        print(f"\nCreating splits for level {level_name}...")
        level_splits = create_train_val_test_split(
            level_dataset,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed
        )
        
        level_datasets[f"level_{level_name}"] = level_splits
        
        print(f"Level {level_name} summary:")
        print(f"  Total: {len(level_dataset)} samples")
        print(f"  Train: {len(level_splits['train'])} samples")
        print(f"  Validation: {len(level_splits['validation'])} samples")
        print(f"  Test: {len(level_splits['test'])} samples")
    
    # Merge all levels into total dataset
    print("\n" + "=" * 60)
    print("Merging all levels into TOTAL dataset...")
    print("=" * 60)
    total_dataset = merge_datasets(all_datasets)
    
    print(f"\nTotal merged dataset: {len(total_dataset)} samples")
    
    # Create train/val/test splits for total
    print("\n" + "=" * 60)
    print("Creating train/val/test splits for TOTAL...")
    print("=" * 60)
    total_splits = create_train_val_test_split(
        total_dataset,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )
    
    # Create final DatasetDict structure (flattened)
    # Structure: {level_1_train, level_1_validation, level_1_test, level_2_train, ..., total_train, total_validation, total_test}
    dataset_dict = DatasetDict()
    
    # Add level splits (flattened)
    for level_name, splits in level_datasets.items():
        for split_name, split_data in splits.items():
            dataset_dict[f"{level_name}_{split_name}"] = split_data
    
    # Add total splits
    for split_name, split_data in total_splits.items():
        dataset_dict[f"total_{split_name}"] = split_data
    
    print("\n" + "=" * 60)
    print("Final Dataset Summary")
    print("=" * 60)
    
    # Group by level for summary
    level_summaries = {}
    for key in dataset_dict.keys():
        if key.startswith("level_"):
            parts = key.split("_")
            level = parts[0] + "_" + parts[1]
            split = parts[2]
            if level not in level_summaries:
                level_summaries[level] = {}
            level_summaries[level][split] = len(dataset_dict[key])
        elif key.startswith("total_"):
            split = key.split("_", 1)[1]
            if "total" not in level_summaries:
                level_summaries["total"] = {}
            level_summaries["total"][split] = len(dataset_dict[key])
    
    for level_name, splits in level_summaries.items():
        print(f"\n{level_name.upper()}:")
        print(f"  Train: {splits.get('train', 0)} samples")
        print(f"  Validation: {splits.get('validation', 0)} samples")
        print(f"  Test: {splits.get('test', 0)} samples")
    
    print(f"\nTotal samples across all levels: {len(total_dataset)}")
    print(f"Total splits in dataset: {len(dataset_dict)}")
    
    # Save locally if output path provided
    if output_path:
        dataset_dict.save_to_disk(output_path)
        print(f"\nDataset saved to: {output_path}")
    
    # Push to HuggingFace Hub if requested
    if push_to_hub:
        if not repo_id:
            raise ValueError("repo_id must be provided when push_to_hub=True")
        
        # Load HF_TOKEN from .env file
        load_dotenv()
        hf_token = os.getenv("HF_TOKEN")
        
        if not hf_token:
            raise ValueError("HF_TOKEN not found in .env file. Please create a .env file with HF_TOKEN=your_token")
        
        print(f"\nPushing dataset to HuggingFace Hub: {repo_id}")
        dataset_dict.push_to_hub(repo_id, token=hf_token)
        print(f"Dataset successfully pushed to: https://huggingface.co/datasets/{repo_id}")
    
    return dataset_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Create HuggingFace dataset from image folders",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create easy mode dataset (merges Easy/, Easy/Easy new set/, Easy/Fair/)
  python utils/create_dataset.py --easy-mode Easy --output ./dataset_easy
  
  # Create difficulty levels dataset (from Easy by difficulty/ folder)
  python utils/create_dataset.py --difficulty-levels "Easy by difficulty" --output ./dataset_levels
  
  # Push difficulty levels dataset to HuggingFace Hub
  python utils/create_dataset.py --difficulty-levels "Easy by difficulty" --repo-id username/curvebench-levels --push-to-hub
  
  # Create traditional easy/hard dataset
  python utils/create_dataset.py Easy Hard --output ./dataset
        """
    )
    
    parser.add_argument("--easy-mode", type=str, default=None, 
                       help="Path to Easy folder for easy mode (merges Easy/, Easy/Easy new set/, Easy/Fair/)")
    parser.add_argument("--difficulty-levels", type=str, default=None,
                       help="Path to 'Easy by difficulty' folder with level subfolders (1, 2, 3, etc.)")
    parser.add_argument("easy_folder", type=str, nargs="?", default=None,
                       help="Path to folder containing easy images (for traditional mode)")
    parser.add_argument("hard_folder", type=str, nargs="?", default=None,
                       help="Path to folder containing hard images (for traditional mode)")
    parser.add_argument("--output", type=str, default=None, help="Path to save the dataset locally")
    parser.add_argument("--repo-id", type=str, default=None, help="HuggingFace repository ID (e.g., username/dataset-name)")
    parser.add_argument("--push-to-hub", action="store_true", help="Push dataset to HuggingFace Hub")
    parser.add_argument("--train-ratio", type=float, default=0.7, help="Proportion for training split (default: 0.7)")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Proportion for validation split (default: 0.15)")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Proportion for test split (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting (default: 42)")
    
    args = parser.parse_args()
    
    if args.difficulty_levels:
        # Difficulty levels mode: process each level folder and create splits
        dataset_dict = create_difficulty_levels_dataset(
            args.difficulty_levels,
            output_path=args.output,
            repo_id=args.repo_id,
            push_to_hub=args.push_to_hub,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        print("\nDifficulty levels dataset created successfully!")
        print(f"Dataset features: {dataset_dict['total_train'].features}")
    elif args.easy_mode:
        # Easy mode: merge Easy/, Easy/Easy new set/, Easy/Fair/ and create splits
        dataset_dict = create_easy_mode_dataset(
            args.easy_mode,
            output_path=args.output,
            repo_id=args.repo_id,
            push_to_hub=args.push_to_hub,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
        print("\nEasy mode dataset created successfully!")
        print(f"Dataset features: {dataset_dict['total'].features}")
    else:
        # Traditional mode: easy/hard splits
        if not args.easy_folder or not args.hard_folder:
            parser.error("Either --easy-mode must be specified, or both easy_folder and hard_folder must be provided")
        
        dataset_dict = create_dataset(
            args.easy_folder, 
            args.hard_folder, 
            output_path=args.output,
            repo_id=args.repo_id,
            push_to_hub=args.push_to_hub
        )
        print("\nDataset created successfully!")
        print(f"Dataset features: {dataset_dict['easy'].features}")

