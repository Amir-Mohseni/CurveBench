#!/usr/bin/env python3
"""
Script to process images in a folder and create a HuggingFace dataset
with image, number of nodes, and tree structure.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from datasets import Dataset, DatasetDict
from typing import List, Tuple, Dict
import json


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


def create_dataset(image_folder: str, output_path: str = None):
    """
    Process all images in a folder and create a HuggingFace dataset.
    
    Args:
        image_folder: Path to folder containing images
        output_path: Optional path to save the dataset
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
    
    # Save if output path provided
    if output_path:
        dataset.save_to_disk(output_path)
        print(f"Dataset saved to: {output_path}")
    
    return dataset


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create HuggingFace dataset from images")
    parser.add_argument("image_folder", type=str, help="Path to folder containing images")
    parser.add_argument("--output", type=str, default=None, help="Path to save the dataset")
    
    args = parser.parse_args()
    
    dataset = create_dataset(args.image_folder, args.output)
    print("\nDataset created successfully!")
    print(f"Dataset features: {dataset.features}")

