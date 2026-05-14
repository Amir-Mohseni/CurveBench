"""
CurveBench (Hard) dataset loading script.

The full dataset is hosted on the Hugging Face Hub at
``AmirMohseni/CurveBench``.
"""

import ast
from datasets import load_dataset


DATASET_ID = "AmirMohseni/CurveBench"

AVAILABLE_SPLITS = [
    "polygon",
    "topographical",
    "maze",
    "counting",
    "combined",
]

CATEGORIES = ["Polygon", "Topographical", "Maze", "Counting"]


def load_hard(split: str = "combined"):
    """Load a CurveBench (Hard) split.

    Args:
        split: One of 'polygon', 'topographical', 'maze', 'counting', 'combined'.

    Returns:
        A HuggingFace Dataset object with fields:
          - image: PIL Image
          - category: str (one of CATEGORIES)
          - filename: str (original image filename)
          - num_nodes: int (number of tree nodes including root)
          - tree: str (stringified list of (parent, child) tuples)
    """
    return load_dataset(DATASET_ID, split=split)


def parse_tree(tree_str: str):
    """Parse the stringified tree field back to a list of (parent, child) tuples.

    Args:
        tree_str: String like "[(0, 1), (0, 2), (1, 3)]"

    Returns:
        List of (int, int) tuples.
    """
    return ast.literal_eval(tree_str)


if __name__ == "__main__":
    print("Loading CurveBench combined split...")
    ds = load_hard("combined")
    print(f"Loaded {len(ds)} examples.")
    print(f"Features: {ds.features}")
    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  category: {sample['category']}")
    print(f"  filename: {sample['filename']}")
    print(f"  num_nodes: {sample['num_nodes']}")
    print(f"  tree (first 3 edges): {parse_tree(sample['tree'])[:3]}")
    print(f"  image size: {sample['image'].size}")
