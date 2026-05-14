"""
CurveBench-Easy dataset loading script.

The full dataset is hosted on the Hugging Face Hub at
``AmirMohseni/CurveBench-Easy``.
"""

from datasets import load_dataset


DATASET_ID = "AmirMohseni/CurveBench-Easy"

AVAILABLE_SPLITS = [
    "total_train",
    "total_validation",
    "total_test",
    "level_1_train",
    "level_1_validation",
    "level_1_test",
    "level_2_train",
    "level_2_validation",
    "level_2_test",
    "level_3_train",
    "level_3_validation",
    "level_3_test",
]


def load_easy(split: str = "total_train"):
    """Load a CurveBench-Easy split.

    Args:
        split: One of the available split names (see AVAILABLE_SPLITS).

    Returns:
        A HuggingFace Dataset object with fields:
          - image: PIL Image
          - num_nodes: int (number of tree nodes including root)
          - tree: List[List[int]] (edge list [[parent, child], ...])
    """
    return load_dataset(DATASET_ID, split=split)


if __name__ == "__main__":
    print("Loading CurveBench-Easy total_train split...")
    ds = load_easy("total_train")
    print(f"Loaded {len(ds)} examples.")
    print(f"Features: {ds.features}")
    sample = ds[0]
    print(f"\nSample 0:")
    print(f"  num_nodes: {sample['num_nodes']}")
    print(f"  tree: {sample['tree']}")
    print(f"  image size: {sample['image'].size}")
