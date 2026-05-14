"""Build and push the CurveBench dataset to Hugging Face.

For each image, runs OpenCV contour-based tree extraction to compute
num_nodes and tree edges before uploading.
"""

import sys
from pathlib import Path

from datasets import Dataset, DatasetDict, Features, Image, Sequence, Value
from huggingface_hub import HfApi

# Make utils importable
sys.path.insert(0, str(Path(__file__).parent))
from utils.create_dataset import process_image

CURVEDATA_DIR = Path(__file__).parent / "curvedata"

CATEGORIES = ["Counting", "Maze", "Polygon", "Topographical"]

FOLDER_NAMES = {
    "Counting": "Counting",
    "Maze": "Maze",
    "Polygon": "Polygon",
    "Topographical": "Topographical ",  # trailing space on disk
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


def collect_samples(category: str) -> list[dict]:
    folder = CURVEDATA_DIR / FOLDER_NAMES[category]
    samples = []
    for f in sorted(folder.iterdir()):
        if f.suffix in IMAGE_EXTENSIONS:
            try:
                result = process_image(f)
                samples.append({
                    "image": str(f),
                    "category": category,
                    "filename": f.name,
                    "num_nodes": result["num_nodes"],
                    "tree": str(result["tree"]),
                })
                print(f"  {f.name}: {result['num_nodes']} nodes, {len(result['tree'])} edges")
            except Exception as e:
                print(f"  ERROR {f.name}: {e}")
    return samples


FIELDS = ("image", "category", "filename", "num_nodes", "tree")


def make_dataset(samples: list[dict], features: Features) -> Dataset:
    return Dataset.from_dict(
        {k: [s[k] for s in samples] for k in FIELDS},
        features=features,
    )


def main():
    features = Features({
        "image": Image(),
        "category": Value("string"),
        "filename": Value("string"),
        "num_nodes": Value("int32"),
        "tree": Value("string"),
    })

    splits = {}
    all_samples = []

    for cat in CATEGORIES:
        print(f"\n{'=' * 50}")
        print(f"Processing {cat}...")
        print("=" * 50)
        samples = collect_samples(cat)
        print(f"{cat}: {len(samples)} images processed")
        splits[cat.lower()] = make_dataset(samples, features)
        all_samples.extend(samples)

    print(f"\nTotal: {len(all_samples)} images")
    splits["combined"] = make_dataset(all_samples, features)

    ds = DatasetDict(splits)
    print(ds)

    api = HfApi()
    api.create_repo("AmirMohseni/CurveBench", repo_type="dataset", exist_ok=True)
    ds.push_to_hub("AmirMohseni/CurveBench")
    print("Done — pushed to https://huggingface.co/datasets/AmirMohseni/CurveBench")


if __name__ == "__main__":
    main()
