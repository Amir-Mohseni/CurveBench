# Image2Tree Dataset Creator

This script processes images in easy and hard folders and creates a HuggingFace dataset with splits containing the image, number of nodes (areas), and tree structure extracted from contour hierarchies.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Setup

Create a `.env` file in the project root with your HuggingFace token:

```
HF_TOKEN=your_huggingface_token_here
```

## Usage

Process images in easy and hard folders and create a HuggingFace dataset with splits:

```bash
python utils/create_dataset.py <easy_folder> <hard_folder> [--output <output_path>] [--repo-id <repo_id>] [--push-to-hub]
```

### Arguments

- `easy_folder`: Path to the folder containing easy images
- `hard_folder`: Path to the folder containing hard images
- `--output` (optional): Path where to save the HuggingFace dataset locally. If not provided, the dataset is created in memory only.
- `--repo-id` (optional): HuggingFace repository ID (e.g., `username/dataset-name`). Required if using `--push-to-hub`.
- `--push-to-hub`: Push the dataset to HuggingFace Hub (requires `--repo-id` and `HF_TOKEN` in `.env`)

### Examples

```bash
# Process easy and hard folders and save dataset locally
python utils/create_dataset.py ./easy ./hard --output ./dataset

# Process and push to HuggingFace Hub
python utils/create_dataset.py ./easy ./hard --repo-id username/curvebench --push-to-hub

# Process without saving (dataset in memory only)
python utils/create_dataset.py ./easy ./hard
```

## Supported Image Formats

The script supports the following image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Dataset Format

The created dataset is a `DatasetDict` with two splits: `easy` and `hard`. Each split contains three columns:

1. **image**: PIL Image object of the processed image
2. **num_nodes**: Integer representing the number of merged regions (nodes) in the tree (excluding the virtual root)
3. **tree**: List of tuples `(parent, child)` representing the tree edges (root is always node 0)

## How It Works

For each image, the script:

1. Converts the image to grayscale
2. Applies inverse binary thresholding
3. Performs morphological erosion to thin lines
4. Detects contours with hierarchical relationships using OpenCV
5. Builds a tree structure from contour hierarchies
6. Merges outer and inner contour pairs into logical regions
7. Converts the tree to edge format
8. Counts the number of nodes (regions)

## Output

The script prints progress information and summary statistics:
- Number of images found
- Processing progress for each image
- Final dataset statistics
- Dataset features

