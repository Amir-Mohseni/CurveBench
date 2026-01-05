# Image2Tree Dataset Creator

This script processes images in a folder and creates a HuggingFace dataset containing the image, number of nodes (areas), and tree structure extracted from contour hierarchies.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Process images in a folder and create a HuggingFace dataset:

```bash
python create_dataset.py <image_folder> [--output <output_path>]
```

### Arguments

- `image_folder`: Path to the folder containing images to process
- `--output` (optional): Path where to save the HuggingFace dataset. If not provided, the dataset is created in memory only.

### Example

```bash
# Process images in ./images folder and save dataset to ./dataset
python create_dataset.py ./images --output ./dataset

# Process images without saving
python create_dataset.py ./images
```

## Supported Image Formats

The script supports the following image formats:
- PNG (.png)
- JPEG (.jpg, .jpeg)
- BMP (.bmp)
- TIFF (.tiff, .tif)

## Dataset Format

The created dataset contains three columns:

1. **image**: PIL Image object of the processed image
2. **num_nodes**: Integer representing the number of merged regions (nodes) in the tree (excluding the virtual root)
3. **tree**: List of tuples `(parent, child)` representing the tree edges

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

