# CurveBench Environment

A Prime Intellect Environments Hub environment for evaluating vision-language models on hierarchical tree structure extraction from images.

## Task Description

Given an image containing nested regions/shapes, the model must extract the hierarchical tree structure representing the parent-child relationships between these regions. The tree structure is represented as a list of edges `(parent, child)`, where:
- The root node is always 0
- Each region is assigned a unique node number
- Edges represent containment relationships (a parent region contains a child region)

## Dataset

This environment uses the [CurveBench-Easy](https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy) dataset, which contains:
- **Level 1**: 120 samples (easiest)
- **Level 2**: 112 samples (medium)
- **Level 3**: 68 samples (hardest)
- **Total**: 300 samples across all difficulty levels

Each sample includes:
- An image with nested regions
- Ground truth tree structure as edge list
- Number of nodes (excluding root)

## Evaluation

Models are evaluated based on tree isomorphism:
- **Reward = 1.0**: If the predicted tree structure is isomorphic to the ground truth
- **Reward = 0.0**: Otherwise

Tree isomorphism means the structures are equivalent up to node relabeling, allowing for flexibility in how nodes are numbered while maintaining the same hierarchical relationships.

## Expected Response Format

Models should return the tree structure in one of these formats:

**JSON format (preferred):**
```json
{"tree": [[0, 1], [0, 2], [1, 3]]}
```

**Simple format:**
```
TREE: (0,1) (0,2) (1,3)
```

**List format:**
```
[[0,1], [0,2], [1,3]]
```

## Installation

```bash
prime env install amirmohseni/curvebench-env
```

## Usage

```python
import verifiers as vf
from curvebench_env import load_environment

env = load_environment()
# Use with verifiers evaluation framework
```

## Citation

If you use this environment, please cite the CurveBench dataset.
