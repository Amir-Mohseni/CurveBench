# CurveBench Environment (Easy)

A Prime Intellect Environments Hub environment for evaluating vision-language
models on hierarchical tree structure extraction from images.

## Task Description

Given an image containing nested regions/shapes, the model must extract the
hierarchical tree structure representing the parent-child relationships between
these regions.

- The root node is always 0
- Each region is assigned a unique node number
- Edges represent containment relationships (a parent region contains a child region)

## Dataset

This environment uses the [`AmirMohseni/CurveBench-Easy`](https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy) dataset on Hugging Face Hub:

| Split   | Level 1 | Level 2 | Level 3 | Total |
|---------|---------|---------|---------|-------|
| Train   | 84      | 78      | 47      | 210   |
| Val     | 18      | 16      | 10      | 45    |
| Test    | 18      | 18      | 11      | 45    |

Available split names: `total_train`, `total_validation`, `total_test`,
`level_1_train`, `level_1_test`, `level_2_train`, `level_2_test`,
`level_3_train`, `level_3_test`, etc.

## Expected Response Format

Models should respond inside `<answer>...</answer>` tags. The first line is the
number of nodes (excluding the root), followed by one edge per line as `u v`
(child, parent):

```
<answer>
3
1 0
2 0
3 1
</answer>
```

## Scoring

| Reward             | Weight | Description                                           |
|--------------------|--------|-------------------------------------------------------|
| `tree_reward`      | 0.7    | 1.0 if predicted tree is isomorphic to ground truth   |
| `node_count_reward`| 0.3    | 1.0 if predicted node count matches ground truth      |

## Installation

The environment is also published to the Prime Intellect Environments Hub as
`amirmohseni/curvebench-env`. Install via:

```bash
prime env install amirmohseni/curvebench-env
```

Or install locally:

```bash
pip install -e .
```

## Usage

```python
from curvebench_env import load_environment

env = load_environment(split="total_test")
```

## License

MIT License — see `../../LICENSE-CODE.txt`
