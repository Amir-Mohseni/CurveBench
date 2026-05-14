# CurveBench-Hard Environment

A Prime Intellect Environments Hub environment for evaluating vision-language
models on hierarchical tree structure extraction from complex images.

## Task Description

Given an image containing nested regions/shapes, the model must extract the
hierarchical tree structure representing the parent-child relationships between
these regions.

- The root node is always 0
- Each region is assigned a unique node number
- Edges represent containment relationships (a parent region contains a child region)

## Dataset

This environment uses the [`AmirMohseni/CurveBench`](https://huggingface.co/datasets/AmirMohseni/CurveBench) dataset on Hugging Face Hub (Hard split):

| Category     | Count |
|-------------|-------|
| Polygon      | 199   |
| Topographical| 100   |
| Maze         | 100   |
| Counting     | 57    |
| Combined     | 456   |

## Expected Response Format

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
`amirmohseni/curvebench-hard-env`. Install via:

```bash
prime env install amirmohseni/curvebench-hard-env
```

Or install locally:

```bash
pip install -e .
```

## License

MIT License — see `../../LICENSE-CODE.txt`
