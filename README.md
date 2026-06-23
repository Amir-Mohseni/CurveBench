# CurveBench

[![arXiv](https://img.shields.io/badge/arXiv-2605.14068-b31b1b.svg)](https://arxiv.org/abs/2605.14068)
[![Hugging Face Collection](https://img.shields.io/badge/Hugging%20Face-Collection-yellow)](https://huggingface.co/collections/AmirMohseni/curvebench)
[![Datasets](https://img.shields.io/badge/Datasets-CurveBench-blue)](https://huggingface.co/collections/AmirMohseni/curvebench)
[![License: MIT](https://img.shields.io/badge/Code-MIT-green.svg)](supplement/LICENSE-CODE.txt)
[![License: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

**CurveBench** is a benchmark for evaluating **topological reasoning** in vision-language models (VLMs). Given an image of non-crossing closed curves drawn on a plane, a model must infer the global containment structure as a rooted tree, where each node is a bounded region and each edge connects a region to the one it is directly contained in.

This repository contains the reference code, datasets, evaluation environments, training recipes, and logs for the CurveBench paper.

> **Paper:** [CurveBench: A Benchmark for Exact Topological Reasoning over Nested Jordan Curves](https://arxiv.org/abs/2605.14068)

---

## Overview

CurveBench tests whether VLMs can recover the exact containment structure induced by nested, non-intersecting closed curves. The target output is a rooted tree representing which regions directly contain which other regions.

The benchmark includes:

- **CurveBench-Easy**: foundational split with exhaustive topology up to 5 curves.
- **CurveBench-Hard**: harder split covering Polygon, Topographical, Maze, and Counting categories.
- **Verifiable evaluation** using tree isomorphism.
- **RLVR training recipes** using GRPO and binary verifiable rewards.
- **Prime Intellect evaluation environments** for reproducible model evaluation.

---

## Dataset

| Split | Size | Description |
|---|---:|---|
| **CurveBench-Easy** | 300 images | Foundational split with ≤5 curves and exhaustive topology. Used for RLVR training and evaluation. |
| **CurveBench-Hard** | 456 images | Hard split across four categories: Polygon, Topographical, Maze, and Counting. |

### Links

| Resource | Link |
|---|---|
| Hugging Face collection | [AmirMohseni/curvebench](https://huggingface.co/collections/AmirMohseni/curvebench) |
| CurveBench-Easy | [AmirMohseni/CurveBench-Easy](https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy) |
| CurveBench-Hard | [AmirMohseni/CurveBench](https://huggingface.co/datasets/AmirMohseni/CurveBench) |

### Loading the datasets

```python
from datasets import load_dataset

ds_easy = load_dataset("AmirMohseni/CurveBench-Easy", split="total_train")
ds_hard = load_dataset("AmirMohseni/CurveBench", split="combined")
```

---

## Models

We train three models with GRPO on CurveBench-Easy using verifiable binary rewards.

| Model | Base Model | Reward |
|---|---|---|
| [curvebench-qwen3-vl-8b-only-tree](https://huggingface.co/AmirMohseni/curvebench-qwen3-vl-8b-only-tree) | Qwen/Qwen3-VL-8B-Thinking | Tree isomorphism only |
| [curvebench-qwen3-vl-8b](https://huggingface.co/AmirMohseni/curvebench-qwen3-vl-8b) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOn6cBZGpEYZnItTdJA0ATQsn9kp4O3f?usp=sharing) | Qwen/Qwen3-VL-8B-Thinking | Tree isomorphism + node count |
| [curvebench-gemma-3-12b](https://huggingface.co/AmirMohseni/curvebench-gemma-3-12b) | google/gemma-3-12b-it | Tree isomorphism + node count |

The rewards are:

- **Tree reward**: binary reward based on exact tree isomorphism.
- **Node-count reward**: binary auxiliary reward for predicting the correct number of regions/nodes.

---

## RLVR Training

Training is built on a fork of TRL with multimodal GRPO support:

[github.com/AmirTuring/trl/tree/curvebench](https://github.com/AmirTuring/trl/tree/curvebench)

### Setup

```bash
git clone -b curvebench https://github.com/AmirTuring/trl
cd trl
sh setup.sh
cp examples/accelerate_configs/single_gpu.yaml accelerate_config.yaml
```

### Run training

```bash
# Qwen3-VL-8B, tree-only reward
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_config.yaml

# Qwen3-VL-8B, tree + node count reward
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_treenode_config.yaml

# Gemma 3 12B, tree + node count reward
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_gemma3_12b_config.yaml
```

Training configs are in:

```text
trl/examples/cli_configs/
```

Reward functions are in:

```text
trl/reward_funcs/
```

---

## Evaluation

CurveBench evaluation environments are available on the Prime Intellect Environments Hub, so models can be evaluated without local setup.

| Environment | Splits | Prime Intellect |
|---|---|---|
| CurveBench-Easy | `total_test`, `level_1_test`, `level_2_test`, `level_3_test` | [amirmohseni/curvebench-env](https://app.primeintellect.ai/dashboard/environments/amirmohseni/curvebench-env) |
| CurveBench-Hard | `combined`, `polygon`, `topographical`, `maze`, `counting` | [amirmohseni/curvebench-hard-env](https://app.primeintellect.ai/dashboard/environments/amirmohseni/curvebench-hard-env) |

### Evaluate on CurveBench-Easy

```bash
prime eval run amirmohseni/curvebench-env \
  -m "your-model-name" \
  -n -1 \
  -a '{"split": "total_test"}' \
  -r 4
```

### Evaluate on CurveBench-Hard

```bash
prime eval run amirmohseni/curvebench-hard-env \
  -m "your-model-name" \
  -n -1 \
  -a '{"split": "combined"}' \
  -r 4
```

For local evaluation, see the self-contained environment packages:

- [`curvebench_env/`](curvebench_env/README.md)
- [`curvebench_hard_env/`](curvebench_hard_env/README.md)

---

## Training Logs

Static W&B exports with all user, team, and project metadata removed are available in [`training_logs/`](training_logs/).

| Path | Description |
|---|---|
| [`training_logs/training_curves/`](training_logs/training_curves/) | GRPO reward and loss curves for all three training runs |
| [`training_logs/evaluation_curves/`](training_logs/evaluation_curves/) | Tree-isomorphism accuracy across dataset splits and categories |
| [`training_logs/report.tex`](training_logs/report.tex) | Full LaTeX report; compile with `pdflatex` or upload to Overleaf |

---

## Repository Structure

```text
CurveBench/
├── trl/                         # RLVR training fork: GRPO + VLM support
│   └── examples/
│       ├── cli_configs/         # Per-model training configs
│       └── scripts/             # Training entry points
├── curvebench_env/              # Evaluation environment: CurveBench-Easy
├── curvebench_hard_env/         # Evaluation environment: CurveBench-Hard
├── training_logs/
│   ├── training_curves/         # Training metric charts
│   └── evaluation_curves/       # Evaluation accuracy charts
├── curvedata/                   # Raw image data and annotations
├── dataset_levels/              # Split-level dataset files
└── supplement/                  # Full paper supplement
```

---

## Licenses

| Component | License |
|---|---|
| Dataset images and annotations | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Code | [MIT](supplement/LICENSE-CODE.txt) |

---

## Citation

If you use CurveBench in your research, please cite:

```bibtex
@misc{mohseni2026curvebench,
      title={CurveBench: A Benchmark for Exact Topological Reasoning over Nested Jordan Curves},
      author={Amirreza Mohseni and Mona Mohammadi and Morteza Saghafian and Naser Talebizadeh Sardari},
      year={2026},
      eprint={2605.14068},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.14068},
}
```
