# CurveBench

**CurveBench** is a benchmark for evaluating **topological reasoning** in vision-language models (VLMs). Given an image of non-crossing closed curves drawn on a plane, a model must infer the global containment structure as a rooted tree — where each node is a bounded region and each edge connects a region to the one it is directly contained in.

This repository is the reference code for the CurveBench paper.

---

## Dataset

| Split | Size | Description |
|---|---|---|
| **CurveBench-Easy** | 300 images | Foundational split (≤5 curves, exhaustive topology). Used for RLVR training and evaluation. |
| **CurveBench (Hard)** | 456 images | Hard split across four categories: Polygon, Topographical, Maze, Counting. |

- HF collection: <https://huggingface.co/collections/AmirMohseni/curvebench>
- CurveBench-Easy: <https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy>
- CurveBench (Hard): <https://huggingface.co/datasets/AmirMohseni/CurveBench>

```python
from datasets import load_dataset

ds_easy = load_dataset("AmirMohseni/CurveBench-Easy", split="total_train")
ds_hard = load_dataset("AmirMohseni/CurveBench", split="combined")
```

---

## RLVR Training

We train three models with GRPO on CurveBench-Easy using two verifiable binary rewards:

| Model | Base Model | Reward |
|---|---|---|
| [curvebench-qwen3-vl-8b-only-tree](https://huggingface.co/AmirMohseni/curvebench-qwen3-vl-8b-only-tree) | Qwen/Qwen3-VL-8B-Thinking | Tree isomorphism only (weight 1.0) |
| [curvebench-qwen3-vl-8b](https://huggingface.co/AmirMohseni/curvebench-qwen3-vl-8b) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1qOn6cBZGpEYZnItTdJA0ATQsn9kp4O3f?usp=sharing) | Qwen/Qwen3-VL-8B-Thinking | Tree (0.7) + node count (0.3) |
| [curvebench-gemma-3-12b](https://huggingface.co/AmirMohseni/curvebench-gemma-3-12b) | google/gemma-3-12b-it | Tree (0.7) + node count (0.3) |

Training is built on a fork of the TRL library with multimodal GRPO support:
<https://github.com/AmirTuring/trl/tree/curvebench>

### Setup

```bash
git clone -b curvebench https://github.com/AmirTuring/trl
cd trl
sh setup.sh
cp examples/accelerate_configs/single_gpu.yaml accelerate_config.yaml
```

### Run training

```bash
# model-a — Qwen3-VL-8B, tree-only reward
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_config.yaml

# model-b — Qwen3-VL-8B, tree + node count reward
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_treenode_config.yaml

# model-c — Gemma 3 12B, tree + node count reward
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_gemma3_12b_config.yaml
```

The training configs live in `trl/examples/cli_configs/`. The reward functions are in `trl/reward_funcs/`.

---

## Evaluation

The evaluation environments are also hosted on Prime Intellect Environments Hub — no local setup required:

| Environment | Split | Prime Intellect |
|---|---|---|
| CurveBench-Easy | `total_test`, `level_1_test`, `level_2_test`, `level_3_test` | [amirmohseni/curvebench-env](https://app.primeintellect.ai/dashboard/environments/amirmohseni/curvebench-env) |
| CurveBench-Hard | `combined`, `polygon`, `topographical`, `maze`, `counting` | [amirmohseni/curvebench-hard-env](https://app.primeintellect.ai/dashboard/environments/amirmohseni/curvebench-hard-env) |

```bash
# Evaluate any model on CurveBench-Easy
prime eval run amirmohseni/curvebench-env \
  -m "your-model-name" \
  -n -1 \
  -a '{"split": "total_test"}' \
  -r 4

# Evaluate any model on CurveBench-Hard
prime eval run amirmohseni/curvebench-hard-env \
  -m "your-model-name" \
  -n -1 \
  -a '{"split": "combined"}' \
  -r 4
```

For local evaluation, the environments are self-contained Python packages in [`curvebench_env/`](curvebench_env/README.md) and [`curvebench_hard_env/`](curvebench_hard_env/README.md).

---

## Training Logs

Static W&B exports (all user/team/project metadata removed) are in [`training_logs/`](training_logs/):

- [`training_logs/training_curves/`](training_logs/training_curves/) — GRPO reward and loss curves for all three runs
- [`training_logs/evaluation_curves/`](training_logs/evaluation_curves/) — tree isomorphism accuracy across dataset splits and categories
- [`training_logs/report.tex`](training_logs/report.tex) — full LaTeX report (compile with `pdflatex` or upload to Overleaf)

---

## Repository Structure

```
CurveBench/
├── trl/                         ← RLVR training fork (GRPO + VLM support)
│   └── examples/
│       ├── cli_configs/         ← Per-model training configs (YAML)
│       └── scripts/             ← Training entry points
├── curvebench_env/              ← Evaluation environment (CurveBench-Easy)
├── curvebench_hard_env/         ← Evaluation environment (CurveBench Hard)
├── training_logs/
│   ├── training_curves/         ← Training metric charts (PNG)
│   └── evaluation_curves/       ← Evaluation accuracy charts (PNG)
├── curvedata/                   ← Raw image data and annotations
├── dataset_levels/              ← Split-level dataset files
└── supplement/                  ← Full paper supplement (code, datasets, envs)
```

---

## Licenses

| Component | License |
|---|---|
| Dataset images and annotations | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| All code | [MIT](supplement/LICENSE-CODE.txt) |

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
