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

| Model | Architecture | Reward |
|---|---|---|
| model-a | Qwen3-VL-8B | Tree isomorphism only (weight 1.0) |
| model-b | Qwen3-VL-8B | Tree (0.7) + node count (0.3) |
| model-c | Gemma 3 12B | Tree (0.7) + node count (0.3) |

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

The evaluation environments are self-contained Python packages that score model outputs using tree isomorphism (via NetworkX).

```bash
pip install -r curvebench_env/requirements.txt
python curvebench_env/curvebench_env.py --split total_test
```

See [`curvebench_env/README.md`](curvebench_env/README.md) for full usage details, including multi-GPU batch evaluation and the expected response format.

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

---

## Contact

- Author: Amir Mohseni — `amir.mohseni@student.maastrichtuniversity.nl`
- Hugging Face: <https://huggingface.co/AmirMohseni>
