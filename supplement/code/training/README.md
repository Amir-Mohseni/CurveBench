# CurveBench Training Code

RLVR (GRPO) training code for CurveBench. Based on a fork of the
[TRL library](https://github.com/huggingface/trl) with multimodal (VLM)
GRPO support added for the CurveBench region-tree task.

The full training fork is available at
<https://github.com/AmirTuring/trl/tree/curvebench>.

---

## Setup

```bash
git clone -b curvebench https://github.com/AmirTuring/trl
cd trl
sh setup.sh
cp examples/accelerate_configs/single_gpu.yaml accelerate_config.yaml
# or for multi-GPU:
cp examples/accelerate_configs/multi_gpu.yaml accelerate_config.yaml
```

---

## Training Commands

### model-a — Qwen3-VL-8B, tree-only reward

```bash
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_config.yaml
```

### model-b — Qwen3-VL-8B, tree + node count reward

```bash
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_treenode_config.yaml
```

### model-c — Gemma 3 12B, tree + node count reward

```bash
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/grpo_region_tree.py \
  --config examples/cli_configs/grpo_region_tree_gemma3_12b_config.yaml
```

### SFT baseline

```bash
accelerate launch --config_file accelerate_config.yaml \
  examples/scripts/sft_region_tree.py \
  --config examples/cli_configs/sft_region_tree_config.yaml
```

---

## Key Files in This Directory

| File | Description |
|---|---|
| `reward_funcs/region_tree_reward.py` | Tree isomorphism and node-count reward functions |
| `examples/cli_configs/grpo_region_tree_config.yaml` | Training config for model-a (Qwen3-VL-8B, tree-only) |
| `examples/cli_configs/grpo_region_tree_treenode_config.yaml` | Training config for model-b (Qwen3-VL-8B, tree + node count) |
| `examples/cli_configs/grpo_region_tree_gemma3_12b_config.yaml` | Training config for model-c (Gemma 3 12B, tree + node count) |

---

## Reward Function

Training uses a weighted sum of two verifiable binary rewards:

| Reward | Weight | Description |
|---|---|---|
| `tree_correctness_reward` | 0.7 | 1.0 if predicted tree is isomorphic to ground truth |
| `num_nodes_reward` | 0.3 | 1.0 if predicted node count matches ground truth |

Both rewards are computed deterministically without a learned reward model.

---

## Model Labels

| Label | Architecture | Reward | Config file |
|---|---|---|---|
| model-a | qwen3-vl-8b-tree-only | Tree isomorphism only (weight 1.0) | `grpo_region_tree_config.yaml` |
| model-b | qwen3-vl-8b-tree-node | Tree (0.7) + node count (0.3) | `grpo_region_tree_treenode_config.yaml` |
| model-c | gemma3-12b-rlvr | Tree (0.7) + node count (0.3) | `grpo_region_tree_gemma3_12b_config.yaml` |

---

## License

Code in this directory: MIT License (see `../../LICENSE-CODE.txt`).
The TRL base library is Apache-2.0 licensed.
