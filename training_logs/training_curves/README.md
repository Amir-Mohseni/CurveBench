# Training Curves

Static exports of RLVR training metrics for all three CurveBench training runs.
All W&B run metadata, usernames, team names, and project names have been removed.

## Model Key

| Label | Architecture | Reward configuration |
|---|---|---|
| model-a | qwen3-vl-8b-tree-only | Tree isomorphism reward only (weight 1.0) |
| model-b | qwen3-vl-8b-tree-node | Tree (0.7) + node count (0.3) |
| model-c | gemma3-12b-rlvr | Tree (0.7) + node count (0.3) |

All models trained with GRPO on CurveBench-Easy (`total_train`, 210 examples).

## Reward Components

- **tree_reward** (binary): 1.0 if the predicted tree is isomorphic to the
  ground truth; 0.0 otherwise.
- **node_count_reward** (binary): 1.0 if the predicted node count matches the
  ground truth; 0.0 otherwise.
- **total_reward** = reward_weight[0] × tree_reward + reward_weight[1] × node_count_reward

## Full Training Report

The complete W&B LaTeX export (all panels, all runs) is in:

```
../Curvebench Training Results.zip
```

Compile with:
```bash
pdflatex report.tex
```

Or upload to Overleaf (New Project → Upload Project).

## Notes

- W&B export performed with all user/team/project metadata stripped.
- Run IDs replaced with model labels above.
- The LaTeX export uses the `neurips_2019.sty` style file included in the ZIP.
