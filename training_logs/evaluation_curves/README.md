# Evaluation Curves

Evaluation accuracy for the three CurveBench RLVR models across dataset splits
and categories. All W&B metadata has been removed.

## Model Key

| Label | Architecture | Reward | Color |
|---|---|---|---|
| model-a | Qwen3-VL-8B | Tree isomorphism only (weight 1.0) | Blue |
| model-b | Qwen3-VL-8B | Tree (0.7) + node count (0.3) | Red |
| model-c | Gemma 3 12B | Tree (0.7) + node count (0.3) | Green |

## Metric

**Tree isomorphism accuracy**: fraction of examples where the predicted tree is
structurally isomorphic to the ground truth (binary, via NetworkX).

## Full Evaluation Report

All evaluation panels are included in Section 8 of the LaTeX report at
`../report.tex` (compile with `pdflatex` or upload to Overleaf).

## Notes

- W&B export with all identifying metadata stripped.
- Run names replaced with model labels above.
