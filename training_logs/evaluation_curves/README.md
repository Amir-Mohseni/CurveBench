# Evaluation Curves

Evaluation accuracy for all CurveBench models across dataset splits and
categories. All W&B metadata has been removed.

## Model Key

| Label | Architecture | Reward |
|---|---|---|
| model-a | qwen3-vl-8b-tree-only | Tree only |
| model-b | qwen3-vl-8b-tree-node | Tree + node count |
| model-c | gemma3-12b-rlvr | Tree + node count |
| baseline-qwen3-8b | Qwen3-VL-8B-Instruct | zero-shot |
| baseline-gemma3-12b | Gemma-3-12B-IT | zero-shot |
| baseline-gemma3-27b | Gemma-3-27B-IT | zero-shot |
| baseline-qwen3-235b | Qwen3-VL-235B-A22B-Thinking | zero-shot |

## Metric

**Tree isomorphism accuracy**: fraction of examples where the predicted tree is
structurally isomorphic to the ground truth (binary, via NetworkX).

## Full Evaluation Report

All evaluation panels are included in Section 8 of the LaTeX report:

```
../Curvebench Training Results.zip  →  Section 8 (Panels 0–46)
```

## Notes

- W&B export with all identifying metadata stripped.
- Run names replaced with model labels above.
