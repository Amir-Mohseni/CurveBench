# CurveBench Artifacts Inventory

Inventory of all CurveBench artifacts referenced by the NeurIPS 2026 ED Track
single-blind dataset paper submission.

---

## 1. Hugging Face datasets

| Artifact | URL | Croissant file |
|---|---|---|
| `CurveBench-Easy` (300 images) | <https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy> | `croissant/curvebench-easy-croissant.json` |
| `CurveBench` (Hard, 456 images) | <https://huggingface.co/datasets/AmirMohseni/CurveBench> | `croissant/curvebench-croissant.json` |

Landing page (collection): <https://huggingface.co/collections/AmirMohseni/curvebench>

## 2. Code repositories

| Artifact | URL | Included in this supplement under |
|---|---|---|
| Main repository (benchmark construction + evaluation harness) | <https://github.com/Amir-Mohseni/CurveBench> | `code/benchmark_construction/`, `code/evaluation/` |
| TRL fork with multimodal GRPO support (training) | <https://github.com/AmirTuring/trl/tree/curvebench> | `code/training/` (key files only) |

## 3. Prime Intellect Environments Hub

| Artifact | ID on Prime Intellect | Included in this supplement under |
|---|---|---|
| CurveBench-Easy verifier environment | `amirmohseni/curvebench-env` | `environments/curvebench_easy/` |
| CurveBench-Hard verifier environment | `amirmohseni/curvebench-hard-env` | `environments/curvebench_hard/` |

## 4. Local supplement files

| File | Purpose |
|---|---|
| `README.md` | Top-level reviewer instructions |
| `LICENSE-DATASET.txt` | CC BY 4.0 (data + annotations) |
| `LICENSE-CODE.txt` | MIT (code + environments) |
| `croissant/*.json` | Validated Croissant 1.1 metadata, including the NeurIPS-required minimal RAI fields |
| `croissant/validation_reports/*.md` | Saved output from the official NeurIPS Croissant validator |
| `datasets/curvebench-easy/`, `datasets/curvebench/` | Per-dataset cards, loading scripts, and 10-row sample JSONL |
| `code/` | Subset of code repositories used for benchmark construction, evaluation, and training |
| `environments/` | Self-contained Prime Intellect verifier environments |
| `logs/training_curves/`, `logs/evaluation_curves/` | Static PNG/CSV exports of W&B runs (live W&B project linked from paper) |
| `scripts/test_croissant_loading.py` | Loads both Croissant files and verifies all required core + RAI fields |
| `croissant_loading_test_output.txt` | Output from the test script above |
| `croissant_validation_report.md` | Combined report from `mlcroissant`, NeurIPS croissant-checker, and NeurIPS rai-checker |
| `dataset_statistics.json` | Verified per-split row counts |
| `MANIFEST.json` | File list with sizes and checksums |
| `SHA256SUMS.txt` | SHA-256 checksums of every file in the supplement |

## 5. Files explicitly NOT in this supplement

- `.git/` history
- `.env` files or any API tokens
- Raw image files (the datasets are accessed via the Hugging Face URLs above; per-dataset 10-row JSONL samples without images are included for reviewer convenience)
