# Pre-Submission Review Checklist — CurveBench NeurIPS 2026 ED Track

Final review checklist before uploading to OpenReview. The supplement folder
is `anonymous_supplement/` and the bundle to upload is `supplement.zip`
(one level up from this folder).

---

## What's in this package

```
anonymous_supplement/
│
├── README.md                            ← reviewer instructions
├── LICENSE-DATASET.txt                  ← CC BY 4.0 (Amir Mohseni)
├── LICENSE-CODE.txt                     ← MIT (Amir Mohseni)
│
├── croissant/
│   ├── curvebench-easy-croissant.json   ← Croissant 1.1 + minimal RAI
│   ├── curvebench-croissant.json        ← Croissant 1.1 + minimal RAI
│   └── validation_reports/
│       ├── curvebench-easy-checker.md   ← NeurIPS validator output
│       └── curvebench-hard-checker.md   ← NeurIPS validator output
│
├── datasets/
│   ├── curvebench-easy/                 ← dataset card + load script + 10-row JSONL
│   └── curvebench/                      ← dataset card + load script + 10-row JSONL
│
├── code/
│   ├── benchmark_construction/          ← OpenCV contour pipeline + push scripts
│   ├── evaluation/                      ← tree_isomorphism.py
│   └── training/
│       ├── grpo_region_tree.py          ← main GRPO training script
│       ├── reward_funcs/region_tree_reward.py
│       └── examples/cli_configs/        ← 3 model configs (a, b, c)
│
├── environments/
│   ├── curvebench_easy/                 ← Prime Intellect eval env (Easy)
│   └── curvebench_hard/                 ← Prime Intellect eval env (Hard)
│
├── logs/
│   ├── Curvebench Training Results.zip  ← W&B LaTeX export (compile with pdflatex)
│   ├── training_curves/README.md
│   └── evaluation_curves/README.md
│
├── scripts/
│   └── test_croissant_loading.py        ← validates both Croissant files
│
├── croissant_validation_report.md       ← all checks pass (records-gen depends on dataset being public)
├── croissant_loading_test_output.txt    ← test script output
├── artifacts_inventory.md
├── dataset_statistics.json
├── MANIFEST.json
└── SHA256SUMS.txt
```

## Model key (verify these are correct)

| Label | Architecture | Reward |
|---|---|---|
| model-a | Qwen3-VL-8B | Tree isomorphism only |
| model-b | Qwen3-VL-8B | Tree (0.7) + node count (0.3) |
| model-c | Gemma 3 12B | Tree (0.7) + node count (0.3) |

## Pre-submission checks

- [ ] Dataset cards (`datasets/curvebench*/README.md`) — descriptions accurate?
- [ ] Dataset statistics (`dataset_statistics.json`) — counts correct? (Easy: 300, Hard: 456, Total: 756)
- [ ] Model labels above — architectures and reward configs correct?
- [ ] Training configs (`code/training/examples/cli_configs/`) — hyperparameters match what was actually run?
- [ ] W&B export (`logs/Curvebench Training Results.zip`) — compile and check all panels
- [ ] Croissant files — validated by NeurIPS checker (`croissant_validation_report.md`)
- [ ] Hugging Face datasets `AmirMohseni/CurveBench-Easy` and `AmirMohseni/CurveBench` are PUBLIC (not gated). Without this the NeurIPS validator's records-generation test will fail with 401.
- [ ] GitHub repo `Amir-Mohseni/CurveBench` is public and README is complete
- [ ] Hugging Face collection `AmirMohseni/curvebench` is public and links both datasets
- [ ] Manuscript PDF uses the default LaTeX option `\usepackage[eandd]{neurips_2026}` and contains NO author names (single-blind still requires this)
- [ ] OpenReview submission form: single-blind option selected with justification text
- [ ] License choice: dataset = CC BY 4.0, code = MIT — agreed?

## Compile the W&B report

```bash
cd /tmp && unzip "anonymous_supplement/logs/Curvebench Training Results.zip" -d wb_report && cd wb_report && pdflatex report.tex
```

Or upload the ZIP directly to [Overleaf](https://www.overleaf.com/) → New Project → Upload Project.

## Validate Croissant files

```bash
pip install mlcroissant
python anonymous_supplement/scripts/test_croissant_loading.py
```

Expected output: both `CurveBench-Easy` and `CurveBench (Hard)` → PASS, all required core + RAI fields present, mlcroissant validation OK.
