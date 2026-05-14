# CurveBench — Supplementary Material

> **NeurIPS 2026 Evaluations & Datasets Track — Single-Blind Submission**
>
> This supplement accompanies the CurveBench dataset paper. The dataset is
> hosted publicly on the Hugging Face Hub under the `AmirMohseni` namespace,
> the code is on GitHub at `Amir-Mohseni/CurveBench`, and the metadata is
> documented via the validated Croissant files in `croissant/`. Per the
> NeurIPS 2026 ED Track policy, the manuscript itself does not contain
> author names, but URLs, project pages, and identifying metadata are
> permitted in the supplement.

---

## What is CurveBench?

CurveBench is, to the best of our knowledge, the first dataset explicitly
designed to benchmark the **topological reasoning** capabilities of large
vision-language models (VLMs) by mapping visual containment to exact
combinatorial structures.

A model is asked to infer a global topological structure — specifically, a
**rooted tree** where each node represents a contiguous bounded region and each
edge denotes the boundary curve separating two adjacent regions (parent
contains child).

The full benchmark comprises **756 hand-drawn images** across five categories:

| Category | Count | Difficulty |
|---|---|---|
| Easy | 300 | Foundational (≤5 curves, exhaustive topology) |
| Polygon | 199 | Hard |
| Topographical | 100 | Hard |
| Maze | 100 | Hard |
| Counting | 57 | Hard |

---

## Package Structure

```
supplement/
├── README.md                          ← this file
├── MANIFEST.json                      ← file manifest with sizes and checksums
├── SHA256SUMS.txt                     ← SHA-256 checksums for all files
├── LICENSE-DATASET.txt                ← CC BY 4.0 (images + annotations)
├── LICENSE-CODE.txt                   ← MIT License (all code)
│
├── croissant/
│   ├── curvebench-croissant.json      ← Croissant metadata for CurveBench (Hard)
│   └── curvebench-easy-croissant.json ← Croissant metadata for CurveBench-Easy
│
├── datasets/
│   ├── curvebench/
│   │   ├── README.md                  ← Dataset card (Hard)
│   │   ├── load_dataset.py            ← Dataset loading script
│   │   └── sample_records.jsonl       ← 10 sample records (no images)
│   └── curvebench-easy/
│       ├── README.md                  ← Dataset card (Easy)
│       ├── load_dataset.py            ← Dataset loading script
│       └── sample_records.jsonl       ← 10 sample records (no images)
│
├── code/
│   ├── benchmark_construction/        ← OpenCV pipeline, tree extraction scripts
│   ├── evaluation/                    ← Tree isomorphism evaluation harness
│   └── training/                      ← RLVR training scripts (key files)
│
├── environments/
│   ├── curvebench_easy/               ← Prime Intellect verifier environment (Easy)
│   └── curvebench_hard/               ← Prime Intellect verifier environment (Hard)
│
├── logs/
│   ├── training_curves/               ← Static PNG/CSV exports of RLVR training runs
│   └── evaluation_curves/             ← Static PNG/CSV exports of evaluation sweeps
│
├── scripts/
│   └── test_croissant_loading.py      ← Croissant loading and validation test
│
├── croissant_validation_report.md     ← mlcroissant + NeurIPS validator results
├── croissant_loading_test_output.txt  ← Output from test_croissant_loading.py
├── artifacts_inventory.md             ← Full artifact inventory
└── dataset_statistics.json            ← Verified dataset statistics
```

---

## How to Load the Datasets

### Option 1 — Hugging Face `datasets` library (recommended)

The full datasets are hosted publicly on the Hugging Face Hub:

```python
from datasets import load_dataset

# CurveBench-Easy (foundational split)
ds_easy = load_dataset("AmirMohseni/CurveBench-Easy", split="total_train")

# CurveBench (hard split)
ds_hard = load_dataset("AmirMohseni/CurveBench", split="combined")
```

### Option 2 — Load from local sample records

```python
import json

with open("./datasets/curvebench-easy/sample_records.jsonl") as f:
    samples = [json.loads(line) for line in f]

print(samples[0].keys())
# dict_keys(['num_nodes', 'tree', 'level', 'split'])
```

### Option 3 — Run the loading script

```bash
python datasets/curvebench-easy/load_dataset.py
python datasets/curvebench/load_dataset.py
```

---

## How to Run Evaluation

The evaluation environments are self-contained Python packages. Install
dependencies with:

```bash
pip install -r environments/curvebench_easy/requirements.txt
# or
pip install -r environments/curvebench_hard/requirements.txt
```

Then run:

```bash
python environments/curvebench_easy/curvebench_env.py --split total_test
python environments/curvebench_hard/curvebench_hard_env.py --split combined
```

Scoring uses **tree isomorphism** (via NetworkX):
- `tree_reward` (weight 0.7): 1.0 if predicted tree is structurally isomorphic to ground truth
- `node_count_reward` (weight 0.3): 1.0 if predicted node count matches ground truth

---

## How to Inspect Croissant Metadata

Install mlcroissant:

```bash
pip install mlcroissant
```

Validate:

```bash
python -m mlcroissant validate croissant/curvebench-croissant.json
python -m mlcroissant validate croissant/curvebench-easy-croissant.json
```

Load record sets:

```bash
python scripts/test_croissant_loading.py
```

---

## Licenses

| Component | License |
|---|---|
| Dataset images and annotations | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) — see `LICENSE-DATASET.txt` |
| All code and environments | MIT License — see `LICENSE-CODE.txt` |

---

## Citation

```bibtex
@misc{mohseni2026curvebench,
  title        = {CurveBench: A Benchmark for Topological Reasoning in Vision-Language Models},
  author       = {Mohseni, Amir},
  year         = {2026},
  publisher    = {Hugging Face},
  howpublished = {\url{https://huggingface.co/collections/AmirMohseni/curvebench}}
}
```

---

## Contact

- Author: Amir Mohseni — `amir.mohseni@student.maastrichtuniversity.nl`
- Hugging Face: <https://huggingface.co/AmirMohseni>
- GitHub: <https://github.com/Amir-Mohseni/CurveBench>
