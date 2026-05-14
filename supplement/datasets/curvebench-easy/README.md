---
license: cc-by-4.0
task_categories:
  - visual-question-answering
  - image-to-text
language:
  - en
tags:
  - topology
  - rooted-tree
  - visual-reasoning
  - multimodal
  - benchmark
  - rlvr
pretty_name: CurveBench-Easy
size_categories:
  - n<1K
---

# CurveBench-Easy

**CurveBench-Easy** is the foundational split of CurveBench, a benchmark designed
to evaluate the **topological reasoning** capabilities of large vision-language
models (VLMs). Each sample is a hand-drawn image of disjoint curves paired with
the exact rooted-tree structure that encodes the nestedness relationships visible
in the image.

> **Sister dataset:** the four harder categories (Polygon, Topographical, Maze,
> Counting) are in the companion CurveBench dataset (see
> `../curvebench/README.md`).

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

## What is CurveBench?

CurveBench is, to the best of our knowledge, the first dataset explicitly
designed to benchmark the topological reasoning capabilities of VLMs by mapping
visual containment to exact combinatorial structures. While existing datasets
often evaluate semantic segmentation or geometric object detection, CurveBench
**isolates containment and separation** as the core signals for visual reasoning.

A model is asked to infer a global topological structure — specifically, a
**rooted tree** where:

- each **node** represents a contiguous bounded region, and
- each **edge** denotes the boundary curve that separates two adjacent regions
  (parent contains child).

The full benchmark contains **756 rigorously hand-drawn images** across five
categories:

| Category | Count | Dataset |
|---|---|---|
| Easy | 300 | **this dataset** |
| Polygon | 199 | CurveBench (Hard) |
| Topographical | 100 | CurveBench (Hard) |
| Maze | 100 | CurveBench (Hard) |
| Counting | 57 | CurveBench (Hard) |

---

## Category Description

**Easy (300 images):** This subset establishes a fundamental baseline,
containing spatial configurations with **fewer than six curves**. To ensure
comprehensive coverage of the topological space, all possible rooted-tree
structures with up to six nodes were enumerated. For each unique combinatorial
tree shape, at least two structurally distinct visual representations were
manually authored, guaranteeing that models cannot exploit superficial visual
similarity.

---

## Difficulty Levels

The Easy set is further divided into three difficulty levels based on structural
complexity:

| Level | Description | Train | Validation | Test | Total |
|---|---|---|---|---|---|
| Level 1 | Fewest curves | 84 | 18 | 18 | 120 |
| Level 2 | Moderate | 78 | 16 | 18 | 112 |
| Level 3 | Most complex | 47 | 10 | 11 | 68 |
| **Total** | | **210** | **45** | **45** | **300** |

---

## Dataset Structure

### Splits

| Split name | Size |
|---|---|
| `level_1_train` | 84 |
| `level_1_validation` | 18 |
| `level_1_test` | 18 |
| `level_2_train` | 78 |
| `level_2_validation` | 16 |
| `level_2_test` | 18 |
| `level_3_train` | 47 |
| `level_3_validation` | 10 |
| `level_3_test` | 11 |
| `total_train` | 210 |
| `total_validation` | 45 |
| `total_test` | 45 |

### Data Fields

| Field | Type | Description |
|---|---|---|
| `image` | `Image` | Hand-drawn image of disjoint curves (PNG or JPEG) |
| `num_nodes` | `int64` | Number of nodes in the rooted tree (including the implicit root = outer region) |
| `tree` | `List[List[int]]` | Edge list of the rooted tree; each element is `[parent, child]` (0-indexed; 0 = outermost/root region) |

### Annotation Format

The ground-truth annotation is a **rooted tree** represented as an edge list.

- Root node `0` always represents the outer/background region.
- Each inner region is assigned a positive integer index.
- An edge `[parent, child]` means the child region is directly enclosed within
  the parent region.

```python
# Example: chain topology (outer → region 1 → region 2 → region 3)
{
    "image": <PIL.Image>,
    "num_nodes": 4,
    "tree": [[0, 1], [1, 2], [2, 3]],
}
```

---

## Data Collection and Generation Process

Images were **manually hand-drawn** by the annotators using drawing software.
The construction process ensured:

1. All rooted-tree topologies with up to 6 nodes were enumerated combinatorially.
2. For each topology, at least 2 structurally distinct visual renderings were
   produced.
3. The three difficulty levels correspond to increasing numbers of curves
   (regions) in each image.

Ground-truth rooted trees were produced using an **automated OpenCV
contour-based extraction pipeline** that traces the boundary curves in each
image and assembles parent–child containment relationships into a rooted tree.

---

## Human Verification Process

Every annotation was **individually human-verified** to confirm:

- The extracted tree correctly represents the visible containment relationships.
- No topological errors (missed regions, merged regions, or incorrect hierarchy)
  were present.
- Images with ambiguous or noisy contours were corrected or removed.

---

## Intended Uses

- **Research evaluation** of visual topological reasoning in VLMs.
- **RLVR training** with verifiable binary rewards (tree isomorphism check).
- **Baseline comparison** for nested-region understanding tasks.
- **Ablation studies** on structural complexity (easy → hard).

---

## Out-of-Scope Uses

CurveBench-Easy should **not** be used as the sole basis for deployment in:

- Medical imaging diagnostics
- Surveillance or biometric identification
- Military or autonomous navigation systems
- Legal or financial decision-making

The dataset is intended exclusively for AI research evaluation.

---

## Limitations

- The Easy category evaluates containment topology with ≤5 curves only; it does
  not measure all forms of spatial reasoning, metric geometry, 3D structure, or
  temporal dynamics.
- Hand-authored examples reflect the drawing conventions of the annotators and
  may not represent all visual styles or cultural diagrammatic conventions.
- The dataset does not cover topologies with more than 6 nodes (those are
  covered partially in the Hard split).

---

## Privacy Statement

This dataset **does not contain**:

- Personal data
- Faces or biometric information
- Private text or communications
- Human-subject records

All images are synthetic hand-drawn diagrams.

---

## Bias and Representativeness

- Examples reflect the drawing style and tool conventions of the annotation team.
- The topological coverage is mathematically exhaustive for ≤6 nodes, but
  visual diversity per topology is limited to ≥2 examples.
- Cultural or regional diagrammatic conventions are not represented.

---

## Misuse and Dual-Use Risks

Methods trained on CurveBench could contribute to automated map vectorization
or geospatial analysis pipelines that may have surveillance or military
applications. Researchers should be aware of these potential downstream uses.

---

## License

Dataset (images and annotations): **CC BY 4.0**
See `../../LICENSE-DATASET.txt`

---

## Croissant Metadata

Machine-readable metadata is available at:
`../../croissant/curvebench-easy-croissant.json`
