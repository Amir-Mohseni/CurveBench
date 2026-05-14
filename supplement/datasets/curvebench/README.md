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
pretty_name: CurveBench
size_categories:
  - n<1K
---

# CurveBench

**CurveBench** (also referred to as **CurveBench-Hard**) is the hard evaluation
split of the CurveBench benchmark, designed to evaluate the **topological
reasoning** capabilities of large vision-language models (VLMs). Each sample is
a hand-drawn or procedurally inspired image of disjoint curves paired with the
exact rooted-tree structure encoding the nestedness relationships in the image.

> **Sister dataset:** the foundational Easy category (fewer than 6 curves,
> exhaustive over all rooted trees with up to 6 nodes) is in the companion
> CurveBench-Easy dataset (see `../curvebench-easy/README.md`).

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
| Easy | 300 | CurveBench-Easy |
| Polygon | 199 | **this dataset** |
| Topographical | 100 | **this dataset** |
| Maze | 100 | **this dataset** |
| Counting | 57 | **this dataset** |

---

## Categories

### Polygon (199 images)

Following a systematic construction methodology identical to the Easy category,
this subset restricts the geometries entirely to **non-intersecting polygons**.
This tests a model's robustness to sharp angles and piecewise-linear boundaries
compared to smooth, continuous Jordan curves.

### Topographical (100 images)

Grounded in applied distributions, these images are directly inspired by
**real-world topographic maps**. They mimic the natural behaviour of elevation
level sets, bridging the gap between theoretical combinatorial benchmarks and
practical visual understanding domains.

### Maze (100 images)

Designed to stress-test **long-range spatial reasoning**, this category features
highly convoluted, labyrinthine curves with deep nesting. The spatial
entanglement makes distinguishing the interior from the exterior of a boundary
visually demanding, forcing models to track complex geometric boundaries over
long distances.

### Counting (57 images)

This densely populated subset evaluates a model's **scalability and capacity
limits**. Focused primarily on the volume of nested entities, these images are
packed with a high number of disjoint curves, challenging the model to construct
massive corresponding rooted trees without accumulating structural errors.

---

## Dataset Structure

### Splits

| Split name | Size | Description |
|---|---|---|
| `polygon` | 199 | Piecewise-linear polygon boundaries |
| `topographical` | 100 | Topographic-map-inspired curves |
| `maze` | 100 | Labyrinthine, deeply nested curves |
| `counting` | 57 | High-density curve configurations |
| `combined` | 456 | All four categories merged |

### Data Fields

| Field | Type | Description |
|---|---|---|
| `image` | `Image` | Hand-drawn image of disjoint curves (PNG) |
| `category` | `string` | One of `"Counting"`, `"Maze"`, `"Polygon"`, `"Topographical"` |
| `filename` | `string` | Original filename of the image (e.g. `"1.PNG"`) |
| `num_nodes` | `int32` | Number of nodes in the rooted tree (including the implicit root = outer region) |
| `tree` | `string` | Stringified edge list, e.g. `"[(0, 1), (0, 2), (1, 3)]"` — each tuple is `(parent, child)`; 0 = outermost/root region |

### Annotation Format

The ground-truth annotation is a **rooted tree** encoded as a Python-stringified
list of `(parent, child)` integer tuples.

```python
import ast

sample = {
    "image": <PIL.Image>,
    "category": "Counting",
    "filename": "1.PNG",
    "num_nodes": 26,
    "tree": "[(0, 1), (0, 6), (1, 2), (2, 3), ...]",
}

# Parse back to list of tuples:
edges = ast.literal_eval(sample["tree"])
```

---

## Data Collection and Generation Process

- **Polygon** images were manually authored using piecewise-linear drawing.
- **Topographical** images were inspired by real-world topographic maps and
  hand-drawn to mimic elevation level-set patterns.
- **Maze** images were manually constructed to maximize spatial entanglement.
- **Counting** images were packed with high volumes of nested disjoint curves.

Ground-truth rooted trees were produced using an **automated OpenCV
contour-based extraction pipeline** that traces boundary curves and assembles
parent–child containment relationships into a rooted tree.

---

## Human Verification Process

Every annotation was **individually human-verified** to confirm:

- The extracted tree correctly represents the visible containment relationships.
- No topological errors (missed regions, merged regions, or incorrect hierarchy)
  were present.
- Images with ambiguous or noisy contours were corrected or removed.

---

## Evaluation

Predicted trees are compared to the ground truth using **tree isomorphism**
(via NetworkX): a prediction receives full credit only if the predicted edge set
is structurally identical to the ground-truth tree up to node relabelling. This
provides a deterministic, binary evaluation metric.

Reward components used in RLVR training:

| Reward | Weight | Description |
|---|---|---|
| `tree_reward` | 0.7 | 1.0 if predicted tree is isomorphic to ground truth |
| `node_count_reward` | 0.3 | 1.0 if predicted node count matches ground truth |

---

## Intended Uses

- **Research evaluation** of visual topological reasoning in VLMs.
- **RLVR training** with verifiable binary rewards.
- **Stress-testing** large VLMs on structurally complex curve images.
- **Ablation studies** by category (polygon, topographical, maze, counting).

---

## Out-of-Scope Uses

CurveBench should **not** be used as the sole basis for deployment in:

- Medical imaging diagnostics
- Surveillance or biometric identification
- Military or autonomous navigation systems
- Legal or financial decision-making

---

## Limitations

- Evaluates rooted region-containment topology only.
- Does not measure metric geometry, 3D structure, temporal dynamics, or semantic
  map understanding.
- Counting category (57 examples) is smaller than the other categories.
- Hand-authored images reflect the visual conventions of the annotation team.

---

## Privacy Statement

This dataset **does not contain**:

- Personal data
- Faces or biometric information
- Private text or communications
- Human-subject records

All images are synthetic hand-drawn or procedurally inspired diagrams.

---

## Bias and Representativeness

- Examples reflect the drawing style and conventions of the annotation team.
- Topographical images are inspired by real-world maps but are not actual
  cartographic data.
- Cultural or regional diagrammatic conventions are not systematically
  represented.

---

## Misuse and Dual-Use Risks

Methods trained on CurveBench could contribute to automated map vectorization
or geospatial analysis pipelines with possible surveillance or military
applications. Researchers should be aware of these potential downstream uses.

---

## License

Dataset (images and annotations): **CC BY 4.0**
See `../../LICENSE-DATASET.txt`

---

## Croissant Metadata

Machine-readable metadata is available at:
`../../croissant/curvebench-croissant.json`
