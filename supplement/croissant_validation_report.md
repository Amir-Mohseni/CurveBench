# Croissant Validation Report

## Overview

Validation of both Croissant JSON-LD metadata files using:

1. **`mlcroissant`** Python API (local) — schema validation and end-to-end record generation.
2. **NeurIPS Croissant Checker** — <https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker>
   (RAI metadata verification is now built into this checker; the formerly-separate
   `croissant-rai-checker` Space has been retired.)

Both files declare conformance to **Croissant 1.1**, include the full set of
NeurIPS 2026 minimal Responsible AI (RAI) fields, include PROV-O activity
provenance via `prov:wasGeneratedBy`, and use the canonical Hugging Face
`FileObject(git+https) → FileSet` distribution pattern that lets validators
stream the underlying parquet files directly from the
`refs/convert/parquet` ref.

---

## File 1: `curvebench-easy-croissant.json`

**File path:** `anonymous_supplement/croissant/curvebench-easy-croissant.json`

| Validator | Check | Result |
|---|---|---|
| `mlcroissant` Python API | schema errors | 0 |
| `mlcroissant` Python API | warnings | 0 |
| `mlcroissant` Python API | end-to-end record generation | PASS — image bytes, num_nodes, tree all materialise correctly |
| NeurIPS croissant-checker | JSON Format Validation | PASS |
| NeurIPS croissant-checker | Croissant Schema Validation | PASS |
| NeurIPS croissant-checker | Responsible AI Metadata | PASS — all required fields present |
| NeurIPS croissant-checker | Records Generation Test | PASS — `easy-examples` and `splits` both validated |

RecordSets: `easy-examples`, `splits` (12 inline rows).
Distribution: 1 `cr:FileObject` (`encodingFormat: git+https`, sha256 placeholder
per [croissant#80](https://github.com/mlcommons/croissant/issues/80)) plus 1
`cr:FileSet` over `default/*/*.parquet` on the auto-generated parquet ref.

---

## File 2: `curvebench-croissant.json`

**File path:** `anonymous_supplement/croissant/curvebench-croissant.json`

| Validator | Check | Result |
|---|---|---|
| `mlcroissant` Python API | schema errors | 0 |
| `mlcroissant` Python API | warnings | 0 |
| `mlcroissant` Python API | end-to-end record generation | PASS — image bytes, category, filename, num_nodes, tree all materialise correctly |
| NeurIPS croissant-checker | JSON Format Validation | PASS |
| NeurIPS croissant-checker | Croissant Schema Validation | PASS |
| NeurIPS croissant-checker | Responsible AI Metadata | PASS — all required fields present |
| NeurIPS croissant-checker | Records Generation Test | PASS — `hard-examples` and `category-splits` both validated |

RecordSets: `hard-examples`, `category-splits` (5 inline rows).
Distribution: 1 `cr:FileObject` (`encodingFormat: git+https`, sha256 placeholder
per [croissant#80](https://github.com/mlcommons/croissant/issues/80)) plus 1
`cr:FileSet` over `default/*/*.parquet` on the auto-generated parquet ref.

---

## Responsible AI fields present (both files)

NeurIPS 2026 ED Track requires the *minimal RAI* set. All eight required fields
are populated, plus three additional Croissant-RAI fields for richer
documentation:

Required:

- `rai:dataLimitations`
- `rai:dataBiases`
- `rai:personalSensitiveInformation`
- `rai:dataUseCases`
- `rai:dataSocialImpact`
- `rai:hasSyntheticData`
- `prov:wasDerivedFrom`
- `prov:wasGeneratedBy`

Additional:

- `rai:dataCollection`
- `rai:dataAnnotationProtocol`
- `rai:dataAnnotationAnalysis`

---

## Hosting status

Both source datasets are now public (no gating, no manual approval) and the
parquet shards on the `refs/convert/parquet` ref are reachable anonymously, so
the NeurIPS validator's *Records Generation Test* can stream them end-to-end:

- <https://huggingface.co/datasets/AmirMohseni/CurveBench-Easy>
- <https://huggingface.co/datasets/AmirMohseni/CurveBench>

---

## Per-file validator output

Saved to:

- `anonymous_supplement/croissant/validation_reports/curvebench-easy-checker.md`
- `anonymous_supplement/croissant/validation_reports/curvebench-hard-checker.md`

---

## Status

All checks pass on both files. No outstanding Croissant-side actions remain.
The validated `curvebench-easy-croissant.json` and `curvebench-croissant.json`
are bundled in `croissant.zip` for the OpenReview multi-dataset upload, and
copies are also shipped inside `supplement.zip` under
`anonymous_supplement/croissant/`.
