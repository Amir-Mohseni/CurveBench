"""
Test that the Croissant JSON-LD metadata files can be loaded and inspected.

Usage:
    python scripts/test_croissant_loading.py

Requires:
    pip install mlcroissant
"""

import json
import sys
from pathlib import Path


CROISSANT_DIR = Path(__file__).parent.parent / "croissant"
EASY_FILE = CROISSANT_DIR / "curvebench-easy-croissant.json"
HARD_FILE = CROISSANT_DIR / "curvebench-croissant.json"

OUTPUT_FILE = Path(__file__).parent.parent / "croissant_loading_test_output.txt"


def load_and_inspect_json(path: Path) -> dict:
    """Load a JSON file and return parsed content."""
    with open(path) as f:
        return json.load(f)


def inspect_croissant(data: dict, label: str) -> list[str]:
    """Inspect a Croissant document and return a list of result lines."""
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"Croissant file: {label}")
    lines.append(f"{'='*60}")

    lines.append(f"  @type:       {data.get('@type', 'MISSING')}")
    lines.append(f"  name:        {data.get('name', 'MISSING')}")
    lines.append(f"  version:     {data.get('version', 'MISSING')}")
    lines.append(f"  license:     {data.get('license', 'MISSING')}")
    lines.append(f"  url:         {data.get('url', 'MISSING')}")
    lines.append(f"  datePublished: {data.get('datePublished', 'MISSING')}")

    creator = data.get("creator", {})
    lines.append(f"  creator:     {creator.get('name', 'MISSING')}")

    distribution = data.get("distribution", [])
    lines.append(f"\n  Distribution objects: {len(distribution)}")
    for d in distribution:
        lines.append(f"    - {d.get('name', '?')} ({d.get('encodingFormat', '?')})")

    record_sets = data.get("recordSet", [])
    lines.append(f"\n  RecordSets: {len(record_sets)}")
    for rs in record_sets:
        rs_name = rs.get("name", "?")
        fields = rs.get("field", [])
        inline_data = rs.get("data", [])
        lines.append(f"    RecordSet '{rs_name}':")
        lines.append(f"      Fields:      {len(fields)}")
        if inline_data:
            lines.append(f"      Inline rows: {len(inline_data)}")
        for fld in fields:
            lines.append(f"        field '{fld.get('name', '?')}': {fld.get('dataType', '?')}")

    rai = data.get("ml:responsibleAIMetadata", {})
    if rai:
        lines.append(f"\n  RAI metadata keys: {list(rai.keys())}")

    return lines


def verify_metadata_completeness(data: dict, label: str) -> list[str]:
    """Check that NeurIPS 2026 ED Track minimal metadata fields are populated."""
    lines = []
    required_top = ["name", "url", "license", "citeAs", "creator", "conformsTo"]
    required_rai = [
        "rai:dataLimitations", "rai:dataBiases", "rai:personalSensitiveInformation",
        "rai:dataUseCases", "rai:dataSocialImpact", "rai:hasSyntheticData",
        "prov:wasDerivedFrom", "prov:wasGeneratedBy",
    ]
    lines.append(f"\n  Metadata completeness check for {label}:")
    missing_top = [k for k in required_top if not data.get(k) and data.get(k) != False]
    missing_rai = [k for k in required_rai if k not in data]
    if not missing_top and not missing_rai:
        lines.append("    [PASS] All required core + minimal RAI fields are present.")
    else:
        for k in missing_top:
            lines.append(f"    [FAIL] Missing top-level field: {k}")
        for k in missing_rai:
            lines.append(f"    [FAIL] Missing RAI/prov field: {k}")
    return lines


def load_sample_records(label: str) -> list[str]:
    """Load and display sample records from JSONL file."""
    import ast

    lines = []
    base = Path(__file__).parent.parent / "datasets"
    if "easy" in label.lower():
        jsonl_path = base / "curvebench-easy" / "sample_records.jsonl"
    else:
        jsonl_path = base / "curvebench" / "sample_records.jsonl"

    lines.append(f"\n  Sample records from {jsonl_path.name}:")
    try:
        with open(jsonl_path) as f:
            records = [json.loads(line) for line in f if line.strip()]
        sample = records[0]
        lines.append(f"    First sample keys: {list(sample.keys())}")
        lines.append(f"    split:     {sample.get('split', 'N/A')}")
        lines.append(f"    num_nodes: {sample.get('num_nodes', 'N/A')}")
        tree = sample.get("tree", "N/A")
        if isinstance(tree, str):
            tree = ast.literal_eval(tree)
        lines.append(f"    tree:      {tree[:3]}{'...' if len(tree) > 3 else ''}")
        category = sample.get("category", sample.get("level", "N/A"))
        lines.append(f"    category/level: {category}")
        lines.append(f"    [PASS] Sample record loaded successfully.")
    except Exception as e:
        lines.append(f"    [FAIL] Error loading sample records: {e}")
    return lines


def run_mlcroissant_validation(path: Path) -> list[str]:
    """Attempt to run mlcroissant validation (if installed)."""
    lines = []
    try:
        import mlcroissant as mlc
        lines.append(f"\n  mlcroissant validation for {path.name}:")
        try:
            dataset = mlc.Dataset(jsonld=str(path))
            record_sets = list(dataset.metadata.record_sets)
            lines.append(f"    [PASS] mlcroissant loaded dataset successfully.")
            lines.append(f"    Record sets found: {len(record_sets)}")
            for rs in record_sets:
                lines.append(f"      - {rs.name}")
        except Exception as e:
            lines.append(f"    [WARN] mlcroissant raised: {e}")
    except ImportError:
        lines.append(f"\n  mlcroissant not installed — skipping programmatic validation.")
        lines.append(f"  Install with: pip install mlcroissant")
        lines.append(f"  Then validate with: python -m mlcroissant validate {path}")
    return lines


def main():
    all_lines = []
    all_lines.append("CurveBench Croissant Loading Test")
    all_lines.append(f"Run from: {Path(__file__).resolve()}")
    all_lines.append("")

    for path, label in [(EASY_FILE, "CurveBench-Easy"), (HARD_FILE, "CurveBench (Hard)")]:
        if not path.exists():
            all_lines.append(f"[ERROR] File not found: {path}")
            continue

        data = load_and_inspect_json(path)
        all_lines += inspect_croissant(data, label)
        all_lines += verify_metadata_completeness(data, label)
        all_lines += load_sample_records(label)
        all_lines += run_mlcroissant_validation(path)

    all_lines.append("\n" + "="*60)
    all_lines.append("Test complete.")
    all_lines.append("="*60)

    output = "\n".join(all_lines)
    print(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        f.write(output)
    print(f"\nOutput saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
