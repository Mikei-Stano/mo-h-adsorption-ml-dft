#!/usr/bin/env python3
"""Build a manifest of noble-dopant structures (Au, Ir, Pd, Pt, Ru).

By default this scans data/inputs/VASP_inputs and writes a manifest that can be
passed to gpaw_h_adsorption.py via --structure-list.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build manifest for noble dopants.")
    parser.add_argument(
        "--inputs",
        default="data/inputs/VASP_inputs",
        help="Input directory containing structure folders (default: %(default)s)",
    )
    parser.add_argument(
        "--output",
        default="data/outputs/manifests/noble_dopants_all.txt",
        help="Output manifest path (default: %(default)s)",
    )
    parser.add_argument(
        "--elements",
        default="Au,Ir,Pd,Pt,Ru",
        help="Comma-separated dopants to include (default: %(default)s)",
    )
    parser.add_argument(
        "--failed-only-csv",
        default=None,
        help="Optional CSV path; if provided, keep only structures marked failed there.",
    )
    return parser.parse_args()


def load_failed_names(csv_path: Path, target_elements: list[str]) -> set[str]:
    failed = set()
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("status") or "").strip().lower() != "failed":
                continue
            surface = (row.get("surface_facet") or "").strip()
            if any(f"_dop{el}" in surface for el in target_elements):
                formula = (row.get("formula") or "").strip()
                # Directory naming convention used in this repository.
                failed.add(f"{formula}_{surface}")
    return failed


def main() -> int:
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[3]
    inputs = (repo_root / args.inputs).resolve()
    output = (repo_root / args.output).resolve()
    elements = [x.strip() for x in args.elements.split(",") if x.strip()]

    if not inputs.is_dir():
        print(f"ERROR: inputs directory not found: {inputs}")
        return 2

    failed_names = None
    if args.failed_only_csv:
        failed_csv = (repo_root / args.failed_only_csv).resolve()
        if not failed_csv.is_file():
            print(f"ERROR: --failed-only-csv not found: {failed_csv}")
            return 2
        failed_names = load_failed_names(failed_csv, elements)
        print(f"Loaded {len(failed_names)} failed noble entries from: {failed_csv}")

    selected = []
    per_element = Counter()

    for entry in sorted(inputs.iterdir(), key=lambda p: p.name):
        if not entry.is_dir():
            continue
        if not (entry / "POSCAR").is_file():
            continue
        matched = [el for el in elements if f"_dop{el}" in entry.name]
        if not matched:
            continue
        if failed_names is not None and entry.name not in failed_names:
            continue
        selected.append(entry.name)
        for el in matched:
            per_element[el] += 1

    if not selected:
        print("WARNING: no noble-dopant structures matched the filters.")

    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Noble dopant manifest",
        f"# Elements: {', '.join(elements)}",
        f"# Count: {len(selected)}",
        "#",
    ]
    lines.extend(selected)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Wrote manifest: {output}")
    print(f"Total structures: {len(selected)}")
    for el in elements:
        print(f"  {el:>2}: {per_element[el]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
