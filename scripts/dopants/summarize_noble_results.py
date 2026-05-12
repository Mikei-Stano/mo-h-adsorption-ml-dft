#!/usr/bin/env python3
"""Summarize noble-dopant results from a GPAW output CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize noble-dopant CSV results.")
    parser.add_argument(
        "--csv",
        default="data/outputs/gpaw_h_adsorption_stage2_refinement_dzp.csv",
        help="Input CSV to summarize (default: %(default)s)",
    )
    parser.add_argument("--elements", default="Au,Ir,Pd,Pt,Ru")
    parser.add_argument("--top-n", type=int, default=15)
    return parser.parse_args()


def pick_delta_column(fieldnames: list[str]) -> str | None:
    for candidate in ("ΔGH_eV", "DGH_eV", "dgh_eV", "descriptor_eV"):
        if candidate in fieldnames:
            return candidate
    return None


def safe_float(value: str | None) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    csv_path = (repo_root / args.csv).resolve()
    elements = [x.strip() for x in args.elements.split(",") if x.strip()]

    if not csv_path.is_file():
        print(f"ERROR: csv not found: {csv_path}")
        return 2

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        delta_col = pick_delta_column(fieldnames)
        if delta_col is None:
            print("ERROR: Could not find delta column in CSV.")
            return 2

        rows = []
        for row in reader:
            surface = (row.get("surface_facet") or "").strip()
            if any(f"_dop{el}" in surface for el in elements):
                rows.append(row)

    total = len(rows)
    completed = [r for r in rows if (r.get("status") or "").strip().lower() == "completed"]
    failed = [r for r in rows if (r.get("status") or "").strip().lower() == "failed"]

    scored = []
    for r in completed:
        dgh = safe_float(r.get(delta_col))
        if dgh is None:
            continue
        scored.append((abs(dgh), dgh, r))
    scored.sort(key=lambda x: x[0])

    print(f"CSV: {csv_path}")
    print(f"Noble filter elements: {', '.join(elements)}")
    print(f"Total noble rows: {total}")
    print(f"Completed: {len(completed)}")
    print(f"Failed: {len(failed)}")

    bins = [(0.2, 0), (0.3, 0), (0.5, 0)]
    for _, dgh, _ in scored:
        ad = abs(dgh)
        bins = [(thr, cnt + (1 if ad <= thr else 0)) for thr, cnt in bins]
    for thr, cnt in bins:
        print(f"|dGH| <= {thr:.1f} eV: {cnt}")

    print("\nTop candidates (closest to zero):")
    print("rank  formula   surface_facet                     site           dGH_eV")
    for i, (_, dgh, r) in enumerate(scored[: max(1, args.top_n)], start=1):
        formula = (r.get("formula") or "")[:8]
        surface = (r.get("surface_facet") or "")[:30]
        site = (r.get("adsorption_site") or "")[:12]
        print(f"{i:>4}  {formula:<8} {surface:<30} {site:<12} {dgh:>8.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
