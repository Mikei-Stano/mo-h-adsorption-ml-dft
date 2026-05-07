"""
Stage 0 screening with MACE-MP-0 universal ML potential.

Per project SOTA roadmap (see SOTA.txt and docs/gpaw_sota_workflow.md), this is
the cheapest funnel: ~10-30s per structure on CPU vs ~5h in GPAW.

Outputs are written into the SAME CSV format as gpaw_h_adsorption.py so the
shortlist passed to Stage 1/2 is fully traceable. Source column is set to
'MACE-MP-0' to make filtering trivial.

Usage:
    python scripts/mace_screening.py                       # all structures
    python scripts/mace_screening.py --machine node1       # MoB family
    python scripts/mace_screening.py --include 'MoB_*'
    python scripts/mace_screening.py --structure-name MoB_edge_B

The script reuses helper functions from gpaw_h_adsorption.py (discover, filter,
H site search, CSV header) so the funnel stays consistent.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.optimize import BFGS

# Reuse the GPAW workflow's helpers so funnel semantics stay identical.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from gpaw_h_adsorption import (  # noqa: E402  (import after sys.path edit)
    CSV_COLUMNS,
    DATA_INPUTS,
    DATA_OUTPUTS,
    MACHINE_SPLITS,
    _candidate_h_positions,
    _prepare_slab,
    _reapply_bottom_fix,
    discover_structures,
    filter_structures_by_name,
)


STAGE0_CSV = DATA_OUTPUTS / "mace_h_adsorption_stage0_screening.csv"
STAGE0_JSON = DATA_OUTPUTS / "mace_h_adsorption_stage0_screening.json"
H2_CACHE = DATA_OUTPUTS / "h2_reference_stage0_mace_mp_0.json"
MACE_OUTPUTS = DATA_OUTPUTS / "mace_calculations"


def _ensure_csv_header(csv_path: Path) -> None:
    csv_path = Path(csv_path)
    if csv_path.exists():
        return
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(CSV_COLUMNS)


def _append_row(row: list, csv_path: Path) -> None:
    csv_path = Path(csv_path)
    with open(csv_path, "a", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            csv.writer(f).writerow(row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_completed_keys(csv_path: Path) -> set:
    completed = set()
    if not csv_path.exists():
        return completed
    try:
        import pandas as pd

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if row.get("status") == "completed":
                completed.add((str(row["formula"]), str(row["surface_facet"])))
    except Exception:
        pass
    return completed


def build_calculator(model: str = "small", device: str | None = None, dtype: str = "float32"):
    """Construct a MACE-MP-0 calculator.

    Args:
        model: 'small' (≈100MB) | 'medium' | 'large'. Small is recommended for CPU.
        device: 'cpu' (default) or 'cuda'.
        dtype: 'float32' (faster on CPU) or 'float64' (matches DFT precision better).
    """
    from mace.calculators import mace_mp

    if device is None:
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"
    print(f"[MACE] Loading model='{model}' device='{device}' dtype='{dtype}'")
    return mace_mp(model=model, device=device, default_dtype=dtype)


def compute_h2_reference(calc, force_recompute: bool = False) -> tuple[float, str]:
    if H2_CACHE.exists() and not force_recompute:
        with open(H2_CACHE) as f:
            return float(json.load(f)["E_h2_eV"]), "cache"
    h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.741]])
    h2.center(vacuum=8)
    h2.calc = calc
    BFGS(h2, logfile=None).run(fmax=0.01, steps=50)
    e = float(h2.get_potential_energy())
    H2_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with open(H2_CACHE, "w") as f:
        json.dump(
            {
                "E_h2_eV": e,
                "model": "mace-mp-0",
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"[MACE] H2 reference: {e:.6f} eV (cached at {H2_CACHE.name})")
    return e, "computed"


def relax_with_mace(atoms: Atoms, calc, fmax: float, steps: int, label: str,
                    fix_indices=None) -> Atoms:
    """Relax with MACE.

    If ``fix_indices`` is provided, those atoms are constrained as a single
    rigid frame; this is the ML-descriptor protocol used for screening so the
    universal potential cannot drift the slab into spurious low-energy
    geometries (only the adsorbate is allowed to settle).
    """
    atoms.set_constraint(None)
    if fix_indices is not None:
        if len(fix_indices) > 0:
            atoms.set_constraint(FixAtoms(indices=list(fix_indices)))
    else:
        _reapply_bottom_fix(atoms)
    atoms.calc = calc
    if steps > 0:
        log_path = MACE_OUTPUTS / f"{label}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        opt = BFGS(atoms, logfile=str(log_path))
        opt.run(fmax=fmax, steps=steps)
    return atoms


def screen_structure(formula, surface, poscar_dir, calc, e_h2, h2_source,
                     fmax: float, steps: int, max_sites: int) -> dict:
    poscar = poscar_dir / "POSCAR"
    out_label = poscar_dir.name

    result = {
        "formula": formula,
        "surface": surface,
        "timestamp": datetime.now().isoformat(),
        "status": "pending",
        "E_clean_slab": None,
        "E_slab_with_h": None,
        "E_h2": float(e_h2),
        "ΔGH": None,
        "adsorption_site": None,
        "h_position": None,
        "n_sites_tried": 0,
        "wall_seconds": None,
        "error": None,
    }
    t0 = time.time()
    try:
        slab = _prepare_slab(str(poscar))
        n_slab = len(slab)
        # Stage 0 ML-descriptor protocol: clean slab is a single-point reference
        # (no relaxation), and for slab+H only H is allowed to relax. This
        # mirrors the standard practice for universal MLIPs (e.g. MACE-MP-0,
        # CHGNet) used as fast HER descriptors and avoids spurious slab
        # reconstructions during BFGS.
        clean = slab.copy()
        clean.calc = calc
        e_clean = float(clean.get_potential_energy())
        result["E_clean_slab"] = e_clean

        # H sites
        candidates = _candidate_h_positions(
            slab, h_distance=1.5, max_sites=max_sites,
            site_search="basic", formula=formula,
        )
        result["n_sites_tried"] = len(candidates)

        best_e = None
        best_site = None
        best_pos = None
        for site_label, h_pos in candidates:
            ah = clean.copy()
            ah += Atoms("H", positions=[h_pos])
            # Fix all original slab atoms; only H (last index) can relax.
            ah = relax_with_mace(
                ah, calc, fmax=fmax, steps=steps,
                label=f"{out_label}_h_{site_label}",
                fix_indices=list(range(n_slab)),
            )
            e = float(ah.get_potential_energy())
            if best_e is None or e < best_e:
                best_e = e
                best_site = site_label
                best_pos = h_pos

        result["E_slab_with_h"] = best_e
        result["adsorption_site"] = best_site
        result["h_position"] = best_pos
        result["ΔGH"] = best_e - e_clean - 0.5 * e_h2
        result["status"] = "completed"
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)
    finally:
        result["wall_seconds"] = round(time.time() - t0, 2)
    return result


def write_csv_row(result: dict, h2_source: str, model_tag: str, fmax: float, steps: int):
    row = [
        result["formula"],
        result["surface"],
        "H",
        result.get("E_clean_slab"),
        result.get("E_slab_with_h"),
        result.get("E_h2"),
        result.get("ΔGH"),
        result.get("ΔGH"),
        f"MACE-MP-0_{model_tag}",
        result.get("adsorption_site"),
        h2_source,
        steps > 0,
        result["status"],
        result["timestamp"],
        # Extra metadata columns (matching gpaw_h_adsorption.py CSV_COLUMNS):
        "stage0",
        f"mace_{model_tag}",
        "MACE-MP-0",
        "1,1,1",
        steps,
        fmax,
        "basic",
        1,
        os.cpu_count() or 1,
        False,
    ]
    _append_row(row, STAGE0_CSV)


def main():
    parser = argparse.ArgumentParser(
        description="Stage 0 screening with MACE-MP-0 universal ML potential.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Machine splits (use --machine instead of --include):\n"
            + "\n".join(f"  {k:14s} → {' , '.join(v)}" for k, v in MACHINE_SPLITS.items())
        ),
    )
    parser.add_argument("--include", default=None,
                        help="Comma-separated glob patterns to filter structures.")
    parser.add_argument("--machine", default=None, choices=list(MACHINE_SPLITS.keys()))
    parser.add_argument("--structure-name", action="append", default=None,
                        help="Run only the given structure directory name (repeatable / comma-separated).")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="Only list matched structures, then exit.")
    parser.add_argument("--model", default="small", choices=["small", "medium", "large"],
                        help="MACE-MP-0 model size (default small, fastest).")
    parser.add_argument("--device", default=None, help="Force device, e.g. 'cpu' or 'cuda'.")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"])
    parser.add_argument("--fmax", type=float, default=0.05,
                        help="BFGS force threshold for H relaxation (eV/A).")
    parser.add_argument("--relax-steps", type=int, default=30,
                        help="Maximum BFGS steps for the H atom (slab is rigid in Stage 0).")
    parser.add_argument("--max-sites", type=int, default=6,
                        help="Maximum candidate H adsorption sites per structure.")
    parser.add_argument("--no-resume", action="store_true",
                        help="Recompute structures already marked completed in the CSV.")
    parser.add_argument("--threads", type=int, default=None,
                        help="OMP/MKL/OPENBLAS thread count (default: all CPUs).")
    args = parser.parse_args()

    threads = args.threads or (os.cpu_count() or 1)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    try:
        import torch
        torch.set_num_threads(threads)
    except Exception:
        pass

    include_patterns = None
    if args.machine:
        include_patterns = MACHINE_SPLITS[args.machine]
    elif args.include:
        include_patterns = [p.strip() for p in args.include.split(",")]

    selected_names = None
    if args.structure_name:
        selected_names = []
        for v in args.structure_name:
            selected_names.extend(p.strip() for p in v.split(",") if p.strip())

    structures = discover_structures(DATA_INPUTS, include_patterns=include_patterns)
    structures = filter_structures_by_name(structures, selected_names)

    print(f"[MACE] Stage 0 screening")
    print(f"[MACE] Output CSV: {STAGE0_CSV}")
    print(f"[MACE] Output JSON: {STAGE0_JSON}")
    print(f"[MACE] H2 cache:   {H2_CACHE}")
    print(f"[MACE] Threads:    {threads}")
    print(f"[MACE] Structures matched: {len(structures)}")

    if args.list_only:
        for f, s, d in structures:
            print(f"  {d.name}")
        return

    if not structures:
        print("[MACE] Nothing to do.")
        return

    _ensure_csv_header(STAGE0_CSV)
    completed = set() if args.no_resume else _load_completed_keys(STAGE0_CSV)
    if completed:
        print(f"[MACE] Resume: skipping {len(completed)} already-completed structures")

    pending = [(f, s, d) for f, s, d in structures if (f, s) not in completed]
    if not pending:
        print("[MACE] All matched structures already completed.")
        return
    print(f"[MACE] Pending: {len(pending)}")

    calc = build_calculator(model=args.model, device=args.device, dtype=args.dtype)
    e_h2, h2_source = compute_h2_reference(calc)

    all_results = []
    t_start = time.time()
    for i, (formula, surface, poscar_dir) in enumerate(pending, 1):
        eta = ""
        if i > 1:
            avg = (time.time() - t_start) / (i - 1)
            eta = f" | ETA {(avg * (len(pending) - i + 1)) / 60:.1f} min"
        print(f"\n[{i}/{len(pending)}] {poscar_dir.name}{eta}")
        try:
            result = screen_structure(
                formula, surface, poscar_dir, calc, e_h2, h2_source,
                fmax=args.fmax, steps=args.relax_steps, max_sites=args.max_sites,
            )
        except KeyboardInterrupt:
            print("\n[MACE] Interrupted; CSV is up to date.")
            break
        all_results.append(result)
        write_csv_row(result, h2_source, args.model, args.fmax, args.relax_steps)
        if result["status"] == "completed":
            print(f"  ΔGH = {result['ΔGH']:+.4f} eV  (site={result['adsorption_site']}, "
                  f"{result['wall_seconds']}s)")
        else:
            print(f"  FAILED: {result['error']}")

    # JSON summary
    STAGE0_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(STAGE0_JSON, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Quick ranking
    completed_results = [r for r in all_results if r["status"] == "completed"]
    if completed_results:
        completed_results.sort(key=lambda r: abs(r["ΔGH"]))
        print("\n[MACE] Top 15 by |ΔGH| (Stage 0 ranking):")
        print(f"  {'rank':>4}  {'|ΔGH| eV':>10}  {'ΔGH eV':>10}  structure")
        for rank, r in enumerate(completed_results[:15], 1):
            print(f"  {rank:>4}  {abs(r['ΔGH']):>10.4f}  {r['ΔGH']:>+10.4f}  "
                  f"{r['formula']}_{r['surface']}  ({r['adsorption_site']})")

    elapsed = (time.time() - t_start) / 60.0
    print(f"\n[MACE] Done. Total wall time: {elapsed:.1f} min  "
          f"({elapsed * 60 / max(1, len(pending)):.1f} s/structure)")
    print(f"[MACE] CSV:  {STAGE0_CSV}")
    print(f"[MACE] JSON: {STAGE0_JSON}")


if __name__ == "__main__":
    main()
