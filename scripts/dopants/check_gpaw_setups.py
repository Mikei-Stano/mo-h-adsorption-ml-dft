#!/usr/bin/env python3
"""Validate GPAW setup/basis availability for selected elements."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpaw_setup_auto import autodetect_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check GPAW setup path and noble basis files.")
    parser.add_argument(
        "--basis",
        default="dzp",
        help="Basis name to validate (default: %(default)s)",
    )
    parser.add_argument(
        "--elements",
        default="Au,Ir,Pd,Pt,Ru",
        help="Comma-separated element symbols (default: %(default)s)",
    )
    parser.add_argument(
        "--setup-path",
        default=None,
        help="Override setup path instead of GPAW_SETUP_PATH (colon-separated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    elements = [x.strip() for x in args.elements.split(",") if x.strip()]

    result = autodetect_setup(elements, args.basis, override=args.setup_path)

    print("Checking setup paths:")
    for p in result["searched_paths"]:
        print(f"  - {p}")

    if result["setup_path"]:
        os.environ["GPAW_SETUP_PATH"] = result["setup_path"]
        print(f"\nDetected GPAW_SETUP_PATH: {result['setup_path']}")
    else:
        print("\nDetected GPAW_SETUP_PATH: <none> (using GPAW defaults/registered paths)")

    try:
        from ase import Atoms
        from gpaw import GPAW
    except ModuleNotFoundError as exc:
        print("\nFAIL: required module missing in current Python environment:", exc)
        print("Tip: run checker via the same interpreter used by GPAW job, e.g.")
        print("  /home/mikei/.pyenv/versions/cemea-env/bin/python scripts/dopants/check_gpaw_setups.py --basis dzp --elements Au,Ir,Pd,Pt,Ru")
        return 2

    def runtime_check(element: str, basis: str) -> tuple[bool, str]:
        try:
            # Keep the atom centered to avoid boundary artefacts during setup validation.
            atoms = Atoms(element, positions=[[4.0, 4.0, 4.0]], cell=[8.0, 8.0, 8.0], pbc=False)
            calc = GPAW(mode="lcao", basis=basis, xc="PBE", kpts=(1, 1, 1), txt=None)
            atoms.calc = calc
            _ = atoms.get_potential_energy()
            return True, "runtime OK"
        except Exception as exc:
            msg = str(exc)
            # Missing basis/setup is fatal.
            if "Could not find required basis set file" in msg or "GPAW_SETUP_PATH" in msg:
                return False, msg
            # SCF convergence is not a setup-path problem; treat as setup usable.
            if "Did not converge" in msg:
                return True, "basis loaded (SCF did not converge in quick test)"
            return True, f"basis likely loaded (non-setup exception: {msg})"

    failed: list[str] = []
    for el in elements:
        ok, msg = runtime_check(el, args.basis)
        if ok:
            print(f"OK: {el} ({args.basis}) -> {msg}")
        else:
            print(f"MISSING/UNUSABLE: {el} ({args.basis}) -> {msg}")
            failed.append(el)

    if failed:
        print("\nFAIL: basis unusable for: " + ", ".join(failed))
        return 1

    print("\nPASS: basis runtime check passed for all requested elements.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
