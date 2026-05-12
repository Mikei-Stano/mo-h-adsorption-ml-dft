#!/usr/bin/env python3
"""Portable GPAW setup-path autodetection utilities."""

from __future__ import annotations

import os
from pathlib import Path


def split_paths(value: str | None) -> list[Path]:
    """Split colon-separated paths and keep existing directories only."""
    if not value:
        return []
    out: list[Path] = []
    for raw in value.split(":"):
        raw = raw.strip()
        if not raw:
            continue
        p = Path(raw).expanduser().resolve()
        if p.is_dir() and p not in out:
            out.append(p)
    return out


def _gpaw_related_dirs() -> list[Path]:
    """Try to infer setup dirs from the installed GPAW package location."""
    dirs: list[Path] = []
    try:
        import gpaw  # type: ignore

        gpaw_dir = Path(gpaw.__file__).resolve().parent
        candidates = [
            gpaw_dir / "setups",
            gpaw_dir / "basis_data",
            gpaw_dir.parent / "gpaw-setups",
            gpaw_dir.parent / "share" / "gpaw-setups",
        ]
        for c in candidates:
            if c.is_dir() and c not in dirs:
                dirs.append(c)
    except Exception:
        pass
    return dirs


def build_candidate_dirs(extra_setup_path: str | None = None) -> list[Path]:
    """Build setup-dir candidates in priority order."""
    candidates: list[Path] = []

    # 1) Explicit override passed from CLI
    candidates.extend(split_paths(extra_setup_path))

    # 2) Current environment
    candidates.extend(split_paths(os.environ.get("GPAW_SETUP_PATH")))

    # 3) Common user/system locations
    common = [
        Path.home() / ".gpaw",
        Path.home() / ".local" / "share" / "gpaw-setups",
        Path.home() / "gpaw-setups",
        Path("/usr/share/gpaw-setups"),
        Path("/usr/local/share/gpaw-setups"),
        Path("/opt/gpaw-setups"),
    ]
    for c in common:
        c = c.expanduser().resolve()
        if c.is_dir() and c not in candidates:
            candidates.append(c)

    # 4) Package-derived hints
    for c in _gpaw_related_dirs():
        if c not in candidates:
            candidates.append(c)

    return candidates


def find_basis(paths: list[Path], element: str, basis: str) -> tuple[Path | None, Path | None]:
    """Return (basis_file, root_dir_used) for element+basis."""
    target = f"{element}.{basis}.basis"
    for base in paths:
        direct = base / target
        if direct.is_file():
            return direct, base
        for hit in base.rglob(target):
            if hit.is_file():
                return hit, base
    return None, None


def autodetect_setup(
    elements: list[str],
    basis: str,
    override: str | None = None,
) -> dict:
    """Detect setup path and verify required basis files are available."""
    paths = build_candidate_dirs(extra_setup_path=override)
    found: dict[str, str] = {}
    roots: list[Path] = []
    missing: list[str] = []

    for el in elements:
        hit, root = find_basis(paths, el, basis)
        if hit is None:
            missing.append(el)
            continue
        found[el] = str(hit)
        if root is not None and root not in roots:
            roots.append(root)

    setup_path = ":".join(str(r) for r in roots) if roots else None
    return {
        "ok": len(missing) == 0,
        "setup_path": setup_path,
        "missing": missing,
        "found": found,
        "searched_paths": [str(p) for p in paths],
    }
