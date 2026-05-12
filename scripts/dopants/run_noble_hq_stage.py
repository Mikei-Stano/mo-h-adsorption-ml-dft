#!/usr/bin/env python3
"""Run noble-dopant recalculation in high-quality profile (dzp/PBE)."""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gpaw_setup_auto import autodetect_setup


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run noble dopants with HQ GPAW settings.")
    parser.add_argument("--manifest", default="data/outputs/manifests/noble_dopants_all.txt")
    parser.add_argument("--ranks", type=int, default=16)
    parser.add_argument("--pybin", default=None, help="Python executable (default from env/PYENV)")
    parser.add_argument("--log-dir", default="scripts/dopants/logs")
    parser.add_argument("--checkpoint-every-scf", type=int, default=3)
    parser.add_argument("--setup-path", default=None, help="Optional GPAW setup path override.")
    parser.add_argument("--skip-setup-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--background", action="store_true")
    return parser.parse_args()


def resolve_pybin(repo_root: Path, pybin_arg: str | None) -> Path:
    if pybin_arg:
        return Path(pybin_arg)
    env_pybin = os.environ.get("PYBIN")
    if env_pybin:
        return Path(env_pybin)
    pyenv_root = Path(os.environ.get("PYENV_ROOT", str(Path.home() / ".pyenv")))
    env_name = os.environ.get("ENV_NAME", "cemea-env")
    return pyenv_root / "versions" / env_name / "bin" / "python"


def count_manifest_entries(path: Path) -> int:
    count = 0
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            count += 1
    return count


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    manifest = (repo_root / args.manifest).resolve()
    pybin = resolve_pybin(repo_root, args.pybin)
    log_dir = (repo_root / args.log_dir).resolve()

    if not manifest.is_file():
        print(f"ERROR: manifest not found: {manifest}")
        return 2
    n_structs = count_manifest_entries(manifest)
    if n_structs == 0:
        print(f"ERROR: manifest has no structure entries: {manifest}")
        return 2

    if not pybin.is_file():
        print(f"ERROR: python executable not found: {pybin}")
        return 2

    setup_result = autodetect_setup(
        elements=["Au", "Ir", "Pd", "Pt", "Ru"],
        basis="dzp",
        override=args.setup_path,
    )
    env = os.environ.copy()
    if setup_result["setup_path"]:
        env["GPAW_SETUP_PATH"] = setup_result["setup_path"]
        print(f"Using GPAW_SETUP_PATH={setup_result['setup_path']}")
    else:
        print("GPAW_SETUP_PATH not auto-detected; using GPAW defaults/registered paths.")
        if args.setup_path:
            print(f"Note: provided --setup-path not usable: {args.setup_path}")
        print("Searched paths:")
        for p in setup_result["searched_paths"]:
            print(f"  - {p}")

    if not args.skip_setup_check:
        checker = Path(__file__).with_name("check_gpaw_setups.py")
        check_cmd = [
            str(pybin),
            str(checker),
            "--basis",
            "dzp",
            "--elements",
            "Au,Ir,Pd,Pt,Ru",
        ]
        if setup_result["setup_path"]:
            check_cmd.extend(["--setup-path", setup_result["setup_path"]])
        print("Running setup check:")
        print("  " + " ".join(shlex.quote(x) for x in check_cmd))
        rc = subprocess.run(check_cmd, cwd=repo_root, env=env).returncode
        if rc != 0:
            print("ERROR: setup check failed; refusing to start noble HQ run.")
            return rc

    cmd = [
        "mpiexec", "-n", str(args.ranks), str(pybin), "scripts/gpaw_h_adsorption.py",
        "--stage", "refinement",
        "--mpi",
        "--scalapack",
        "--basis", "dzp",
        "--xc", "PBE",
        "--kpts", "4,4,1",
        "--relax-steps", "20",
        "--fmax", "0.05",
        "--site-search", "rich",
        "--checkpoint-every-scf", str(max(1, args.checkpoint_every_scf)),
        "--max-hours-per-structure", "0",
        "--structure-list", str(manifest),
    ]

    print(f"Noble structures in manifest: {n_structs}")
    print("Command:")
    print("  " + " ".join(shlex.quote(x) for x in cmd))

    if args.dry_run:
        print("Dry run only; command not started.")
        return 0

    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"noble_hq_mpi{args.ranks}_{stamp}.log"

    if args.background:
        with log_path.open("w", encoding="utf-8") as fout:
            proc = subprocess.Popen(cmd, cwd=repo_root, env=env, stdout=fout, stderr=subprocess.STDOUT)
        print(f"Started in background. PID={proc.pid}")
        print(f"Log: {log_path}")
        return 0

    print(f"Streaming to log: {log_path}")
    with log_path.open("w", encoding="utf-8") as fout:
        rc = subprocess.run(cmd, cwd=repo_root, env=env, stdout=fout, stderr=subprocess.STDOUT).returncode
    print(f"Finished with exit code {rc}")
    print(f"Log: {log_path}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
