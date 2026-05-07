"""
GPAW-based H Adsorption Energy Calculator (SOTA two-stage funnel)

Stage 1 (screening):  fast LCAO/sz scan for ranking candidates.
Stage 2 (refinement): dzp/PBE rerun with rich H-site search for shortlist.

Robustness:
- GPAW .gpw checkpoint + ASE BFGS trajectory restart (per-structure resume)
- Real GPAW MPI parallelism (mpiexec -n N python ...) with ScaLAPACK/ELPA
- MPI-safe I/O (rank 0 writes CSV/JSON/H2 cache; barriers where required)
- Profile-specific H2 reference cache (never shared across basis/xc/mode/stage)
- Per-structure timeout retained as opt-in calibration/watchdog only
- Backwards compatible: ProcessPoolExecutor path preserved when MPI is off
"""

import os
import csv
import json
import signal
import fcntl
import argparse
import fnmatch
import time
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from ase import Atoms
from ase.io import read
from gpaw import GPAW
from ase.optimize import BFGS
from ase.constraints import FixAtoms
from datetime import datetime


# ── MPI helpers (safe fallback when ASE/GPAW MPI not available) ──
try:
    from ase.parallel import world as _ase_world, paropen as _paropen, parprint as _parprint  # noqa: F401

    def _mpi_rank():
        return int(_ase_world.rank)

    def _mpi_size():
        return int(_ase_world.size)

    def _barrier():
        try:
            _ase_world.barrier()
        except Exception:
            pass
except Exception:  # pragma: no cover - fallback for non-MPI envs
    _ase_world = None

    def _mpi_rank():
        return 0

    def _mpi_size():
        return 1

    def _barrier():
        return None


def _is_rank0():
    return _mpi_rank() == 0


def _parprint(*args, **kwargs):
    if _is_rank0():
        print(*args, **kwargs)


def is_mpi_run():
    """True when running under real GPAW/ASE MPI (world.size > 1)."""
    return _mpi_size() > 1


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"
DATA_OUTPUTS = REPO_ROOT / "data" / "outputs"
GPAW_OUTPUTS = DATA_OUTPUTS / "gpaw_calculations"

# Default (legacy) configuration; mutated at runtime via stage profiles + CLI overrides.
GPAW_CONFIG = {
    'mode': 'lcao',
    'basis': 'dzp',
    'xc': 'LDA',
    'kpts': (4, 4, 1),
    'h': 0.20,
    'txt': 'gpaw.txt',
    'convergence': {
        'energy': 1e-5,
        'density': 1e-4,
        'eigenstates': 1e-5,
    },
    # Pulay mixer tuned for metals (low beta, high weight).
    'mixer': {'beta': 0.05, 'nmaxold': 5, 'weight': 100.0},
    'maxiter': 333,
}

RELAXATION_CONFIG = {
    'fmax': 0.10,
    'steps': 8,
}

USE_RELAXATION = True

# Parallelism: cores allocated per GPAW calculation (auto worker detection)
CORES_PER_CALC = 11
# RAM per calculation (GB) for auto-detection
RAM_PER_CALC_GB = 4
# Per-structure hard timeout (hours). 0 disables timeout.
MAX_HOURS_PER_STRUCTURE = 0.0


# ── Stage profiles (SOTA two-stage funnel) ───────────────────────
# Screening uses aggressive metal-friendly defaults: low-beta Pulay mixer with
# high "weight" for metals, looser SCF convergence (ranking-only), and a coarser
# real-space grid. These are the actual levers in this GPAW build (no MPI/native
# BLAS available — the bottleneck is grid density/potential matrix ops).
STAGE_PROFILES = {
    'screening': {
        'basis': 'sz',
        'xc': 'LDA',
        'kpts': (2, 2, 1),
        'h': 0.25,
        'relax_steps': 3,
        'fmax': 0.15,
        'site_search': 'basic',
        'mixer': {'beta': 0.05, 'nmaxold': 5, 'weight': 100.0},
        'convergence': {'energy': 5e-4, 'density': 1e-3, 'eigenstates': 5e-4},
        'maxiter': 200,
        'output_csv_name': 'gpaw_h_adsorption_stage1_screening_sz.csv',
        'output_json_name': 'gpaw_h_adsorption_stage1_screening_sz.json',
    },
    'refinement': {
        'basis': 'dzp',
        'xc': 'PBE',
        'kpts': (4, 4, 1),
        'h': 0.20,
        'relax_steps': 20,
        'fmax': 0.05,
        'site_search': 'rich',
        'mixer': {'beta': 0.05, 'nmaxold': 5, 'weight': 100.0},
        'convergence': {'energy': 1e-5, 'density': 1e-4, 'eigenstates': 1e-5},
        'maxiter': 333,
        'output_csv_name': 'gpaw_h_adsorption_stage2_refinement_dzp.csv',
        'output_json_name': 'gpaw_h_adsorption_stage2_refinement_dzp.json',
    },
}


# Module-level runtime config; populated by main() after CLI parsing.
RUN_CONFIG = {
    'stage': 'screening',
    'output_csv': None,
    'output_json': None,
    'h2_reference_file': None,
    'site_search': 'basic',
    'mpi_enabled': False,
    'mpi_world_size': 1,
    'mpi_rank': 0,
    'omp_threads': 1,
    'checkpoint_enabled': True,
    'checkpoint_every_scf': 5,
    'scalapack': False,
    'elpa': False,
    'max_hours_per_structure': 0.0,
}


# CSV header (preserves legacy columns, appends SOTA metadata columns)
LEGACY_CSV_COLUMNS = [
    'formula', 'surface_facet', 'adsorbate',
    'E_clean_slab_eV', 'E_slab_with_h_eV', 'E_h2_eV',
    'ΔGH_eV', 'descriptor_eV', 'source',
    'adsorption_site', 'h2_source', 'relaxed',
    'status', 'timestamp',
]
EXTRA_CSV_COLUMNS = [
    'stage', 'basis', 'xc', 'kpts', 'relax_steps', 'fmax',
    'site_search', 'mpi_ranks', 'omp_threads', 'restart_used',
]
CSV_COLUMNS = LEGACY_CSV_COLUMNS + EXTRA_CSV_COLUMNS

# ── Graceful shutdown ────────────────────────────────────────────
_shutdown_requested = False


def _signal_handler(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    print("\n⚠️  Shutdown requested — finishing current calculations, then exiting...")


signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ── Auto-detect parallelism ─────────────────────────────────────
def _detect_max_workers(override_workers=None):
    """Determine how many parallel GPAW calculations to run."""
    if override_workers is not None:
        print(f"Using manual worker override: {override_workers}")
        return max(1, int(override_workers))

    cpu_count = os.cpu_count() or 4
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    ram_gb = int(line.split()[1]) / (1024 * 1024)
                    break
            else:
                ram_gb = 16
    except OSError:
        ram_gb = 16

    by_cpu = max(1, cpu_count // CORES_PER_CALC)
    by_ram = max(1, int(ram_gb // RAM_PER_CALC_GB))
    workers = min(by_cpu, by_ram)
    print(f"Auto-detected: {cpu_count} CPUs, {ram_gb:.0f} GB RAM → {workers} parallel workers")
    return workers


def _set_thread_env(threads_per_calc):
    """Set thread environment for BLAS/OpenMP libraries."""
    t = max(1, int(threads_per_calc))
    os.environ['OMP_NUM_THREADS'] = str(t)
    os.environ['OPENBLAS_NUM_THREADS'] = str(t)
    os.environ['MKL_NUM_THREADS'] = str(t)
    os.environ['NUMEXPR_NUM_THREADS'] = str(t)
    RUN_CONFIG['omp_threads'] = t
    _parprint(f"Thread env: OMP_NUM_THREADS={t}, OPENBLAS_NUM_THREADS={t}, MKL_NUM_THREADS={t}")


# ── Runtime metadata helpers (for CSV rows) ──────────────────────
def _kpts_str():
    k = GPAW_CONFIG.get('kpts', (1, 1, 1))
    return f"{k[0]},{k[1]},{k[2]}"


def _result_metadata(restart_used=False):
    return {
        'stage': RUN_CONFIG.get('stage'),
        'basis': GPAW_CONFIG.get('basis'),
        'xc': GPAW_CONFIG.get('xc'),
        'kpts': _kpts_str(),
        'relax_steps': RELAXATION_CONFIG.get('steps'),
        'fmax': RELAXATION_CONFIG.get('fmax'),
        'site_search': RUN_CONFIG.get('site_search'),
        'mpi_ranks': RUN_CONFIG.get('mpi_world_size', 1),
        'omp_threads': RUN_CONFIG.get('omp_threads', 1),
        'restart_used': bool(restart_used),
    }


# ── Incremental CSV (MPI-safe: rank 0 only) ──────────────────────
def _ensure_csv_header(csv_path):
    """Create CSV with header if it doesn't exist (rank 0 only)."""
    csv_path = Path(csv_path)
    if _is_rank0() and not csv_path.exists():
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(CSV_COLUMNS)
    _barrier()


def _append_result_csv(result_row, csv_path):
    """Append a single result row to CSV with file locking (rank 0 only)."""
    if not _is_rank0():
        return
    csv_path = Path(csv_path)
    with open(csv_path, 'a', newline='') as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            writer = csv.writer(f)
            writer.writerow(result_row)
            f.flush()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _load_completed_keys(csv_path):
    """Load set of (formula, surface_facet) already completed in CSV."""
    csv_path = Path(csv_path)
    completed = set()
    if not csv_path.exists():
        return completed
    try:
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            if row.get('status') == 'completed':
                completed.add((str(row['formula']), str(row['surface_facet'])))
    except Exception:
        pass
    return completed


def discover_structures(base_dir, include_patterns=None):
    """Discover all POSCAR files under base_dir and return labels.

    Args:
        base_dir: directory containing structure subdirectories
        include_patterns: optional list of glob patterns (e.g. ["Ni_Mo2C_*", "Mo2N_*"]).
            If provided, only directories matching at least one pattern are included.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    items = []
    for entry in sorted(base_path.iterdir()):
        if not entry.is_dir():
            continue
        poscar = entry / "POSCAR"
        if not poscar.exists():
            continue
        # Apply include filter
        if include_patterns:
            if not any(fnmatch.fnmatch(entry.name, pat) for pat in include_patterns):
                continue
        parts = entry.name.split("_", 1)
        formula = parts[0]
        surface = parts[1] if len(parts) > 1 else "unknown"
        items.append((formula, surface, entry))
    return items


def filter_structures_by_name(structures, selected_names):
    """Keep only structure directories whose names were explicitly requested."""
    if not selected_names:
        return structures

    requested = [name.strip() for name in selected_names if name and name.strip()]
    if not requested:
        return structures

    requested_set = set(requested)
    filtered = [item for item in structures if item[2].name in requested_set]
    missing = [name for name in requested if name not in {item[2].name for item in filtered}]
    if missing:
        raise ValueError(f"Requested structures not found: {', '.join(missing)}")
    return filtered


def _build_parallel_dict():
    """Build GPAW parallel dict honoring ScaLAPACK / ELPA flags."""
    parallel = {}
    if RUN_CONFIG.get('scalapack'):
        parallel['sl_auto'] = True
    if RUN_CONFIG.get('elpa'):
        parallel['use_elpa'] = True
    return parallel


def _gpw_meta_for_label(label):
    return Path(str(label) + '.gpw.meta.json')


def _gpw_path_for_label(label):
    return Path(str(label) + '.gpw')


def _write_gpw_meta(label):
    """Sidecar metadata describing how a .gpw was produced."""
    if not _is_rank0():
        return
    meta = {
        'mode': GPAW_CONFIG.get('mode'),
        'basis': GPAW_CONFIG.get('basis'),
        'xc': GPAW_CONFIG.get('xc'),
        'kpts': list(GPAW_CONFIG.get('kpts', (1, 1, 1))),
        'stage': RUN_CONFIG.get('stage'),
        'relax_steps': RELAXATION_CONFIG.get('steps'),
        'fmax': RELAXATION_CONFIG.get('fmax'),
        'site_search': RUN_CONFIG.get('site_search'),
        'timestamp': datetime.now().isoformat(),
    }
    meta_path = _gpw_meta_for_label(label)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)


def _meta_matches_current(label):
    """Return True if existing sidecar metadata matches current run profile."""
    meta_path = _gpw_meta_for_label(label)
    if not meta_path.exists():
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except Exception:
        return False
    if meta.get('mode') != GPAW_CONFIG.get('mode'):
        return False
    if meta.get('basis') != GPAW_CONFIG.get('basis'):
        return False
    if meta.get('xc') != GPAW_CONFIG.get('xc'):
        return False
    if list(meta.get('kpts', [])) != list(GPAW_CONFIG.get('kpts', (1, 1, 1))):
        return False
    return True


def setup_gpaw_calculator(label='gpaw', gpw_restart=None, checkpoint=None,
                          checkpoint_every_scf=None, h2_kpts=None):
    """Create a GPAW calculator, optionally resuming from a .gpw checkpoint.

    Args:
        label: filename stem used for txt log + .gpw checkpoint.
        gpw_restart: explicit .gpw path to restart from (overrides label-based discovery).
        checkpoint: enable periodic .gpw write; defaults to RUN_CONFIG['checkpoint_enabled'].
        checkpoint_every_scf: SCF cadence for periodic write.
        h2_kpts: special override for isolated H2 reference (uses (1,1,1)).

    Returns:
        (calc, restart_used)
    """
    if checkpoint is None:
        checkpoint = bool(RUN_CONFIG.get('checkpoint_enabled', True))
    if checkpoint_every_scf is None:
        checkpoint_every_scf = int(RUN_CONFIG.get('checkpoint_every_scf', 5))

    label = str(label)
    txt_path = label + '.txt'
    gpw_path = _gpw_path_for_label(label)
    candidate_restart = gpw_restart or (gpw_path if gpw_path.exists() else None)

    restart_used = False
    if candidate_restart and Path(candidate_restart).exists() and _meta_matches_current(label):
        try:
            calc = GPAW(str(candidate_restart), txt=txt_path)
            restart_used = True
            _parprint(f"     ↺ Resuming from checkpoint: {candidate_restart}")
        except Exception as exc:
            _parprint(f"     ⚠️  Failed to load checkpoint {candidate_restart} ({exc}); starting fresh")
            calc = None
    else:
        calc = None

    if calc is None:
        kpts = (1, 1, 1) if h2_kpts else GPAW_CONFIG['kpts']
        from gpaw import Mixer, PoissonSolver  # local import to avoid hard dep at module load
        mixer_cfg = GPAW_CONFIG.get('mixer') or {}
        gpaw_kwargs = dict(
            mode=GPAW_CONFIG['mode'],
            basis=GPAW_CONFIG['basis'],
            xc=GPAW_CONFIG['xc'],
            kpts=kpts,
            h=GPAW_CONFIG.get('h', 0.20),
            txt=txt_path,
            convergence=GPAW_CONFIG['convergence'],
            maxiter=int(GPAW_CONFIG.get('maxiter', 333)),
            # Disable symmetry so relaxation steps that displace atoms don't
            # trigger "Broken symmetry!" on subsequent SCF calls.
            symmetry='off',
        )
        if mixer_cfg:
            try:
                gpaw_kwargs['mixer'] = Mixer(
                    beta=float(mixer_cfg.get('beta', 0.05)),
                    nmaxold=int(mixer_cfg.get('nmaxold', 5)),
                    weight=float(mixer_cfg.get('weight', 50.0)),
                )
            except Exception as exc:
                _parprint(f"     ⚠️  Could not build custom Mixer ({exc}); using GPAW default")
        parallel = _build_parallel_dict()
        if parallel:
            gpaw_kwargs['parallel'] = parallel
        try:
            calc = GPAW(**gpaw_kwargs)
        except Exception as exc:
            if parallel.get('use_elpa'):
                _parprint(f"     ⚠️  ELPA unavailable ({exc}); retrying without ELPA")
                parallel.pop('use_elpa', None)
                if parallel:
                    gpaw_kwargs['parallel'] = parallel
                else:
                    gpaw_kwargs.pop('parallel', None)
                # Disable ELPA for the remainder of the run to avoid repeated failures.
                RUN_CONFIG['elpa'] = False
                calc = GPAW(**gpaw_kwargs)
            else:
                raise

    if checkpoint and not h2_kpts:
        try:
            calc.attach(calc.write, checkpoint_every_scf, str(gpw_path), mode='all')
        except Exception as exc:
            _parprint(f"     ⚠️  Could not attach checkpoint writer ({exc})")
        _write_gpw_meta(label)

    return calc, restart_used


def _prepare_slab(slab_file):
    """Load slab and enforce minimum vacuum."""
    slab = read(slab_file)
    if slab.cell[2, 2] < 10:
        slab.cell[2, 2] = 15
        slab.center(axis=2, vacuum=0)
    return slab


def _reapply_bottom_fix(atoms):
    """Reapply bottom-half FixAtoms constraint if missing."""
    if atoms.constraints:
        return
    positions = atoms.get_positions()
    z = positions[:, 2]
    z_mid = (np.min(z) + np.max(z)) / 2
    fixed = [i for i in range(len(atoms)) if atoms[i].z < z_mid]
    if fixed:
        atoms.set_constraint(FixAtoms(indices=fixed))


def _maybe_relax(atoms, label):
    """Optional local relaxation with ASE BFGS trajectory + Hessian restart."""
    if not USE_RELAXATION:
        return atoms, False

    _reapply_bottom_fix(atoms)

    label = str(label)
    traj_path = f"{label}.traj"
    bfgs_restart = f"{label}.bfgs.json"
    relax_log = f"{label}_relax.log"

    restart_used = False
    if os.path.exists(traj_path):
        try:
            prev_calc = atoms.calc
            atoms = read(traj_path, index=-1)
            _reapply_bottom_fix(atoms)
            atoms.calc = prev_calc
            restart_used = True
            _parprint(f"     ↺ BFGS resume from trajectory: {traj_path}")
        except Exception as exc:
            _parprint(f"     ⚠️  Failed to read existing trajectory ({exc}); starting BFGS fresh")

    optimizer = BFGS(
        atoms,
        logfile=relax_log,
        trajectory=traj_path,
        restart=bfgs_restart,
        append_trajectory=True,
    )
    optimizer.run(fmax=RELAXATION_CONFIG['fmax'], steps=RELAXATION_CONFIG['steps'])
    return atoms, restart_used


def _candidate_h_positions(slab, h_distance=1.5, max_sites=None,
                           site_search=None, formula=None):
    """Generate adsorption candidate sites.

    site_search='basic' preserves the legacy top + bridge_center + centroid behaviour.
    site_search='rich' adds nearest-neighbour bridge sites, hollow centroids from
    triples of top atoms, and dual-metal bridges for interface structures.
    """
    if site_search is None:
        site_search = RUN_CONFIG.get('site_search', 'basic')
    if max_sites is None:
        max_sites = 6 if site_search == 'basic' else 20

    positions = slab.get_positions()
    symbols = list(slab.get_chemical_symbols())
    top_z = np.max(positions[:, 2])
    z_tol = 0.35

    top_indices = [i for i, pos in enumerate(positions) if (top_z - pos[2]) <= z_tol]
    if not top_indices:
        center_xy = np.mean(positions[:, :2], axis=0)
        return [("center_fallback", [center_xy[0], center_xy[1], top_z + h_distance])]

    top_xy = positions[top_indices, :2]
    top_syms = [symbols[i] for i in top_indices]
    center_xy = np.mean(top_xy, axis=0)
    distances = [np.linalg.norm(xy - center_xy) for xy in top_xy]
    order = sorted(range(len(top_indices)), key=lambda k: distances[k])
    ranked_xy = [top_xy[k] for k in order]
    ranked_syms = [top_syms[k] for k in order]

    candidates = []
    for i, xy in enumerate(ranked_xy[:max_sites]):
        candidates.append((f"top_{i+1}", [float(xy[0]), float(xy[1]), float(top_z + h_distance)]))

    if len(ranked_xy) >= 2:
        bridge_xy = 0.5 * (ranked_xy[0] + ranked_xy[1])
        candidates.append(("bridge_center", [float(bridge_xy[0]), float(bridge_xy[1]), float(top_z + h_distance)]))

    candidates.append(("top_centroid", [float(center_xy[0]), float(center_xy[1]), float(top_z + h_distance)]))

    if site_search == 'rich':
        # Pairwise bridge sites between nearest top-layer atoms
        n = len(ranked_xy)
        pair_seen = set()
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(ranked_xy[i] - ranked_xy[j]))
                if d > 4.0:  # skip far-apart pairs
                    continue
                key = (i, j)
                if key in pair_seen:
                    continue
                pair_seen.add(key)
                bxy = 0.5 * (ranked_xy[i] + ranked_xy[j])
                tag = f"bridge_{i+1}_{j+1}"
                candidates.append((tag, [float(bxy[0]), float(bxy[1]), float(top_z + h_distance)]))
                # Dual-metal bridge: highlight interface bonds
                if ranked_syms[i] != ranked_syms[j]:
                    candidates.append(
                        (f"dual_{ranked_syms[i]}{ranked_syms[j]}_{i+1}_{j+1}",
                         [float(bxy[0]), float(bxy[1]), float(top_z + h_distance)])
                    )

        # Hollow sites from nearest triples
        if n >= 3:
            for i in range(min(n, 6)):
                for j in range(i + 1, min(n, 7)):
                    for k in range(j + 1, min(n, 8)):
                        d_ij = np.linalg.norm(ranked_xy[i] - ranked_xy[j])
                        d_jk = np.linalg.norm(ranked_xy[j] - ranked_xy[k])
                        d_ik = np.linalg.norm(ranked_xy[i] - ranked_xy[k])
                        if max(d_ij, d_jk, d_ik) > 4.0:
                            continue
                        hxy = (ranked_xy[i] + ranked_xy[j] + ranked_xy[k]) / 3.0
                        tag = f"hollow_{i+1}_{j+1}_{k+1}"
                        candidates.append((tag, [float(hxy[0]), float(hxy[1]), float(top_z + h_distance)]))

        # MoS2/MoSe2 edge bias: prefer top-layer chalcogen atoms
        if formula and formula in ('MoS2', 'MoSe2'):
            target = 'S' if formula == 'MoS2' else 'Se'
            for idx, sym in enumerate(ranked_syms):
                if sym == target:
                    xy = ranked_xy[idx]
                    candidates.append(
                        (f"edge_{target}_{idx+1}",
                         [float(xy[0]), float(xy[1]), float(top_z + h_distance)])
                    )

    # Dedupe by xy proximity, cap at max_sites
    unique = []
    for label, pos in candidates:
        duplicate = any(np.allclose(pos[:2], other_pos[:2], atol=1e-3) for _, other_pos in unique)
        if not duplicate:
            unique.append((label, pos))
        if len(unique) >= max_sites:
            break

    _parprint(f"     ↳ H site candidates ({site_search}): {len(unique)}")
    return unique


def calculate_clean_slab_energy(slab, output_dir):
    """Calculate energy of clean surface (no adsorbate). Returns (energy, restart_used)."""
    try:
        _parprint(f"  └─ Calculating clean slab energy...")
        slab = slab.copy()
        label = f'{output_dir}/clean_slab'
        calc, calc_restart = setup_gpaw_calculator(label=label)
        slab.calc = calc
        slab, relax_restart = _maybe_relax(slab, label)
        energy = slab.get_potential_energy()
        _parprint(f"     ✓ Clean slab: E = {energy:.6f} eV")
        return energy, (calc_restart or relax_restart)
    except Exception as e:
        _parprint(f"     ✗ Error: {e}")
        return None, False


def calculate_slab_with_h_energy(slab, output_dir, h_distance=1.5, formula=None):
    """Calculate energy of surface with H adsorbate. Returns (energy, site, pos, restart_used)."""
    try:
        _parprint(f"  └─ Calculating slab+H energy...")
        candidates = _candidate_h_positions(slab, h_distance=h_distance, formula=formula)

        best_energy = None
        best_site = None
        best_position = None
        any_restart = False

        for site_label, h_pos in candidates:
            slab_with_h = slab.copy()
            slab_with_h += Atoms('H', positions=[h_pos])

            label = f'{output_dir}/slab_with_h_{site_label}'
            calc, calc_restart = setup_gpaw_calculator(label=label)
            slab_with_h.calc = calc
            slab_with_h, relax_restart = _maybe_relax(slab_with_h, label)
            any_restart = any_restart or calc_restart or relax_restart

            energy = slab_with_h.get_potential_energy()
            _parprint(f"     · site={site_label:<24} E = {energy:.6f} eV")

            if best_energy is None or energy < best_energy:
                best_energy = energy
                best_site = site_label
                best_position = h_pos

        _parprint(f"     ✓ Best slab+H: site={best_site}, E = {best_energy:.6f} eV")
        return best_energy, best_site, best_position, any_restart
    except Exception as e:
        _parprint(f"     ✗ Error: {e}")
        return None, None, None, False


def calculate_h2_molecule_energy(output_dir):
    """Calculate energy of isolated H2 molecule using current basis/xc/mode."""
    try:
        _parprint(f"  └─ Calculating H₂ molecule energy...")
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]])
        h2.center(vacuum=10)
        label = f'{output_dir}/h2_molecule'
        calc, _ = setup_gpaw_calculator(label=label, h2_kpts=True, checkpoint=False)
        h2.calc = calc
        energy = h2.get_potential_energy()
        _parprint(f"     ✓ H₂ molecule: E = {energy:.6f} eV")
        return energy
    except Exception as e:
        _parprint(f"     ✗ Error: {e}")
        return None


def get_h2_reference_path():
    """Return profile-specific H2 reference cache path."""
    stage = RUN_CONFIG.get('stage', 'screening')
    mode = GPAW_CONFIG.get('mode', 'lcao')
    xc = GPAW_CONFIG.get('xc', 'LDA')
    basis = GPAW_CONFIG.get('basis', 'dzp')
    relax = RELAXATION_CONFIG.get('steps', 0)
    name = f"h2_reference_{stage}_{mode}_{xc}_{basis}_k111_relax{relax}.json"
    return DATA_OUTPUTS / name


def get_h2_reference_energy():
    """Get H2 reference energy from profile-specific cache or compute once."""
    h2_file = RUN_CONFIG.get('h2_reference_file') or get_h2_reference_path()
    h2_file = Path(h2_file)
    if h2_file.exists():
        try:
            with open(h2_file, 'r') as f:
                payload = json.load(f)
            energy = float(payload['E_h2_eV'])
            _parprint(f"✓ Loaded cached H2 reference: {energy:.6f} eV ({h2_file.name})")
            return energy, "cache"
        except Exception as exc:
            _parprint(f"⚠️  Ignoring invalid H2 cache ({exc}), recomputing...")

    h2_dir = GPAW_OUTPUTS / f"h2_reference_{RUN_CONFIG.get('stage','screening')}_{GPAW_CONFIG.get('basis','dzp')}_{GPAW_CONFIG.get('xc','LDA')}"
    if _is_rank0():
        h2_dir.mkdir(parents=True, exist_ok=True)
    _barrier()

    energy = calculate_h2_molecule_energy(str(h2_dir))
    if energy is None:
        raise RuntimeError("Failed to compute H2 reference energy; aborting run")

    if _is_rank0():
        h2_file.parent.mkdir(parents=True, exist_ok=True)
        with open(h2_file, 'w') as f:
            json.dump(
                {
                    'E_h2_eV': float(energy),
                    'timestamp': datetime.now().isoformat(),
                    'stage': RUN_CONFIG.get('stage'),
                    'gpaw_config': {
                        'mode': GPAW_CONFIG.get('mode'),
                        'basis': GPAW_CONFIG.get('basis'),
                        'xc': GPAW_CONFIG.get('xc'),
                        'kpts': [1, 1, 1],
                    },
                    'relaxation_steps': RELAXATION_CONFIG['steps'],
                    'use_relaxation': USE_RELAXATION,
                },
                f,
                indent=2,
            )
        _parprint(f"✓ Saved H2 reference cache: {h2_file}")
    _barrier()
    return energy, "computed"


def calculate_surface_properties(formula, miller, slab_file, output_dir, e_h2, h2_source):
    """Calculate all energies and ΔGH for a surface; write extended CSV row."""
    _parprint(f"\n{'='*60}")
    _parprint(f"Calculating {formula} {miller}")
    _parprint(f"{'='*60}")

    if _is_rank0():
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    _barrier()

    result = {
        'formula': formula,
        'surface': miller,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',
        'E_clean_slab': None,
        'E_slab_with_h': None,
        'E_h2': None,
        'ΔGH': None,
        'adsorption_site': None,
        'h_position': None,
        'h2_source': h2_source,
        'relaxed': bool(USE_RELAXATION),
        'restart_used': False,
        'error': None,
    }

    try:
        if not os.path.exists(slab_file):
            raise FileNotFoundError(f"POSCAR not found: {slab_file}")

        slab = _prepare_slab(slab_file)

        e_clean, restart_clean = calculate_clean_slab_energy(slab, output_dir)
        if e_clean is None:
            raise RuntimeError("Failed to calculate clean slab energy")
        result['E_clean_slab'] = float(e_clean)

        e_with_h, best_site, best_position, restart_h = calculate_slab_with_h_energy(
            slab, output_dir, formula=formula
        )
        if e_with_h is None:
            raise RuntimeError("Failed to calculate slab+H energy")
        result['E_slab_with_h'] = float(e_with_h)
        result['adsorption_site'] = best_site
        result['h_position'] = best_position

        result['E_h2'] = float(e_h2)
        delta_gh = e_with_h - e_clean - 0.5 * e_h2
        result['ΔGH'] = float(delta_gh)
        result['status'] = 'completed'
        result['restart_used'] = bool(restart_clean or restart_h)

        _parprint(f"\n  Results for {formula} {miller}:")
        _parprint(f"    E(clean slab) = {e_clean:.6f} eV")
        _parprint(f"    E(slab+H)     = {e_with_h:.6f} eV")
        _parprint(f"    E(H₂)         = {e_h2:.6f} eV")
        _parprint(f"    ΔGH           = {delta_gh:.6f} eV ← KEY RESULT!")

        if -0.2 < delta_gh < 0.2:
            _parprint("    ✓ EXCELLENT! Near optimal for HER")
        elif -0.5 < delta_gh < 0.5:
            _parprint("    ○ GOOD, reasonable for HER")
        else:
            _parprint("    ⚠️  Outside typical HER range")

    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        _parprint(f"  ✗ Error: {e}")

    meta = _result_metadata(restart_used=result['restart_used'])
    source_tag = f"GPAW_{GPAW_CONFIG.get('xc', 'LDA')}"
    row = [
        result['formula'],
        result['surface'],
        'H',
        result['E_clean_slab'],
        result['E_slab_with_h'],
        result['E_h2'],
        result['ΔGH'],
        result['ΔGH'],  # descriptor_eV = ΔGH
        source_tag,
        result.get('adsorption_site'),
        result.get('h2_source'),
        result.get('relaxed'),
        result['status'],
        result['timestamp'],
        meta['stage'], meta['basis'], meta['xc'], meta['kpts'],
        meta['relax_steps'], meta['fmax'], meta['site_search'],
        meta['mpi_ranks'], meta['omp_threads'], meta['restart_used'],
    ]
    _append_result_csv(row, RUN_CONFIG['output_csv'])

    return result


def _write_failed_row(formula, surface, h2_source, reason):
    """Write a failed result row directly (used for timeout/worker failures)."""
    meta = _result_metadata(restart_used=False)
    source_tag = f"GPAW_{GPAW_CONFIG.get('xc', 'LDA')}"
    row = [
        formula, surface, 'H',
        None, None, None, None, None,
        source_tag,
        None, h2_source, bool(USE_RELAXATION),
        'failed', datetime.now().isoformat(),
        meta['stage'], meta['basis'], meta['xc'], meta['kpts'],
        meta['relax_steps'], meta['fmax'], meta['site_search'],
        meta['mpi_ranks'], meta['omp_threads'], meta['restart_used'],
    ]
    _append_result_csv(row, RUN_CONFIG['output_csv'])
    _parprint(f"  ✗ Marked failed: {formula} {surface} ({reason})")


def _timeout_handler(signum, frame):
    raise TimeoutError("Per-structure timeout reached")


def _compute_one(args):
    """Worker function for parallel execution."""
    formula, surface, poscar_dir, e_h2, h2_source, max_seconds = args
    poscar_file = poscar_dir / "POSCAR"
    output_dir = GPAW_OUTPUTS / poscar_dir.name

    if max_seconds and max_seconds > 0:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(int(max_seconds))

    t0 = time.time()
    try:
        return calculate_surface_properties(
            formula=formula,
            miller=surface,
            slab_file=str(poscar_file),
            output_dir=str(output_dir),
            e_h2=e_h2,
            h2_source=h2_source,
        )
    except TimeoutError as exc:
        _write_failed_row(formula, surface, h2_source, str(exc))
        return {
            'formula': formula,
            'surface': surface,
            'timestamp': datetime.now().isoformat(),
            'status': 'failed',
            'error': str(exc),
        }
    finally:
        elapsed_min = (time.time() - t0) / 60.0
        print(f"⏱ Finished worker task {formula} {surface} in {elapsed_min:.1f} min")
        if max_seconds and max_seconds > 0:
            signal.alarm(0)


def run_calculations_parallel(formulas=['MoS2', 'MoSe2', 'MoP', 'Mo2N'],
                             millers=['(100)', '(110)', '(111)'],
                             base_dir=None,
                             results_file=None,
                             use_discovery=True,
                             include_patterns=None,
                             selected_structure_names=None,
                             workers_override=None,
                             max_hours_per_structure=None):
    """Run all calculations with parallel workers and checkpoint/resume.

    In MPI mode (RUN_CONFIG['mpi_enabled']) the ProcessPoolExecutor is disabled
    and structures are processed sequentially; parallelism comes from GPAW
    internals (ScaLAPACK / ELPA via mpiexec).
    """

    _parprint("\n" + "="*60)
    _parprint("GPAW H Adsorption Energy Calculator (SOTA two-stage)")
    _parprint("="*60)
    _parprint(f"\nStarting time: {datetime.now()}")
    _parprint(f"Stage: {RUN_CONFIG.get('stage')} | basis={GPAW_CONFIG.get('basis')} "
              f"| xc={GPAW_CONFIG.get('xc')} | kpts={_kpts_str()} | site_search={RUN_CONFIG.get('site_search')}")
    _parprint(f"Relaxation: {'ON (' + str(RELAXATION_CONFIG['steps']) + ' steps)' if USE_RELAXATION else 'OFF'}")
    if include_patterns:
        _parprint(f"Include filter: {', '.join(include_patterns)}")
    if max_hours_per_structure and max_hours_per_structure > 0:
        _parprint(f"Per-structure timeout: {max_hours_per_structure:.2f} hours (calibration mode)")

    base_dir = Path(base_dir) if base_dir else DATA_INPUTS

    output_csv = RUN_CONFIG['output_csv']
    _ensure_csv_header(output_csv)

    completed = _load_completed_keys(output_csv)
    if completed:
        _parprint(f"✓ Checkpoint: {len(completed)} structures already completed, will skip")

    e_h2, h2_source = get_h2_reference_energy()

    all_results = []

    if use_discovery:
        structures = discover_structures(base_dir, include_patterns=include_patterns)
        structures = filter_structures_by_name(structures, selected_structure_names)
        if not structures:
            _parprint("\n⚠️  No POSCAR files found in data/inputs/VASP_inputs")
            return all_results

        pending = []
        max_seconds = 0
        if max_hours_per_structure and max_hours_per_structure > 0:
            max_seconds = int(max_hours_per_structure * 3600)
        for formula, surface, poscar_dir in structures:
            if (formula, surface) in completed:
                continue
            pending.append((formula, surface, poscar_dir, e_h2, h2_source, max_seconds))

        _parprint(f"Structures to compute: {len(pending)} (skipped {len(structures) - len(pending)} completed)")

        if not pending:
            _parprint("✓ All structures already completed!")
            return all_results

        mpi_mode = bool(RUN_CONFIG.get('mpi_enabled'))
        if mpi_mode:
            max_workers = 1
        else:
            max_workers = _detect_max_workers(override_workers=workers_override)

        if max_workers <= 1:
            for args in pending:
                if _shutdown_requested:
                    _parprint("⚠️  Shutdown: stopping before next structure")
                    break
                result = _compute_one(args)
                all_results.append(result)
        else:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_compute_one, args): args for args in pending}
                for future in as_completed(future_map):
                    if _shutdown_requested:
                        _parprint("⚠️  Shutdown: cancelling remaining futures")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    try:
                        result = future.result()
                        all_results.append(result)
                    except Exception as exc:
                        args = future_map[future]
                        _parprint(f"  ✗ Worker exception for {args[0]} {args[1]}: {exc}")
    else:
        for formula in formulas:
            for miller in millers:
                if _shutdown_requested:
                    break
                if (formula, miller) in completed:
                    continue
                dir_name = f"{formula}_{miller.replace('(', '').replace(')', '')}"
                poscar_dir = Path(base_dir) / f"{formula}_{miller}"
                poscar_file = poscar_dir / "POSCAR"
                output_dir = GPAW_OUTPUTS / dir_name

                result = calculate_surface_properties(
                    formula=formula,
                    miller=miller,
                    slab_file=str(poscar_file),
                    output_dir=str(output_dir),
                    e_h2=e_h2,
                    h2_source=h2_source,
                )
                all_results.append(result)

    return all_results


def save_results(results, json_file=None, csv_file=None):
    """Save summary JSON and print statistics (rank 0 only)."""
    if not _is_rank0():
        _barrier()
        return

    _parprint("\n" + "="*60)
    _parprint("Saving Results")
    _parprint("="*60)

    json_file = Path(json_file) if json_file else Path(RUN_CONFIG['output_json'])
    csv_file = Path(csv_file) if csv_file else Path(RUN_CONFIG['output_csv'])

    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    _parprint(f"✓ Full results saved to: {json_file}")
    _parprint(f"✓ Incremental CSV at: {csv_file}")

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        df = pd.DataFrame()

    if len(df) > 0:
        completed = df[df['status'] == 'completed']
        _parprint(f"\nTotal entries in CSV: {len(df)}")
        _parprint(f"Successfully calculated: {len(completed)} surfaces")

        if len(completed) > 0:
            _parprint("\nΔGH statistics by formula:")
            for formula in sorted(completed['formula'].unique()):
                formula_df = completed[completed['formula'] == formula]
                dgh_values = formula_df['ΔGH_eV'].dropna().values
                if len(dgh_values) > 0:
                    _parprint(f"  {formula}:")
                    _parprint(f"    Count: {len(dgh_values)}")
                    _parprint(f"    Mean ΔGH: {np.mean(dgh_values):.4f} eV")
                    _parprint(f"    Min ΔGH:  {np.min(dgh_values):.4f} eV (best surface)")
                    _parprint(f"    Max ΔGH:  {np.max(dgh_values):.4f} eV")

            excellent = completed[completed['ΔGH_eV'].abs() < 0.2]
            if len(excellent) > 0:
                _parprint(f"\n✓ EXCELLENT candidates (|ΔGH| < 0.2 eV): {len(excellent)}")
                _parprint(excellent[['formula', 'surface_facet', 'ΔGH_eV']].to_string(index=False))

        failed = df[df['status'] == 'failed']
        if len(failed) > 0:
            _parprint(f"\n⚠️  Failed calculations: {len(failed)}")
            _parprint(failed[['formula', 'surface_facet', 'status']].to_string(index=False))

    _parprint(f"\nEnd time: {datetime.now()}")
    _barrier()


# ── Pre-defined machine splits ───────────────────────────────────
MACHINE_SPLITS = {
    # DEVANA (64 CPUs): 159 structures (~46.5%)
    'devana': [
        'Ni_Mo2N_interface_*',
        'Ni_MoS2_interface_*',
        'Mo2N_*',
        'MoS2_*',
        'MoSe2_*',
        'MoP_*',
        'graphene_*',
        'Ni2_on_*',
        'Ni4_on_*',
    ],
    # NODE1 (32 CPUs): 79 structures (~23.1%)
    'node1': [
        'Ni_MoB_interface_*',
        'MoB_*',
    ],
    # NODE2 (24 CPUs): 61 structures (~17.8%)
    'node2': [
        'Ni_Mo2C_interface_*',
        'Ni_Ti3C2O2_interface_*',
        'Ti3C2O2_*',
    ],
    # NODE3 (16 CPUs): 43 structures (~12.6%)
    'node3': [
        'Mo2C_*',
    ],
}


def main():
    """Main execution"""

    global USE_RELAXATION, CORES_PER_CALC

    parser = argparse.ArgumentParser(
        description='GPAW H Adsorption Calculator (SOTA two-stage)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Machine splits (use --machine instead of --include):\n'
            + '\n'.join(f'  {k:14s} → {" , ".join(v)}' for k, v in MACHINE_SPLITS.items())
        ),
    )
    parser.add_argument('--include', type=str, default=None,
        help='Comma-separated glob patterns to filter structures.')
    parser.add_argument('--machine', type=str, default=None, choices=list(MACHINE_SPLITS.keys()),
        help='Use a pre-defined split for a specific machine.')
    parser.add_argument('--list', action='store_true', dest='list_only',
        help='Only list structures that would be computed, then exit.')
    parser.add_argument('--write-structure-list', type=str, default=None,
        help='Write matched structure directory names to a text file, one per line, then exit.')
    parser.add_argument('--structure-name', action='append', default=None,
        help='Run only the given structure directory name (repeatable or comma-separated).')
    parser.add_argument('--structure-list', type=str, default=None,
        help='Path to a text file listing structure directory names (one per line; '
             '# comments and blank lines ignored). Merged with --structure-name.')
    parser.add_argument('--workers', type=int, default=None,
        help='Manual number of parallel workers (overrides auto-detection).')
    parser.add_argument('--cores-per-calc', type=int, default=None,
        help='CPU cores budget per calculation used by auto worker detection.')
    parser.add_argument('--relax-steps', type=int, default=None,
        help='Override relaxation steps.')
    parser.add_argument('--fmax', type=float, default=None,
        help='Override BFGS force threshold in eV/A.')
    parser.add_argument('--no-relax', action='store_true',
        help='Disable structural relaxation (single-point energies only).')
    parser.add_argument('--kpts', type=str, default=None,
        help='Override k-point mesh as comma-separated triple, e.g. 2,2,1.')
    parser.add_argument('--basis', type=str, default=None,
        help='Override LCAO basis set, e.g. sz, szp, dzp.')
    parser.add_argument('--xc', type=str, default=None,
        help='Override exchange-correlation functional, e.g. LDA, PBE, RPBE.')
    parser.add_argument('--max-hours-per-structure', type=float, default=0.0,
        help='Hard timeout per structure in hours (0 disables timeout). Calibration only.')

    # SOTA additions
    parser.add_argument('--stage', type=str, default='screening',
        choices=list(STAGE_PROFILES.keys()),
        help='Two-stage funnel profile: screening (sz/LDA fast) or refinement (dzp/PBE).')
    parser.add_argument('--site-search', type=str, default=None, choices=['basic', 'rich'],
        help='H adsorption site search density (default depends on --stage).')
    parser.add_argument('--mpi', action='store_true',
        help='Force MPI execution mode (disables ProcessPoolExecutor; parallelism inside GPAW).')
    parser.add_argument('--scalapack', action='store_true',
        help='Enable GPAW ScaLAPACK (parallel={"sl_auto": True}).')
    parser.add_argument('--elpa', action='store_true',
        help='Enable GPAW ELPA (opt-in; falls back to non-ELPA on failure).')
    parser.add_argument('--checkpoint-every-scf', type=int, default=5,
        help='Write .gpw checkpoint every N SCF steps (default 5).')
    parser.add_argument('--no-checkpoint', action='store_true',
        help='Disable periodic .gpw checkpoint writes.')

    args = parser.parse_args()

    # ── 1. Apply stage profile defaults ────────────────────────────
    profile = STAGE_PROFILES[args.stage]
    GPAW_CONFIG['basis'] = profile['basis']
    GPAW_CONFIG['xc'] = profile['xc']
    GPAW_CONFIG['kpts'] = profile['kpts']
    GPAW_CONFIG['h'] = profile.get('h', 0.20)
    GPAW_CONFIG['mixer'] = profile.get('mixer')
    GPAW_CONFIG['maxiter'] = profile.get('maxiter', 333)
    GPAW_CONFIG['convergence'] = profile.get('convergence', GPAW_CONFIG['convergence'])
    RELAXATION_CONFIG['steps'] = profile['relax_steps']
    RELAXATION_CONFIG['fmax'] = profile['fmax']
    RUN_CONFIG['stage'] = args.stage
    RUN_CONFIG['site_search'] = args.site_search or profile['site_search']
    RUN_CONFIG['output_csv'] = DATA_OUTPUTS / profile['output_csv_name']
    RUN_CONFIG['output_json'] = DATA_OUTPUTS / profile['output_json_name']

    # ── 2. MPI detection / forcing ─────────────────────────────────
    mpi_world = _mpi_size()
    mpi_enabled = bool(args.mpi) or (mpi_world > 1)
    RUN_CONFIG['mpi_enabled'] = mpi_enabled
    RUN_CONFIG['mpi_world_size'] = mpi_world
    RUN_CONFIG['mpi_rank'] = _mpi_rank()
    if mpi_enabled:
        _parprint(f"MPI mode detected: world.size={mpi_world}, rank={_mpi_rank()}; disabling ProcessPoolExecutor")

    # ── 3. ScaLAPACK / ELPA ────────────────────────────────────────
    # ScaLAPACK default-on in MPI mode (per spec); user --scalapack also enables it.
    RUN_CONFIG['scalapack'] = bool(args.scalapack) or mpi_enabled
    RUN_CONFIG['elpa'] = bool(args.elpa)

    # ── 4. Checkpoint settings ─────────────────────────────────────
    RUN_CONFIG['checkpoint_enabled'] = not args.no_checkpoint
    RUN_CONFIG['checkpoint_every_scf'] = max(1, int(args.checkpoint_every_scf))

    # ── 5. Resolve include patterns / structure names ──────────────
    include_patterns = None
    if args.machine:
        include_patterns = MACHINE_SPLITS[args.machine]
        _parprint(f"Using machine split: {args.machine}")
    elif args.include:
        include_patterns = [p.strip() for p in args.include.split(',')]

    selected_structure_names = None
    if args.structure_name or args.structure_list:
        selected_structure_names = []
        if args.structure_name:
            for value in args.structure_name:
                selected_structure_names.extend(part.strip() for part in value.split(',') if part.strip())
        if args.structure_list:
            list_path = Path(args.structure_list)
            if not list_path.is_file():
                raise FileNotFoundError(f"--structure-list file not found: {list_path}")
            for raw in list_path.read_text().splitlines():
                line = raw.split('#', 1)[0].strip()
                if line:
                    selected_structure_names.append(line)
        # de-duplicate while preserving order
        seen = set()
        selected_structure_names = [n for n in selected_structure_names if not (n in seen or seen.add(n))]
        _parprint(f"Explicit structures ({len(selected_structure_names)}): "
                  f"{', '.join(selected_structure_names[:8])}"
                  f"{' …' if len(selected_structure_names) > 8 else ''}")

    # ── 6. CLI overrides (applied AFTER stage profile) ─────────────
    if args.cores_per_calc is not None:
        CORES_PER_CALC = max(1, int(args.cores_per_calc))
        _parprint(f"Override: CORES_PER_CALC={CORES_PER_CALC}")

    _set_thread_env(CORES_PER_CALC)

    if args.no_relax:
        USE_RELAXATION = False
        _parprint("Override: relaxation disabled")

    if args.relax_steps is not None:
        RELAXATION_CONFIG['steps'] = max(0, int(args.relax_steps))
        _parprint(f"Override: relax_steps={RELAXATION_CONFIG['steps']}")

    if args.fmax is not None:
        RELAXATION_CONFIG['fmax'] = float(args.fmax)
        _parprint(f"Override: fmax={RELAXATION_CONFIG['fmax']}")

    if args.kpts:
        try:
            parts = tuple(int(x.strip()) for x in args.kpts.split(','))
            if len(parts) != 3:
                raise ValueError("kpts must have 3 integers")
            GPAW_CONFIG['kpts'] = parts
            _parprint(f"Override: kpts={GPAW_CONFIG['kpts']}")
        except Exception as exc:
            raise ValueError(f"Invalid --kpts '{args.kpts}': {exc}")

    if args.basis:
        GPAW_CONFIG['basis'] = args.basis.strip()
        _parprint(f"Override: basis={GPAW_CONFIG['basis']}")

    if args.xc:
        GPAW_CONFIG['xc'] = args.xc.strip()
        _parprint(f"Override: xc={GPAW_CONFIG['xc']}")

    RUN_CONFIG['max_hours_per_structure'] = float(args.max_hours_per_structure or 0.0)

    # ── 7. Resolve profile-specific H2 reference path AFTER overrides ──
    RUN_CONFIG['h2_reference_file'] = get_h2_reference_path()

    _parprint(f"\nRUN_CONFIG: stage={RUN_CONFIG['stage']} basis={GPAW_CONFIG['basis']} "
              f"xc={GPAW_CONFIG['xc']} kpts={_kpts_str()} site_search={RUN_CONFIG['site_search']} "
              f"mpi={RUN_CONFIG['mpi_enabled']}({RUN_CONFIG['mpi_world_size']}) "
              f"scalapack={RUN_CONFIG['scalapack']} elpa={RUN_CONFIG['elpa']} "
              f"checkpoint={'every '+str(RUN_CONFIG['checkpoint_every_scf'])+' SCF' if RUN_CONFIG['checkpoint_enabled'] else 'OFF'}")
    _parprint(f"Outputs: CSV={RUN_CONFIG['output_csv']}")
    _parprint(f"         JSON={RUN_CONFIG['output_json']}")
    _parprint(f"         H2 cache={RUN_CONFIG['h2_reference_file']}")

    # ── 8. List-only / manifest mode ───────────────────────────────
    if args.list_only or args.write_structure_list:
        structures = discover_structures(DATA_INPUTS, include_patterns=include_patterns)
        structures = filter_structures_by_name(structures, selected_structure_names)
        _parprint(f"\nStructures matched: {len(structures)}")
        for formula, surface, poscar_dir in structures:
            _parprint(f"  {poscar_dir.name}")
        if args.write_structure_list and _is_rank0():
            output_path = Path(args.write_structure_list)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(''.join(f"{poscar_dir.name}\n" for _, _, poscar_dir in structures))
            _parprint(f"\n✓ Wrote structure manifest: {output_path}")
        return

    # ── 9. Single-structure scheduler hint ─────────────────────────
    if selected_structure_names and not RUN_CONFIG['mpi_enabled']:
        args.workers = 1
        _parprint("Single-structure mode: forcing workers=1 for scheduler-friendly execution")

    # ── 10. Run calculations ───────────────────────────────────────
    results = run_calculations_parallel(
        include_patterns=include_patterns,
        selected_structure_names=selected_structure_names,
        workers_override=args.workers,
        max_hours_per_structure=args.max_hours_per_structure,
    )

    save_results(results)

    _parprint("\n✓ Done! Results saved to:")
    _parprint(f"  - {RUN_CONFIG['output_csv']} (incremental, crash-safe)")
    _parprint(f"  - {RUN_CONFIG['output_json']} (JSON summary)")
    _parprint("\nNext step: ranking from Stage 1 → Stage 2 refinement on shortlist.")


if __name__ == '__main__':
    main()
