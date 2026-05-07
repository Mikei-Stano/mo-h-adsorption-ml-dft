# Mo H Adsorption ML-DFT Workflow

A three-stage ML-to-DFT pipeline for high-throughput screening of hydrogen adsorption
free energies ($\Delta G_H$) on Mo-based catalysts (MoB, MoS₂, MoSe₂, Mo₂N, Mo₂C, MoP)
and Ni/Mo interface models for HER applications.

## Pipeline overview

| Stage | Method | Basis / XC | Structures | Purpose |
|-------|--------|------------|------------|---------|
| 0 | MACE-MP-0 | universal ML FF | 342 | pre-filter, rank by ΔGH |
| 1 | GPAW LCAO | sz / LDA | 30 (shortlist) | fast DFT screening |
| 2 | GPAW LCAO | dzp / PBE | 8 (excellent) | quantitative refinement |

Stage 0 and Stage 1 pre-computed results are versioned in `data/outputs/` so other nodes
can skip the expensive calculation steps.

## Repository layout

```
scripts/              MACE prescreening, GPAW screening/refinement, launchers
data/inputs/          POSCAR slab inputs
data/outputs/         cached Stage 0 & 1 results, manifests, H2 references
dopant_shortlist.json noble-metal dopant candidates (Ag, Au, Ir, Ni, Pd, Pt, Ru)
requirements.txt      Python dependencies
```

## Setup

```bash
# Using pyenv (default env name: cemea-env)
pyenv install 3.10.x
pyenv virtualenv 3.10.x cemea-env
pip install -r requirements.txt
# GPAW with MPI + ScaLAPACK: run scripts/rebuild_gpaw_with_mpi.sh
```

## Running the pipeline

### Stage 0 — ML prescreening (9–10 h, skippable if CSV present)
```bash
python scripts/mace_screening.py
```

### Stage 1 — GPAW screening (MPI, ~2–3 h for 30 structures)
```bash
# MPI launcher (portable; set RANKS, MACHINE, ENV_NAME as needed)
bash scripts/run_node1_mpi.sh

# Or directly:
mpiexec -n 16 python scripts/gpaw_h_adsorption.py \
    --stage screening --mpi --scalapack \
    --structure-list data/outputs/manifests/stage1_shortlist_abs_le_0p50.txt
```

### Stage 2 — GPAW refinement (MPI, ~1–2 days for 8 structures)
```bash
mpiexec -n 16 python scripts/gpaw_h_adsorption.py \
    --stage refinement --mpi --scalapack \
    --structure-list data/outputs/manifests/stage1_shortlist_abs_le_0p30.txt
```

### SLURM (DEVANA cluster)
```bash
ACCOUNT=myproject bash scripts/submit_devana_gpaw_array.sh
```

### Other desktop nodes
```bash
MACHINE=node2 bash scripts/run_desktop_machine.sh
```
Set `MACHINE` to `node1`, `node2`, or `node3` depending on the PC profile.
The launchers resolve the Python binary automatically from `PYENV_ROOT` and `ENV_NAME`.

## Pre-computed results (versioned)

| File | Contents |
|------|----------|
| `data/outputs/mace_h_adsorption_stage0_screening.csv` | 342-structure MACE-MP-0 ΔGH |
| `data/outputs/gpaw_h_adsorption_stage1_screening_sz.csv` | 27/30 GPAW Stage 1 results |
| `data/outputs/manifests/` | shortlists for Stage 2 |
| `data/outputs/h2_reference_*.json` | H2 energy caches per profile |

## Stage 1 highlights (sz/LDA/MPI, May 2026)

Top 8 candidates with |ΔGH| ≤ 0.30 eV (see `data/outputs/manifests/stage1_shortlist_abs_le_0p30.txt`).
The Mo₂C family was excluded from refinement (ΔGH ≈ −1.5 eV, far outside the HER optimum).

## Scientific context

This workflow follows the current SOTA for high-throughput HER screening:
ΔGH is used as a first-pass thermodynamic descriptor (Stage 0/1), followed by
higher-fidelity refinement (Stage 2, dzp/PBE) for the top candidates.
See `SOTA.txt` for a detailed discussion of methodological limitations and next steps
(solvation, kinetics, two-descriptor approaches).

## Notes

- Do not use `gpaw python` as the launcher; it swallows script arguments. Use `mpiexec -n N python scripts/gpaw_h_adsorption.py`.
- POTCAR files under `data/inputs/VASP_inputs/` are empty placeholders (VASP licence required to populate them).
- Noble metal dopant shortlist: `dopant_shortlist.json`.
