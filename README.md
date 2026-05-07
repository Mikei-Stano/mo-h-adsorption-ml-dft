# Mo H Adsorption ML-DFT Workflow

This repository contains a two-stage ML-to-DFT workflow for computing hydrogen adsorption energies ($\Delta G_H$) on Mo-based compounds and interfaces. The current pipeline uses MACE-MP-0 for fast Stage 0 prescreening and GPAW for Stage 1 screening plus Stage 2 refinement.

## Structure

- scripts/ — runnable scripts (MACE prescreening, GPAW screening/refinement, launchers)
- data/inputs/ — POSCAR inputs
- data/outputs/ — cached results, shortlists, H2 references, and GPAW outputs
- docs/ — methodology and reports

## Quick Start

1. Create POSCAR inputs
   - Run scripts/generate_structures.py
   - Generates basal slabs, edge ribbons, vacancies, dopants, and interface models
2. Run Stage 0 ML prescreening
   - Run scripts/mace_screening.py
   - Produces data/outputs/mace_h_adsorption_stage0_screening.csv and JSON cache files
3. Run Stage 1 GPAW screening
   - Run scripts/gpaw_h_adsorption.py --stage screening
   - For MPI-enabled desktops, prefer scripts/run_node1_mpi.sh
   - For non-MPI desktops, use scripts/run_desktop_machine.sh or scripts/run_node1_threaded.sh
   - For DEVANA, use scripts/submit_devana_gpaw_array.sh
4. Run Stage 2 GPAW refinement
   - Re-run the shortlist with --stage refinement and a manifest from data/outputs/manifests/

## Main Outputs

- data/outputs/mace_h_adsorption_stage0_screening.csv
- data/outputs/gpaw_h_adsorption_stage1_screening_sz.csv
- data/outputs/manifests/stage1_shortlist_*.txt
- data/outputs/h2_reference_*.json

## Notes

- GPAW must be installed and accessible in your environment.
- For the MPI path, run `mpiexec -n N <python> scripts/gpaw_h_adsorption.py ...`; do not use `gpaw python`.
- Calculations are configured for slab models with vacuum spacing.
- Noble metal dopant shortlist is in dopant_shortlist.json.
- DEVANA usage:
   - Generate and submit an array with ACCOUNT=myproject bash scripts/submit_devana_gpaw_array.sh
   - Each array task runs exactly one structure through scripts/devana_gpaw_array_worker.sh
- Desktop usage (no SLURM):
   - Choose one existing machine profile in scripts/gpaw_h_adsorption.py (`node1`, `node2`, or `node3`) based on available CPU/RAM and assign that profile to your PC.
   - Run scripts/run_desktop_machine.sh with that profile via `MACHINE=<profile>`.
   - If your split differs, update `MACHINE_SPLITS` in scripts/gpaw_h_adsorption.py (or use `--include` patterns directly) so each PC gets a non-overlapping subset of structures.
   - Tune local concurrency with `CORES_PER_CALC` and optional `WORKERS`.

## GitHub Transfer

- The repository is portable across machines via `PYENV_ROOT`, `ENV_NAME`, `PYBIN`, `PYTHON`, `RANKS`, and `MACHINE` environment variables.
- Stage 0 prescreening CSV/JSON, Stage 1 screening CSV/JSON, manifests, and H2 reference caches are intentionally versioned so other nodes can resume without recomputing them.

## Updated Direction (2026-02-11)

- MoS2/MoSe2 basal planes are inert; next modeling should target edge sites and defects.
- Mo2N remains the strongest candidate; refine with defects/dopants.
- Use OCx24 to shortlist noble metal dopants for Ni/Mo composite interfaces.
- Use VASP only for final validation of top candidates.
