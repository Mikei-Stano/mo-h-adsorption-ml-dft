# Noble Dopant Rerun Workflow

This folder contains scripts for a robust, reproducible rerun of noble metal dopant structures (Au, Ir, Pd, Pt, Ru) in high-quality (dzp/PBE) mode, separate from the main pipeline.

## Scripts
- `build_noble_manifest.py`: Generate a manifest of noble-dopant structures for rerun.
- `check_gpaw_setups.py`: Validate GPAW setup/basis availability for noble elements.
- `gpaw_setup_auto.py`: Portable GPAW setup-path autodetection helper.
- `run_noble_hq_stage.py`: Orchestrate the high-quality rerun for noble dopants.
- `summarize_noble_results.py`: Summarize results from the noble rerun output CSV.

## Usage
1. **Build manifest:**
   ```sh
   python build_noble_manifest.py --inputs data/inputs/VASP_inputs --output data/outputs/manifests/noble_dopants_all.txt
   ```
2. **Check GPAW setups:**
   ```sh
   python check_gpaw_setups.py --elements Au,Ir,Pd,Pt,Ru
   ```
3. **Run noble rerun:**
   ```sh
   python run_noble_hq_stage.py --manifest data/outputs/manifests/noble_dopants_all.txt
   ```
   - This script auto-detects GPAW setups and runs the workflow in background.
4. **Summarize results:**
   ```sh
   python summarize_noble_results.py --csv data/outputs/gpaw_h_adsorption_stage2_refinement_dzp.csv
   ```

## Notes
- All scripts are portable and auto-detect GPAW setup paths.
- No log or output artifacts are versioned.
- For details, see docstrings in each script.
