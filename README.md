# Mo H Adsorption Workflow

This repository contains the scripts, inputs, outputs, and documentation for computing hydrogen adsorption energies ($\Delta G_H$) on Mo-based compounds (MoS₂, MoSe₂, MoP, Mo₂N) using GPAW.

## Structure

- scripts/ — runnable scripts (GPAW + structure preparation)
- data/inputs/ — POSCAR inputs
- data/outputs/ — results and GPAW calculation logs
- docs/ — methodology and reports

## Quick Start

1. Create POSCAR inputs
   - Run scripts/generate_structures.py
2. Run GPAW calculations
   - Run scripts/gpaw_h_adsorption.py
3. Results
   - data/outputs/gpaw_h_adsorption_results.csv

## Main Outputs

- data/outputs/gpaw_h_adsorption_results.csv
- data/outputs/gpaw_h_adsorption_results.json

## Notes

- GPAW must be installed and accessible in your environment.
- Calculations are configured for slab models with vacuum spacing.
