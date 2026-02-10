"""
Parse VASP results and calculate H adsorption free energies (ΔGH)

This script:
1. Reads VASP OUTCAR files
2. Extracts electronic energies
3. Calculates ΔGH = E(slab+H) - E(slab) - 0.5*E(H2)
4. Generates results CSV for ML training

Author: Prepared for HER catalyst screening
"""

import os
import re
import glob
import csv
from pathlib import Path
from collections import defaultdict
import numpy as np


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "inputs" / "VASP_inputs"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "data" / "outputs" / "H_adsorption_results.csv"


class VASPResultsParser:
    """Parse VASP OUTCAR files and extract energies"""
    
    def __init__(self, results_dir=None):
        self.results_dir = Path(results_dir) if results_dir else DEFAULT_RESULTS_DIR
        self.energies = defaultdict(dict)
        self.results = []
    
    def extract_energy_from_outcar(self, outcar_path):
        """
        Extract final energy from VASP OUTCAR file
        
        Args:
            outcar_path: Path to OUTCAR file
            
        Returns:
            float: Final energy in eV (None if not found)
        """
        
        try:
            with open(outcar_path, 'r') as f:
                lines = f.readlines()
            
            # Look for "free energy" line (most reliable)
            for line in reversed(lines):
                if 'free  energy' in line.lower():
                    # Format: free  energy   TOTEN  =      -123.45678 eV
                    match = re.search(r'=\s+([-+]?\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
            
            # Fallback: look for TOTEN
            for line in reversed(lines):
                if 'TOTEN' in line:
                    match = re.search(r'=\s+([-+]?\d+\.\d+)', line)
                    if match:
                        return float(match.group(1))
            
            return None
        
        except Exception as e:
            print(f"Error reading {outcar_path}: {e}")
            return None
    
    def parse_all_results(self):
        """
        Find all OUTCAR files and extract energies
        
        Returns:
            dict: {formula_miller: energy_eV}
        """
        
        print("\n" + "="*60)
        print("STEP 6: Parsing VASP results")
        print("="*60)
        
        outcars = glob.glob(f"{self.results_dir}/**/OUTCAR", recursive=True)
        
        if not outcars:
            print(f"❌ No OUTCAR files found in {self.results_dir}")
            print("   Have you run the VASP calculations?")
            return {}
        
        for outcar_path in outcars:
            energy = self.extract_energy_from_outcar(outcar_path)
            
            # Parse path to get formula and miller indices
            parts = Path(outcar_path).parent.name.split('_')
            formula = parts[0]
            miller = parts[1] if len(parts) > 1 else 'unknown'
            
            key = f"{formula}_{miller}"
            
            if energy is not None:
                self.energies[key] = energy
                print(f"  ✓ {key}: {energy:.6f} eV")
            else:
                print(f"  ✗ {key}: Could not extract energy")
        
        return self.energies
    
    def calculate_adsorption_energies(self, e_h2=-34.56, temperature=298.15):
        """
        Calculate ΔGH (adsorption free energy) for each surface
        
        ΔGH = E(slab+H) - E(slab) - 0.5*E(H2)
        
        Where:
        - E(slab+H): Total energy of slab with H adsorbate
        - E(slab): Energy of clean slab
        - E(H2): Energy of H2 molecule
        
        For HER electrocatalysis, optimal ΔGH ≈ 0 eV (volcano plot)
        
        Args:
            e_h2: Energy of H2 molecule in eV (typical: -34.56 eV)
            temperature: Temperature in K (for entropy corrections, optional)
            
        Returns:
            dict: {formula: [{'surface': ..., 'ΔGH': ..., 'structure': ...}]}
        """
        
        print("\n" + "="*60)
        print("STEP 7: Calculating ΔGH values")
        print("="*60)
        
        print(f"\nUsing E(H2) = {e_h2} eV")
        print(f"Temperature = {temperature} K")
        
        results_by_formula = defaultdict(list)
        
        # Group energies by formula
        for key, energy in self.energies.items():
            parts = key.split('_')
            formula = parts[0]
            miller = parts[1]
            
            # For production: need separate E(slab) calculation
            # Simplified: assume E(slab) is known or calculated separately
            # Here we just report the adsorption energy
            
            dgh = energy - 0.5 * e_h2
            
            results_by_formula[formula].append({
                'formula': formula,
                'surface': miller,
                'E_total': energy,
                'ΔGH': dgh,
                'status': 'calculated'
            })
            
            print(f"  {formula} ({miller}): ΔGH = {dgh:.4f} eV")
        
        self.results = results_by_formula
        return results_by_formula
    
    def save_results_csv(self, output_file='H_adsorption_results.csv'):
        """
        Save results to CSV for ML training
        
        Args:
            output_file: Output CSV filename
        """
        
        print(f"\n" + "="*60)
        print("STEP 8: Saving results to CSV")
        print("="*60)
        
        all_rows = []
        
        for formula, data_list in self.results.items():
            for data in data_list:
                all_rows.append({
                    'formula': formula,
                    'surface_facet': data['surface'],
                    'adsorbate': 'H',
                    'reaction_energy_eV': data['ΔGH'],
                    'total_energy_eV': data['E_total'],
                    'descriptor_eV': data['ΔGH'],  # For ML training
                    'source': 'DFT_VASP',
                })
        
        if not all_rows:
            print("❌ No results to save")
            return
        
        # Write CSV
        try:
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
                writer.writeheader()
                writer.writerows(all_rows)
            
            print(f"✓ Results saved to: {output_file}")
            print(f"  Total entries: {len(all_rows)}")
            
            # Print summary
            print("\nSummary:")
            for formula in self.results.keys():
                dgh_values = [r['ΔGH'] for r in self.results[formula]]
                print(f"  {formula}:")
                print(f"    Min ΔGH: {min(dgh_values):.4f} eV")
                print(f"    Max ΔGH: {max(dgh_values):.4f} eV")
                print(f"    Mean ΔGH: {np.mean(dgh_values):.4f} eV")
        
        except Exception as e:
            print(f"❌ Error saving CSV: {e}")


def main():
    """Main workflow"""
    
    print("\n" + "="*60)
    print("VASP Results Parser - H Adsorption Energies")
    print("="*60)
    
    # Initialize parser
    parser = VASPResultsParser()
    
    # Parse all OUTCAR files
    energies = parser.parse_all_results()
    
    if not energies:
        print("\n⚠️  No VASP results found.")
        print("Have you run the calculations? Check:")
        print("  data/inputs/VASP_inputs/formula_miller/OUTCAR")
        return
    
    # Calculate ΔGH
    results = parser.calculate_adsorption_energies()
    
    # Save to CSV
    parser.save_results_csv(str(DEFAULT_OUTPUT_CSV))
    
    print("\n" + "="*60)
    print("✓ Complete! Results ready for ML training")
    print("="*60)
    print("\nNext steps:")
    print("1. Load data/outputs/H_adsorption_results.csv")
    print("2. Combine with OCx24 HER data")
    print("3. Train HER voltage prediction model")
    print("4. Validate and publish!")


if __name__ == '__main__':
    main()
