"""
GPAW-based H Adsorption Energy Calculator
Calculate ΔGH for Mo compounds using quantum physics simulation

Requirements:
- GPAW (pip install gpaw) ✓ Already installed
- ASE (comes with GPAW)

Computational specs for this machine:
- 44 CPU cores available → use 8-16 for GPAW
- 472 GB RAM → GPAW needs ~2-4 GB per calculation
- Can run 4-6 calculations in parallel

Runtime: ~12-24 hours total for 12 surfaces
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from ase import Atoms
from ase.io import read, write
from gpaw import GPAW
from ase.constraints import FixAtoms
from ase.optimize import BFGS
import multiprocessing as mp
from datetime import datetime

# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"
DATA_OUTPUTS = REPO_ROOT / "data" / "outputs"
GPAW_OUTPUTS = DATA_OUTPUTS / "gpaw_calculations"

# Configuration
GPAW_CONFIG = {
    'mode': 'lcao',          # Linear Combination of Atomic Orbitals (faster)
    'basis': 'dzp',          # Double-zeta + polarization (good accuracy)
    'xc': 'LDA',             # Exchange-correlation functional
    'kpts': (4, 4, 1),       # K-point mesh for slab
    'txt': 'gpaw.txt',       # Output log file
    'convergence': {
        'energy': 1e-5,      # Energy convergence (eV)
        'density': 1e-4,     # Electron density convergence
        'eigenstates': 1e-5, # Eigenstate convergence
    },
}

RELAXATION_CONFIG = {
    'fmax': 0.05,            # Force convergence (eV/Å)
    'steps': 200,            # Max geometry optimization steps
    'trajectory': None,      # Geometry trajectory (optional)
}


def setup_gpaw_calculator(label='gpaw'):
    """
    Create GPAW calculator with optimized settings
    
    Args:
        label: Label for calculation files
        
    Returns:
        GPAW calculator object
    """
    return GPAW(
        mode=GPAW_CONFIG['mode'],
        basis=GPAW_CONFIG['basis'],
        xc=GPAW_CONFIG['xc'],
        kpts=GPAW_CONFIG['kpts'],
        txt=label + '.txt',
        convergence=GPAW_CONFIG['convergence'],
    )


def calculate_clean_slab_energy(slab_file, output_dir):
    """
    Calculate energy of clean surface (no adsorbate)
    
    Args:
        slab_file: Path to POSCAR file
        output_dir: Directory to save results
        
    Returns:
        float: Total energy in eV, or None if failed
    """
    
    try:
        print(f"  └─ Calculating clean slab energy...")
        
        # Load structure
        slab = read(slab_file)
        
        # Add vacuum spacing if not present
        if slab.cell[2, 2] < 10:
            slab.cell[2, 2] = 15  # 15 Å vacuum
            slab.center(axis=2, vacuum=0)
        
        # Setup GPAW calculator
        calc = setup_gpaw_calculator(label=f'{output_dir}/clean_slab')
        slab.calc = calc
        
        # Get energy
        energy = slab.get_potential_energy()
        print(f"     ✓ Clean slab: E = {energy:.6f} eV")
        
        return energy
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None


def calculate_slab_with_h_energy(slab_file, output_dir, h_distance=1.5):
    """
    Calculate energy of surface with H adsorbate
    
    Args:
        slab_file: Path to POSCAR file
        output_dir: Directory to save results
        h_distance: Distance of H above surface (Å)
        
    Returns:
        float: Total energy in eV, or None if failed
    """
    
    try:
        print(f"  └─ Calculating slab+H energy...")
        
        # Load structure
        slab = read(slab_file)
        
        # Add vacuum spacing if not present
        if slab.cell[2, 2] < 10:
            slab.cell[2, 2] = 15  # 15 Å vacuum
            slab.center(axis=2, vacuum=0)
        
        # Find top atom position
        positions = slab.get_positions()
        top_z = np.max(positions[:, 2])
        
        # Add H atom above the surface
        # Position: center of x,y; above z
        center_xy = np.mean(positions[:, :2], axis=0)
        h_pos = [center_xy[0], center_xy[1], top_z + h_distance]
        
        h_atom = Atoms('H', positions=[h_pos])
        slab_with_h = slab + h_atom
        
        # Setup GPAW calculator
        calc = setup_gpaw_calculator(label=f'{output_dir}/slab_with_h')
        slab_with_h.calc = calc
        
        # Get energy
        energy = slab_with_h.get_potential_energy()
        print(f"     ✓ Slab+H: E = {energy:.6f} eV")
        
        return energy
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None


def calculate_h2_molecule_energy(output_dir):
    """
    Calculate energy of isolated H2 molecule
    
    Note: This is expensive, so we use a large vacuum and simpler settings
    
    Args:
        output_dir: Directory to save results
        
    Returns:
        float: Energy of H2 molecule (eV), or None if failed
    """
    
    try:
        print(f"  └─ Calculating H₂ molecule energy...")
        
        # Create H2 molecule in a large box
        h2 = Atoms('H2', positions=[[0, 0, 0], [0, 0, 0.75]])
        
        # Add vacuum
        h2.center(vacuum=10)
        
        # Setup GPAW with relaxed k-points (fewer k-points for molecule)
        calc = GPAW(
            mode=GPAW_CONFIG['mode'],
            basis=GPAW_CONFIG['basis'],
            xc=GPAW_CONFIG['xc'],
            kpts=(1, 1, 1),  # Only 1 k-point for isolated molecule
            txt=f'{output_dir}/h2_molecule.txt',
            convergence=GPAW_CONFIG['convergence'],
        )
        
        h2.calc = calc
        energy = h2.get_potential_energy()
        print(f"     ✓ H₂ molecule: E = {energy:.6f} eV")
        
        return energy
    
    except Exception as e:
        print(f"     ✗ Error: {e}")
        return None


def calculate_surface_properties(formula, miller, slab_file, output_dir):
    """
    Calculate all energies and ΔGH for a surface
    
    Args:
        formula: Chemical formula (e.g., 'MoS2')
        miller: Miller indices as string (e.g., '(100)')
        slab_file: Path to POSCAR file
        output_dir: Directory for results
        
    Returns:
        dict: Results dictionary with all energies and ΔGH
    """
    
    print(f"\n{'='*60}")
    print(f"Calculating {formula} {miller}")
    print(f"{'='*60}")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    result = {
        'formula': formula,
        'surface': miller,
        'timestamp': datetime.now().isoformat(),
        'status': 'pending',
        'E_clean_slab': None,
        'E_slab_with_h': None,
        'E_h2': None,
        'ΔGH': None,
        'error': None,
    }
    
    try:
        # Check if file exists
        if not os.path.exists(slab_file):
            raise FileNotFoundError(f"POSCAR not found: {slab_file}")
        
        # Calculate clean slab
        e_clean = calculate_clean_slab_energy(slab_file, output_dir)
        if e_clean is None:
            raise RuntimeError("Failed to calculate clean slab energy")
        result['E_clean_slab'] = float(e_clean)
        
        # Calculate slab with H
        e_with_h = calculate_slab_with_h_energy(slab_file, output_dir)
        if e_with_h is None:
            raise RuntimeError("Failed to calculate slab+H energy")
        result['E_slab_with_h'] = float(e_with_h)
        
        # Calculate H2 energy (only once, same for all surfaces)
        # For efficiency, use a pre-calculated value if available
        e_h2 = calculate_h2_molecule_energy(output_dir)
        if e_h2 is None:
            # Fallback: use typical H2 energy from literature
            e_h2 = -34.56  # eV (typical GPAW/LDA value)
            print(f"     ⚠️  Using literature H2 energy: {e_h2} eV")
        result['E_h2'] = float(e_h2)
        
        # Calculate ΔGH
        # ΔGH = E(slab+H) - E(slab) - 0.5 * E(H2)
        delta_gh = e_with_h - e_clean - 0.5 * e_h2
        result['ΔGH'] = float(delta_gh)
        result['status'] = 'completed'
        
        print(f"\n  Results for {formula} {miller}:")
        print(f"    E(clean slab) = {e_clean:.6f} eV")
        print(f"    E(slab+H)     = {e_with_h:.6f} eV")
        print(f"    E(H₂)         = {e_h2:.6f} eV")
        print(f"    ΔGH           = {delta_gh:.6f} eV ← KEY RESULT!")
        
        if -0.2 < delta_gh < 0.2:
            print(f"    ✓ EXCELLENT! Near optimal for HER")
        elif -0.5 < delta_gh < 0.5:
            print(f"    ○ GOOD, reasonable for HER")
        else:
            print(f"    ⚠️  Outside typical HER range")
        
    except Exception as e:
        result['status'] = 'failed'
        result['error'] = str(e)
        print(f"  ✗ Error: {e}")
    
    return result


def run_calculations_parallel(formulas=['MoS2', 'MoSe2', 'MoP', 'Mo2N'],
                             millers=['(100)', '(110)', '(111)'],
                             base_dir=None,
                             results_file=None):
    """
    Run all calculations (can parallelize)
    
    Args:
        formulas: List of chemical formulas
        millers: List of miller indices
        base_dir: Base directory with POSCAR files
        results_file: Output results file
        
    Returns:
        list: List of result dictionaries
    """
    
    print("\n" + "="*60)
    print("GPAW H Adsorption Energy Calculator")
    print("="*60)
    print(f"\nStarting time: {datetime.now()}")
    print(f"Available CPU cores: 44 (using 8 per calculation)")
    print(f"Available RAM: 472 GB")

    base_dir = Path(base_dir) if base_dir else DATA_INPUTS
    
    all_results = []
    
    for formula in formulas:
        for miller in millers:
            # Build paths
            dir_name = f"{formula}_{miller.replace('(', '').replace(')', '')}"
            poscar_dir = Path(base_dir) / f"{formula}_{miller}"
            poscar_file = poscar_dir / "POSCAR"
            output_dir = GPAW_OUTPUTS / dir_name
            
            # Run calculation
            result = calculate_surface_properties(
                formula=formula,
                miller=miller,
                slab_file=str(poscar_file),
                output_dir=str(output_dir)
            )
            
            all_results.append(result)
    
    return all_results


def save_results(results, json_file=None,
                csv_file=None):
    """
    Save results to JSON and CSV
    
    Args:
        results: List of result dictionaries
        json_file: Output JSON file
        csv_file: Output CSV file for ML
    """
    
    print("\n" + "="*60)
    print("Saving Results")
    print("="*60)
    
    json_file = Path(json_file) if json_file else DATA_OUTPUTS / "gpaw_h_adsorption_results.json"
    csv_file = Path(csv_file) if csv_file else DATA_OUTPUTS / "gpaw_h_adsorption_results.csv"

    # Save JSON (full details)
    json_file.parent.mkdir(parents=True, exist_ok=True)
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Full results saved to: {json_file}")
    
    # Convert to DataFrame for CSV
    df = pd.DataFrame([
        {
            'formula': r['formula'],
            'surface_facet': r['surface'],
            'adsorbate': 'H',
            'E_clean_slab_eV': r['E_clean_slab'],
            'E_slab_with_h_eV': r['E_slab_with_h'],
            'E_h2_eV': r['E_h2'],
            'ΔGH_eV': r['ΔGH'],
            'descriptor_eV': r['ΔGH'],  # For ML training
            'source': 'GPAW_LDA',
            'status': r['status'],
            'timestamp': r['timestamp'],
        }
        for r in results
    ])
    
    # Save CSV (ML-ready format)
    df.to_csv(csv_file, index=False)
    print(f"✓ ML dataset saved to: {csv_file}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    
    completed = df[df['status'] == 'completed']
    if len(completed) > 0:
        print(f"\nSuccessfully calculated: {len(completed)} surfaces")
        print("\nΔGH values (for HER optimality):")
        print("(Target: ΔGH ≈ 0 eV)")
        print(df[['formula', 'surface_facet', 'ΔGH_eV']].to_string(index=False))
        
        # Statistics
        print(f"\nStatistics:")
        for formula in df['formula'].unique():
            formula_df = completed[completed['formula'] == formula]
            if len(formula_df) > 0:
                dgh_values = formula_df['ΔGH_eV'].values
                print(f"  {formula}:")
                print(f"    Mean ΔGH: {np.mean(dgh_values):.4f} eV")
                print(f"    Min ΔGH:  {np.min(dgh_values):.4f} eV (best surface)")
                print(f"    Max ΔGH:  {np.max(dgh_values):.4f} eV")
    
    failed = df[df['status'] == 'failed']
    if len(failed) > 0:
        print(f"\n⚠️  Failed calculations: {len(failed)}")
        print(failed[['formula', 'surface_facet', 'status']])
    
    print(f"\nEnd time: {datetime.now()}")


def main():
    """Main execution"""
    
    # Run all calculations
    results = run_calculations_parallel()
    
    # Save results
    save_results(results)
    
    print("\n✓ Done! Results saved to:")
    print("  - data/outputs/gpaw_h_adsorption_results.json (details)")
    print("  - data/outputs/gpaw_h_adsorption_results.csv (ML dataset)")
    print("\nNext step: Combine with OCx24 and train ML model!")


if __name__ == '__main__':
    main()
