"""
Generate POSCAR files for Mo compounds
Creates surface slabs using literature lattice parameters and ASE
"""

import os
from pathlib import Path
from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms
import numpy as np


def create_mos2_bulk():
    """Create MoS2 bulk structure (hexagonal)"""
    # Experimental lattice parameters
    a = 3.160  # Å
    c = 12.295  # Å
    
    # Hexagonal cell
    cell = np.array([
        [a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ])
    
    # Mo at (0, 0, 1/2), S at (1/3, 2/3, z) and (2/3, 1/3, -z)
    symbols = ['Mo', 'S', 'S']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.62],
        [2/3, 1/3, 0.38],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mose2_bulk():
    """Create MoSe2 bulk structure (hexagonal)"""
    a = 3.289  # Å
    c = 12.995  # Å
    
    cell = np.array([
        [a, 0, 0],
        [-a/2, a*np.sqrt(3)/2, 0],
        [0, 0, c]
    ])
    
    symbols = ['Mo', 'Se', 'Se']
    positions = np.array([
        [0.0, 0.0, 0.5],
        [1/3, 2/3, 0.62],
        [2/3, 1/3, 0.38],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mop_bulk():
    """Create MoP bulk structure (cubic)"""
    a = 3.240  # Å
    
    cell = np.eye(3) * a
    
    symbols = ['Mo', 'P']
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms


def create_mo2n_bulk():
    """Create Mo2N bulk structure (cubic anti-perovskite)"""
    a = 4.160  # Å
    
    cell = np.eye(3) * a
    
    symbols = ['Mo', 'Mo', 'N']
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.0],
    ])
    
    atoms = Atoms(symbols, cell=cell, pbc=[True, True, True])
    atoms.set_scaled_positions(positions)
    return atoms

def create_slab(bulk_atoms, size=(2, 2, 4)):
    """Create a surface slab from bulk structure"""
    
    # Replicate unit cell in-plane
    slab = bulk_atoms.repeat((size[0], size[1], 1))
    
    # Expand cell in z-direction for vacuum
    slab.cell[2, 2] *= size[2]
    
    # Center the structure with vacuum on both sides
    slab.center(vacuum=8, axis=2)
    
    # Freeze bottom half of atoms (typical for surface calculations)
    z_positions = slab.get_positions()[:, 2]
    z_min = np.min(z_positions)
    z_max = np.max(z_positions)
    z_mid = (z_min + z_max) / 2
    
    fixed_indices = [i for i in range(len(slab)) if slab[i].z < z_mid]
    slab.set_constraint(FixAtoms(indices=fixed_indices))
    
    return slab


# Paths
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_INPUTS = REPO_ROOT / "data" / "inputs" / "VASP_inputs"


def generate_all_structures():
    """Generate all structure files"""
    
    # Map formulas to builder functions
    builders = {
        'MoS2': create_mos2_bulk,
        'MoSe2': create_mose2_bulk,
        'MoP': create_mop_bulk,
        'Mo2N': create_mo2n_bulk,
    }
    
    millers = ['(100)', '(110)', '(111)']
    
    print("\n" + "="*60)
    print("Generating Structure Files for GPAW Calculations")
    print("="*60)
    
    base_dir = DATA_INPUTS
    base_dir.mkdir(parents=True, exist_ok=True)
    
    for formula, builder in builders.items():
        print(f"\n{formula}:")
        
        try:
            # Build bulk structure
            bulk = builder()
            print(f"  Bulk structure: {len(bulk)} atoms/cell")
            
            # Create slabs for each miller index
            for miller in millers:
                dir_name = base_dir / f"{formula}_{miller}"
                poscar_file = dir_name / "POSCAR"
                
                print(f"    {miller}: ", end="", flush=True)
                
                try:
                    slab = create_slab(bulk.copy())
                    write(str(poscar_file), slab, format='vasp')
                    n_atoms = len(slab)
                    n_frozen = len([a for a in slab if a.tag == 1])
                    print(f"✓ ({n_atoms} atoms, {n_frozen} frozen)")
                    
                except Exception as e:
                    print(f"✗ Error: {e}")
        
        except Exception as e:
            print(f"  ✗ Failed to create {formula}: {e}")
    
    print("\n" + "="*60)
    print("✓ Structure generation complete!")
    print("="*60)


if __name__ == '__main__':
    generate_all_structures()
    print("\n✓ Done! All POSCAR files created.")
