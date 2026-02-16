"""
Generate POSCAR files for Mo compounds
Creates surface slabs using literature lattice parameters and ASE
"""

import os
from pathlib import Path
from ase import Atoms
from ase.io import write
from ase.constraints import FixAtoms
from ase.build import fcc111, fcc100
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

def _apply_constraints(slab):
    """Freeze bottom half of atoms for stability."""
    z_positions = slab.get_positions()[:, 2]
    z_min = np.min(z_positions)
    z_max = np.max(z_positions)
    z_mid = (z_min + z_max) / 2
    fixed_indices = [i for i in range(len(slab)) if slab[i].z < z_mid]
    slab.set_constraint(FixAtoms(indices=fixed_indices))


def create_slab(bulk_atoms, size=(2, 2, 4), vacuum=8):
    """Create a basal surface slab from bulk structure."""
    slab = bulk_atoms.repeat((size[0], size[1], 1))
    slab.cell[2, 2] *= size[2]
    slab.center(vacuum=vacuum, axis=2)
    _apply_constraints(slab)
    return slab


def create_edge_ribbon(bulk_atoms, width=6, length=2, vacuum=8, edge_type="Mo"):
    """Create a simple edge ribbon by adding vacuum in x and z.

    edge_type:
        "Mo"  -> remove chalcogen atoms at ribbon edges
        "X"   -> remove Mo atoms at ribbon edges (X = S or Se)
    """
    ribbon = bulk_atoms.repeat((width, length, 1))

    # Create vacuum in x and z to form edges and a single-layer ribbon
    ribbon.center(vacuum=vacuum, axis=0)
    ribbon.center(vacuum=vacuum, axis=2)
    ribbon.set_pbc([False, True, False])

    # Determine edge atoms by x position
    positions = ribbon.get_positions()
    x_positions = positions[:, 0]
    x_min = np.min(x_positions)
    x_max = np.max(x_positions)
    tol = 0.3  # Angstroms
    edge_mask = (x_positions - x_min < tol) | (x_max - x_positions < tol)

    # Remove atoms at edges to approximate termination
    if edge_type == "Mo":
        remove_symbols = {"S", "Se"}
    else:
        remove_symbols = {"Mo"}

    remove_indices = [
        i for i, atom in enumerate(ribbon)
        if edge_mask[i] and atom.symbol in remove_symbols
    ]
    if remove_indices:
        del ribbon[remove_indices]

    _apply_constraints(ribbon)
    return ribbon


def create_vacancy_slab(slab, vacancy_symbol):
    """Remove one top-layer atom to create a vacancy."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]
    z_max = np.max(z_positions)
    tol = 0.5  # Angstroms

    candidates = [
        i for i, atom in enumerate(slab)
        if atom.symbol == vacancy_symbol and (z_max - z_positions[i]) < tol
    ]
    if not candidates:
        return slab

    # Remove the candidate closest to xy center
    center_xy = np.mean(positions[:, :2], axis=0)
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in candidates]
    remove_index = candidates[int(np.argmin(distances))]
    del slab[remove_index]
    return slab


def create_substitution_slab(slab, target_symbol, dopant_symbol):
    """Substitute one top-layer atom with a dopant."""
    positions = slab.get_positions()
    z_positions = positions[:, 2]
    z_max = np.max(z_positions)
    tol = 0.5  # Angstroms

    candidates = [
        i for i, atom in enumerate(slab)
        if atom.symbol == target_symbol and (z_max - z_positions[i]) < tol
    ]
    if not candidates:
        return slab

    center_xy = np.mean(positions[:, :2], axis=0)
    distances = [np.linalg.norm(positions[i, :2] - center_xy) for i in candidates]
    replace_index = candidates[int(np.argmin(distances))]
    slab[replace_index].symbol = dopant_symbol
    return slab


def create_ni_slab(miller="(111)", size=(3, 3, 4), vacuum=8):
    """Create a simple Ni slab for interface models."""
    a = 3.52  # Angstrom, fcc Ni
    if miller == "(100)":
        slab = fcc100("Ni", size=size, a=a, vacuum=vacuum)
    else:
        slab = fcc111("Ni", size=size, a=a, vacuum=vacuum)
    slab.set_pbc([True, True, False])
    return slab


def create_ni_mo2n_interface(miller="(111)", separation=2.2):
    """Create a simple Ni/Mo2N interface slab (trend-level model)."""
    ni = create_ni_slab(miller=miller, size=(3, 3, 4), vacuum=8)
    mo2n_bulk = create_mo2n_bulk()
    mo2n = create_slab(mo2n_bulk, size=(2, 2, 2), vacuum=0)
    mo2n.set_pbc([True, True, False])

    # Stack Mo2N on top of Ni
    ni_positions = ni.get_positions()
    ni_top = np.max(ni_positions[:, 2])

    mo2n_positions = mo2n.get_positions()
    mo2n_shift = ni_top + separation - np.min(mo2n_positions[:, 2])
    mo2n.translate([0.0, 0.0, mo2n_shift])

    # Define combined cell with additional vacuum
    cell = ni.cell.copy()
    cell[2, 2] = np.max(mo2n.get_positions()[:, 2]) + 8.0

    interface = ni + mo2n
    interface.set_cell(cell)
    interface.set_pbc([True, True, False])

    # Freeze bottom half of Ni atoms only
    ni_indices = [i for i, atom in enumerate(interface) if atom.symbol == "Ni"]
    ni_z = interface.get_positions()[ni_indices, 2]
    ni_z_mid = (np.min(ni_z) + np.max(ni_z)) / 2
    fixed_indices = [i for i in ni_indices if interface[i].z < ni_z_mid]
    interface.set_constraint(FixAtoms(indices=fixed_indices))
    return interface


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
            
            # Create basal slabs for each miller index
            for miller in millers:
                dir_name = base_dir / f"{formula}_{miller}"
                dir_name.mkdir(parents=True, exist_ok=True)
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
        
            # Create vacancy slabs for MoS2 and MoSe2
            if formula in ["MoS2", "MoSe2"]:
                vacancy_symbol = "S" if formula == "MoS2" else "Se"
                for miller in millers:
                    dir_name = base_dir / f"{formula}_{miller}_vac{vacancy_symbol}"
                    dir_name.mkdir(parents=True, exist_ok=True)
                    poscar_file = dir_name / "POSCAR"
                    print(f"    {miller} vac{vacancy_symbol}: ", end="", flush=True)
                    try:
                        slab = create_slab(bulk.copy())
                        slab = create_vacancy_slab(slab, vacancy_symbol)
                        write(str(poscar_file), slab, format='vasp')
                        print(f"✓ ({len(slab)} atoms)")
                    except Exception as e:
                        print(f"✗ Error: {e}")

                # Create edge ribbons
                edge_variants = [("Mo", f"{formula}_edge_Mo"), ("X", f"{formula}_edge_{vacancy_symbol}")]
                for edge_type, name in edge_variants:
                    dir_name = base_dir / name
                    dir_name.mkdir(parents=True, exist_ok=True)
                    poscar_file = dir_name / "POSCAR"
                    print(f"    edge {edge_type}: ", end="", flush=True)
                    try:
                        ribbon = create_edge_ribbon(bulk.copy(), edge_type=edge_type)
                        write(str(poscar_file), ribbon, format='vasp')
                        print(f"✓ ({len(ribbon)} atoms)")
                    except Exception as e:
                        print(f"✗ Error: {e}")

            # Create Mo2N vacancies and dopants
            if formula == "Mo2N":
                dopants = ["Pt", "Pd", "Ir", "Ru", "Ni"]
                for miller in millers:
                    # N vacancy
                    dir_name = base_dir / f"{formula}_{miller}_vacN"
                    dir_name.mkdir(parents=True, exist_ok=True)
                    poscar_file = dir_name / "POSCAR"
                    print(f"    {miller} vacN: ", end="", flush=True)
                    try:
                        slab = create_slab(bulk.copy())
                        slab = create_vacancy_slab(slab, "N")
                        write(str(poscar_file), slab, format='vasp')
                        print(f"✓ ({len(slab)} atoms)")
                    except Exception as e:
                        print(f"✗ Error: {e}")

                    # Mo vacancy
                    dir_name = base_dir / f"{formula}_{miller}_vacMo"
                    dir_name.mkdir(parents=True, exist_ok=True)
                    poscar_file = dir_name / "POSCAR"
                    print(f"    {miller} vacMo: ", end="", flush=True)
                    try:
                        slab = create_slab(bulk.copy())
                        slab = create_vacancy_slab(slab, "Mo")
                        write(str(poscar_file), slab, format='vasp')
                        print(f"✓ ({len(slab)} atoms)")
                    except Exception as e:
                        print(f"✗ Error: {e}")

                    # Mo-site dopants
                    for dopant in dopants:
                        dir_name = base_dir / f"{formula}_{miller}_dop{dopant}"
                        dir_name.mkdir(parents=True, exist_ok=True)
                        poscar_file = dir_name / "POSCAR"
                        print(f"    {miller} dop{dopant}: ", end="", flush=True)
                        try:
                            slab = create_slab(bulk.copy())
                            slab = create_substitution_slab(slab, "Mo", dopant)
                            write(str(poscar_file), slab, format='vasp')
                            print(f"✓ ({len(slab)} atoms)")
                        except Exception as e:
                            print(f"✗ Error: {e}")

                # Simple Ni/Mo2N interface models
                for miller in ["(111)", "(100)"]:
                    dir_name = base_dir / f"Ni_Mo2N_interface_{miller}"
                    dir_name.mkdir(parents=True, exist_ok=True)
                    poscar_file = dir_name / "POSCAR"
                    print(f"    interface {miller}: ", end="", flush=True)
                    try:
                        interface = create_ni_mo2n_interface(miller=miller)
                        write(str(poscar_file), interface, format='vasp')
                        print(f"✓ ({len(interface)} atoms)")
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
