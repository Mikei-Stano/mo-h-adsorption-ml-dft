#!/bin/bash
#SBATCH --job-name=H_ads_Mo2N
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu  # or compute, depending on your cluster
#SBATCH --output=%x_%j.out

# Load modules
module load intel
module load vasp/6.4.2  # Adjust to your system

# Run VASP
mpirun -np 16 vasp > vasp.out

# Extract results
if [ -f OUTCAR ]; then
    echo "Calculation completed successfully"
    # Extract final energy
    grep "free  energy" OUTCAR | tail -1
fi
