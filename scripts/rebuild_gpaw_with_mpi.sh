#!/usr/bin/env bash
# Rebuild GPAW with MPI + ScaLAPACK + LibXC + FFTW + native OpenBLAS.
#
# This is a one-time operation that unlocks the MPI launchers under
# scripts/run_node1_mpi.sh and the parallel ScaLAPACK/ELPA paths in
# scripts/gpaw_h_adsorption.py (already wired, currently dormant).
#
# Verified diagnosis (see logs/quickbench/): the current GPAW build reports
#   MPI enabled       no
#   scalapack         no (MPI unavailable)
#   Elpa              no (MPI unavailable)
#   libxc             no
#   BLAS              using scipy.linalg.blas and numpy.dot()
# This causes the 33% / 30% Python-grid bottleneck on MoB-class slabs.
#
# Usage:  sudo -v && bash scripts/rebuild_gpaw_with_mpi.sh
set -euo pipefail

PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
PYENV="${PYENV:-${PYENV_ROOT}/versions/${ENV_NAME}}"
PYTHON="$PYENV/bin/python"
PIP="$PYENV/bin/pip"

echo "================================================================"
echo "GPAW rebuild with MPI / ScaLAPACK / LibXC / FFTW"
echo "Pyenv:  $PYENV"
echo "Python: $($PYTHON --version)"
echo "Started: $(date)"
echo "================================================================"

if [[ $EUID -ne 0 ]] && ! sudo -n true 2>/dev/null; then
    echo "Need sudo for system packages. Run: sudo -v && $0"
    exit 1
fi

# 1. System libraries
echo "--- 1/5: installing system libraries ---"
sudo apt update
sudo apt install -y \
    build-essential gfortran \
    libopenmpi-dev openmpi-bin \
    libopenblas-dev libopenblas-openmp-dev \
    liblapack-dev libxc-dev \
    libfftw3-dev libfftw3-mpi-dev \
    libscalapack-openmpi-dev \
    libelpa-dev || echo "(libelpa-dev optional; continuing without ELPA)"

# 2. Verify mpiexec
echo "--- 2/5: verifying MPI runtime ---"
which mpiexec
mpiexec --version | head -3

# 3. Uninstall pip GPAW + recompile from source
echo "--- 3/5: uninstalling old GPAW ---"
"$PIP" uninstall -y gpaw || true

echo "--- 4/5: rebuilding GPAW with native libs ---"
# Using GPAW's auto-detection — it picks up libxc, libscalapack, libfftw3,
# openblas, and openmpi from /usr/lib automatically when present.
GPAW_NEW_FILES_GO_TO_DEVANA="$PWD/_gpaw_build_$(date +%Y%m%d_%H%M%S).log"
"$PIP" install --no-binary=gpaw --force-reinstall --no-cache-dir gpaw 2>&1 | tee "$GPAW_NEW_FILES_GO_TO_DEVANA"

# 5. Smoke-test
echo "--- 5/5: verifying new build ---"
"$PYENV/bin/gpaw" info | grep -E "MPI enabled|scalapack|Elpa|libxc|FFTW|BLAS|OpenMP"
echo
echo "Quick MPI smoke test (2 ranks, isolated H atom):"
mpiexec -n 2 "$PYENV/bin/gpaw" python -c "
from gpaw.mpi import world
from ase import Atoms
from gpaw import GPAW
print(f'rank {world.rank}/{world.size}: alive')
if world.rank == 0:
    a = Atoms('H', positions=[(0,0,0)], cell=[6,6,6], pbc=False)
    a.center()
    a.calc = GPAW(mode='lcao', basis='dzp', txt=None, kpts=(1,1,1))
    print('rank 0: E(H) =', a.get_potential_energy())
"

cat <<EOF

================================================================
GPAW rebuild complete.
Now run:
  bash scripts/run_node1_mpi.sh           # production with MPI ScaLAPACK
or for benchmarking:
  bash scripts/benchmark_node1_mpi.sh MoB_edge_B
================================================================
EOF
