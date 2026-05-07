#!/usr/bin/env bash
# Production launcher for node1 (i9-14900KF, 32 CPUs).
#
# NOTE: this build of GPAW has MPI/ScaLAPACK/ELPA disabled (see `gpaw info`).
# The bottleneck for large MoB slabs is the dense diagonalization, which is
# delegated to numpy/scipy → bundled OpenBLAS 0.3.29 (MAX_THREADS=64).
# We therefore run ONE Python process and let OpenBLAS use all 32 cores.
#
# When GPAW is rebuilt with MPI + ScaLAPACK, switch to scripts/run_node1_mpi.sh.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
PYTHON="${PYTHON:-${PYENV_ROOT}/versions/${ENV_NAME}/bin/python}"
THREADS="${THREADS:-32}"

export OMP_NUM_THREADS="$THREADS"
export OPENBLAS_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export NUMEXPR_NUM_THREADS="$THREADS"
# Avoid thread-pinning conflicts on hybrid P/E core CPUs.
export OMP_PROC_BIND="${OMP_PROC_BIND:-false}"
export OMP_PLACES="${OMP_PLACES:-cores}"

echo "================================================================"
echo "Node1 production launcher (threaded BLAS, $THREADS threads)"
echo "Started: $(date)"
echo "Python : $PYTHON"
echo "Repo   : $REPO_ROOT"
echo "================================================================"

"$PYTHON" scripts/gpaw_h_adsorption.py \
    --machine node1 \
    --stage screening \
    --workers 1 \
    --cores-per-calc "$THREADS" \
    --basis sz \
    --kpts 2,2,1 \
    --relax-steps 3 \
    --fmax 0.15 \
    --site-search basic \
    --checkpoint-every-scf 5 \
    --max-hours-per-structure 0

echo "================================================================"
echo "Finished: $(date)"
echo "================================================================"
