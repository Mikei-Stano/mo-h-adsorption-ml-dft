#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
PYBIN="${PYBIN:-${PYENV_ROOT}/versions/${ENV_NAME}/bin/python}"
RANKS="${RANKS:-16}"
MACHINE="${MACHINE:-node1}"
OMP="${OMP:-1}"

export OMP_NUM_THREADS="$OMP"
export OPENBLAS_NUM_THREADS="$OMP"
export MKL_NUM_THREADS="$OMP"
export NUMEXPR_NUM_THREADS="$OMP"

echo "============================================================"
echo " GPAW MPI Stage 1 (screening) — ${MACHINE}"
echo " ranks=${RANKS}  omp=${OMP}  basis=sz  xc=LDA  kpts=2,2,1  site=basic"
echo " checkpoint every 5 SCF; per-structure timeout disabled"
echo " started: $(date -Is)"
echo "============================================================"

mpiexec -n "$RANKS" "$PYBIN" scripts/gpaw_h_adsorption.py \
    --machine "$MACHINE" \
    --stage screening \
    --mpi \
    --scalapack \
    --basis sz \
    --kpts 2,2,1 \
    --relax-steps 3 \
    --fmax 0.15 \
    --site-search basic \
    --checkpoint-every-scf 5 \
    --max-hours-per-structure 0

echo "============================================================"
echo " DONE: $(date -Is)"
echo "============================================================"
