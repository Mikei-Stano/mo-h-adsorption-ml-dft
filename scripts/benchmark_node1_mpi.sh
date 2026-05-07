#!/usr/bin/env bash
# Benchmark MPI rank/thread split on a single structure.
# Usage:  bash scripts/benchmark_node1_mpi.sh <STRUCTURE_NAME>
set -euo pipefail

cd ~/Dokumenty/code/cemea-calculate/mo-h-adsorption-gpaw

STRUCTURE_NAME="${1:-MoB_example_structure}"
PYBIN="/home/mikei/.pyenv/versions/cemea-env/bin/python"
LOGDIR="logs/benchmark_$(date +%F_%H%M%S)"
mkdir -p "${LOGDIR}"

run_bench() {
    local ranks="$1"
    local threads="$2"
    local tag="r${ranks}t${threads}"
    local log="${LOGDIR}/${tag}.log"

    export OMP_NUM_THREADS="${threads}"
    export OPENBLAS_NUM_THREADS="${threads}"
    export MKL_NUM_THREADS="${threads}"
    export NUMEXPR_NUM_THREADS="${threads}"

    echo "============================================================"
    echo " Benchmark ${tag}: ${ranks} MPI ranks × ${threads} OMP threads"
    echo " structure: ${STRUCTURE_NAME}"
    echo " log: ${log}"
    echo "============================================================"

    /usr/bin/time -v mpiexec -n "${ranks}" gpaw python scripts/gpaw_h_adsorption.py \
        --stage screening \
        --mpi \
        --scalapack \
        --structure-name "${STRUCTURE_NAME}" \
        --basis sz \
        --kpts 2,2,1 \
        --relax-steps 3 \
        --fmax 0.15 \
        --site-search basic \
        --checkpoint-every-scf 5 \
        --max-hours-per-structure 0 \
        > "${log}" 2>&1 || echo "  (run ${tag} exited non-zero; see log)"
}

run_bench 8 4
run_bench 16 2
run_bench 32 1

echo "============================================================"
echo " Benchmarks finished. Logs in: ${LOGDIR}"
echo "============================================================"
