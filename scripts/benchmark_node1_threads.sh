#!/usr/bin/env bash
# Quick BLAS-threading benchmark on one structure.
# Times a single-point + 1-step relax for THREADS=1, 8, 16, 32.
# Goal: confirm threaded OpenBLAS scales before committing to production.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

STRUCT="${1:-MoB_edge_B}"
PYENV_ROOT="${PYENV_ROOT:-${HOME}/.pyenv}"
ENV_NAME="${ENV_NAME:-cemea-env}"
PYTHON="${PYTHON:-${PYENV_ROOT}/versions/${ENV_NAME}/bin/python}"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="logs/benchmark_threads_${TS}"
mkdir -p "$LOGDIR"

echo "Benchmarking structure: $STRUCT"
echo "Logs in: $LOGDIR"

for THREADS in 1 8 16 32; do
    echo "--- THREADS=$THREADS ---"
    LOG="$LOGDIR/threads_${THREADS}.log"
    /usr/bin/time -v -o "$LOGDIR/threads_${THREADS}.time" bash -c "
        export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
        '$PYTHON' scripts/gpaw_h_adsorption.py \
            --stage screening \
            --structure-name '$STRUCT' \
            --workers 1 \
            --cores-per-calc $THREADS \
            --relax-steps 1 \
            --fmax 0.5 \
            --max-hours-per-structure 0.5
    " > "$LOG" 2>&1 || echo "  (non-zero exit; see $LOG)"
    WALL=$(grep "Elapsed (wall clock) time" "$LOGDIR/threads_${THREADS}.time" | awk '{print $NF}')
    echo "  wall=$WALL"
done

echo "Summary:"
for f in "$LOGDIR"/*.time; do
    n=$(basename "$f" .time)
    w=$(grep "Elapsed (wall clock) time" "$f" | awk '{print $NF}')
    echo "  $n  $w"
done
