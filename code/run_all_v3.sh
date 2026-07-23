#!/bin/bash
# run_all_v3.sh — Nature-level experiment campaign
# 7 targets × 9 methods × 10 seeds = 630 runs × 500 iterations each
#
# Each run uses ~1 CPU core and takes ~40 minutes.
# With --parallel N, runs N experiments simultaneously.
#
# Usage:
#   bash code/run_all_v3.sh --parallel 20
#   bash code/run_all_v3.sh --parallel 20 --target 3V8D  # single target
#   bash code/run_all_v3.sh --parallel 20 --method curiosity  # single method
set -e

cd "$(dirname "$0")/.."  # cd to repo root

PYTHON="${PYTHON:-python3}"
DEFAULT_SCRIPT="code/run_experiment_v3.py"
NSGA2_SCRIPT="code/nsga2_experiment.py"
RESULTS="results"
ITERS=500
MAX_PARALLEL=20
FILTER_TARGET=""
FILTER_METHOD=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --parallel) MAX_PARALLEL=$2; shift 2;;
        --target) FILTER_TARGET=$2; shift 2;;
        --method) FILTER_METHOD=$2; shift 2;;
        --iterations) ITERS=$2; shift 2;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

TARGETS="3V8D 1ERE 3EML 1EVE 4DFR 3PJC 4MNE"
METHODS="random imgep_naive imgep curiosity bo ga mapelites novelty nsga2"
SEEDS="0 1 2 3 4 5 6 7 8 9"

[ -n "$FILTER_TARGET" ] && TARGETS="$FILTER_TARGET"
[ -n "$FILTER_METHOD" ] && METHODS="$FILTER_METHOD"

mkdir -p "$RESULTS"

TOTAL=0
SKIPPED=0
FAILED=0
RUNNING=0
PIDS=()
declare -A PID_TAG   # PID -> "target/method/seed" for failure reporting

run_one() {
    local target=$1 method=$2 seed=$3
    local outdir="${RESULTS}/${target}/${method}_seed${seed}"

    if [ -f "${outdir}/results.json" ]; then
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    mkdir -p "$outdir"
    TOTAL=$((TOTAL + 1))

    # Dispatch to the correct runner. NSGA-II uses a dedicated script because
    # it maintains a persistent population across iterations, unlike the
    # single-parent select-then-mutate methods in run_experiment_v3.py.
    local script
    if [ "$method" = "nsga2" ]; then
        script="$NSGA2_SCRIPT"
        args=(--target "$target" --seed "$seed" --iterations "$ITERS" --output_dir "$outdir")
    else
        script="$DEFAULT_SCRIPT"
        args=(--method "$method" --seed "$seed" --iterations "$ITERS" --target "$target" --output_dir "$outdir")
    fi

    echo "[$(date +%H:%M:%S)] START ${target}/${method}/s${seed} (#${TOTAL})"

    $PYTHON "$script" "${args[@]}" \
        > "${outdir}/stdout.log" 2>&1 &

    local pid=$!
    PIDS+=($pid)
    PID_TAG[$pid]="${target}/${method}/s${seed}"
    RUNNING=$((RUNNING + 1))

    # Wait if we hit the parallel limit
    while [ $RUNNING -ge $MAX_PARALLEL ]; do
        # Wait for any child to finish
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                local dead=${PIDS[$i]}
                if ! wait "$dead" 2>/dev/null; then
                    FAILED=$((FAILED + 1))
                    echo "[$(date +%H:%M:%S)] FAIL ${PID_TAG[$dead]}"
                fi
                unset 'PIDS[i]'
                unset 'PID_TAG[$dead]'
                RUNNING=$((RUNNING - 1))
            fi
        done
        # Compact array
        PIDS=("${PIDS[@]}")
        if [ $RUNNING -ge $MAX_PARALLEL ]; then
            sleep 5
        fi
    done
}

echo "============================================================"
echo "  Nature-level experiment campaign"
echo "  Targets: $(echo $TARGETS | wc -w)"
echo "  Methods: $(echo $METHODS | wc -w)"
echo "  Seeds: $(echo $SEEDS | wc -w)"
echo "  Iterations: $ITERS"
echo "  Max parallel: $MAX_PARALLEL"
echo "============================================================"

START_TIME=$(date +%s)

for target in $TARGETS; do
    for method in $METHODS; do
        for seed in $SEEDS; do
            run_one "$target" "$method" "$seed"
        done
    done
done

# Wait for remaining
echo "[$(date +%H:%M:%S)] Waiting for remaining ${RUNNING} jobs..."
for pid in "${PIDS[@]}"; do
    if ! wait "$pid" 2>/dev/null; then
        FAILED=$((FAILED + 1))
        echo "[$(date +%H:%M:%S)] FAIL ${PID_TAG[$pid]}"
    fi
done

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "============================================================"
echo "  DONE: ${TOTAL} runs launched, ${SKIPPED} skipped (already done), ${FAILED} failed"
echo "  Total wall time: $((ELAPSED / 3600))h $((ELAPSED % 3600 / 60))m"
echo "============================================================"

if [ $FAILED -gt 0 ]; then
    exit 1
fi
