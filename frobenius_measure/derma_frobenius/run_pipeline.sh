#!/usr/bin/env bash
# Full Frobenius -> QCNN pipeline for DermaMNIST (7 classes, chance=1/7=0.143).
#
# The 5 encoding-search seeds (and their 5 sweeps) are INDEPENDENT of each
# other, so both stages run all 5 concurrently rather than one at a time.
# Each worker's BLAS/OpenMP thread count is capped to (cores / 5) so 5
# concurrent processes don't oversubscribe the machine the way one
# unconstrained process already can -- see THREADS_PER_JOB below.
#
# 7 classes don't divide the 8-qubit register as a clean power of two, so
# --target-qubits 4 (16 outcomes) is used -- 9 outcomes go unused, but this
# keeps the same 16x16/8-qubit resolution as retina_frobenius/breast_frobenius
# rather than downsampling further just to make the outcome count line up.
#
# Stage 1  (search.py, x5 seeds, parallel)      encoding architecture search
# Stage 1  (sweep_front.py, x5, parallel)        cost vs class-signal curve
# Stage 1  (aggregate_seeds.py)                  pool the 5 curves
# Stage 1  (evaluate.py)                         raw-amplitude linear baseline
# Stage 2a (encode_states.py)                    cache encoded states
# Stage 2b (qcnn_search.py)                      QCNN architecture search
# Stage 2c (qcnn_evaluate.py)                    held-out test number
#
# Usage: ./run_pipeline.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR=dermamnist
MERGE=none
SEEDS=(0 1 2 3 4)
ENCODE_SEED=0   # which seed's front feeds Stage 2

# Prefer the scheduler's own allocation over nproc: on a shared node (e.g. a
# PBS "smp" queue job that got ncpus=24 on a bigger physical node), nproc can
# report the WHOLE node's core count rather than what was actually granted,
# which would oversubscribe badly -- exactly the failure mode that made local
# runs crawl earlier. PBS Pro sets NCPUS (and usually PBS_NCPUS) to the real
# per-job allocation; fall back to nproc only outside a scheduler.
NPROC=${PBS_NCPUS:-${NCPUS:-$(nproc)}}
THREADS_PER_JOB=$(( NPROC / ${#SEEDS[@]} ))
[ "$THREADS_PER_JOB" -lt 1 ] && THREADS_PER_JOB=1
export OMP_NUM_THREADS=$THREADS_PER_JOB
export MKL_NUM_THREADS=$THREADS_PER_JOB
export OPENBLAS_NUM_THREADS=$THREADS_PER_JOB
export NUMEXPR_NUM_THREADS=$THREADS_PER_JOB
export VECLIB_MAXIMUM_THREADS=$THREADS_PER_JOB
echo "[parallel] ${#SEEDS[@]} concurrent jobs, ${THREADS_PER_JOB} threads/job (of ${NPROC} cores)"

# matplotlib builds its font cache on first import; if the 5 parallel workers
# below all hit that at once (their first plot call), it's a race. Point the
# cache at a job-local directory (also avoids $HOME quota issues on shared
# clusters) and warm it once, single-threaded, before fanning out.
export MPLCONFIGDIR="$(pwd)/.mplcache"
mkdir -p "$MPLCONFIGDIR"
python -c "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot" 2>/dev/null || true

run_parallel_seeds() {
    # $1 = human label for logging; remaining args form the command template,
    # with __SEED__ substituted per seed. Logs to experiments/der_seed<N>/<label>.log
    local label="$1"; shift
    local pids=() fails=0 s cmd
    for s in "${SEEDS[@]}"; do
        mkdir -p "experiments/der_seed${s}"
        cmd=("$@")
        cmd=("${cmd[@]//__SEED__/$s}")
        "${cmd[@]}" > "experiments/der_seed${s}/${label}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do
        wait "$pid" || fails=$((fails + 1))
    done
    if [ "$fails" -gt 0 ]; then
        echo "ERROR: ${fails} ${label} job(s) failed -- check experiments/der_seed*/${label}.log"
        exit 1
    fi
}

echo "=== Stage 1: encoding search (5 seeds, parallel) ==="
run_parallel_seeds search python search.py --data-dir "$DATA_DIR" --merge "$MERGE" --pareto-front \
    --pop 24 --gen 12 --fit-steps 150 --device auto \
    --seed __SEED__ --outdir experiments/der_seed__SEED__

echo "=== Stage 1: cost vs class-signal curve (5 seeds, parallel) ==="
run_parallel_seeds sweep python sweep_front.py \
    --result experiments/der_seed__SEED__/search_result.json \
    --data-dir "$DATA_DIR" --merge "$MERGE" --fit-steps 300 --restarts 2 \
    --seed __SEED__ --device auto --outdir experiments/der_seed__SEED__

echo "=== Stage 1: aggregate across seeds ==="
python aggregate_seeds.py --dirs 'experiments/der_seed*' --tolerance 0.03 \
    --outdir experiments/der_aggregate

echo "=== Stage 1: raw-amplitude linear baseline (for Stage 2c) ==="
EVAL_LOG="experiments/der_seed${ENCODE_SEED}/evaluate_stage1.log"
python evaluate.py --genome "experiments/der_seed${ENCODE_SEED}/best_genome.json" \
    --data-dir "$DATA_DIR" --merge "$MERGE" --size 16 --fit-steps 400 --restarts 3 \
    | tee "$EVAL_LOG"
RAW_BASELINE=$(grep "on raw targets" "$EVAL_LOG" | awk '{print $NF}')
echo "raw-amplitude linear baseline = ${RAW_BASELINE}"

echo "=== Stage 2a: cache encoded states (seed ${ENCODE_SEED} front) ==="
python encode_states.py --frob-dir . \
    --genome "experiments/der_seed${ENCODE_SEED}/best_genome.json" \
    --data-dir "$DATA_DIR" --merge "$MERGE" --fit-steps 400 --restarts 3 \
    --out "encoded/derma_seed${ENCODE_SEED}"

echo "=== Stage 2b: QCNN architecture search (target-qubits=4 -> 16 outcomes, 7 used) ==="
# --use-head: 7 classes don't fill all 16 measurement outcomes, so a small
# linear head maps the 16-wide readout onto exactly 7 classes -- otherwise
# nll_loss treats outcomes 7-15 as always-wrong predictable classes, which
# just wastes representational capacity and adds prediction noise early in
# training. See qcnn_search.py's own printed warning for this exact case.
python qcnn_search.py --data "encoded/derma_seed${ENCODE_SEED}" \
    --target-qubits 4 --use-head --pop 12 --gen 10 --epochs 60 --lr 0.01 \
    --device auto --outdir experiments/qcnn_derma

echo "=== Stage 2c: held-out test evaluation ==="
python qcnn_evaluate.py --genome experiments/qcnn_derma/best_qcnn.json \
    --data "encoded/derma_seed${ENCODE_SEED}" --target-qubits 4 --use-head \
    --epochs 150 --seeds 5 --linear-baseline "$RAW_BASELINE"
