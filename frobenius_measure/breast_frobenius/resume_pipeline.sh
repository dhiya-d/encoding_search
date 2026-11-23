#!/usr/bin/env bash
# Resume run_pipeline.sh from where it crashed: Stage 1's 5 seeds + sweeps
# are already done (experiments/b_seed0..4/), so this picks up at
# aggregate_seeds.py (now fixed -- see plot_safety.py) and runs through
# Stage 2c. Safe to re-run if it fails partway again.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

DATA_DIR=breastmnist
MERGE=none
ENCODE_SEED=0

echo "=== Stage 1: aggregate across seeds ==="
python aggregate_seeds.py --dirs 'experiments/b_seed*' --tolerance 0.03 \
    --outdir experiments/b_aggregate

echo "=== Stage 1: raw-amplitude linear baseline (for Stage 2c) ==="
EVAL_LOG="experiments/b_seed${ENCODE_SEED}/evaluate_stage1.log"
python evaluate.py --genome "experiments/b_seed${ENCODE_SEED}/best_genome.json" \
    --data-dir "$DATA_DIR" --merge "$MERGE" --size 16 --fit-steps 400 --restarts 3 \
    | tee "$EVAL_LOG"
RAW_BASELINE=$(grep "on raw targets" "$EVAL_LOG" | awk '{print $NF}')
echo "raw-amplitude linear baseline = ${RAW_BASELINE}"

echo "=== Stage 2a: cache encoded states (seed ${ENCODE_SEED} front) ==="
python encode_states.py --frob-dir . \
    --genome "experiments/b_seed${ENCODE_SEED}/best_genome.json" \
    --data-dir "$DATA_DIR" --merge "$MERGE" --fit-steps 400 --restarts 3 \
    --out "encoded/breast_seed${ENCODE_SEED}"

echo "=== Stage 2b: QCNN architecture search (target-qubits=1 -> 2 outcomes) ==="
python qcnn_search.py --data "encoded/breast_seed${ENCODE_SEED}" \
    --target-qubits 1 --pop 12 --gen 10 --epochs 60 --lr 0.01 \
    --device auto --outdir experiments/qcnn_breast

echo "=== Stage 2c: held-out test evaluation ==="
python qcnn_evaluate.py --genome experiments/qcnn_breast/best_qcnn.json \
    --data "encoded/breast_seed${ENCODE_SEED}" --target-qubits 1 \
    --epochs 150 --seeds 5 --linear-baseline "$RAW_BASELINE"
