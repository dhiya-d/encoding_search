#!/usr/bin/env bash
# Resume run_pipeline.sh a second time: Stage 1 (5 seeds + aggregate) and
# Stage 2a (encode_states.py) are already done -- encoded/breast_seed0/ is
# fresh from the last run. Only Stage 2b/2c remain.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

ENCODE_SEED=0
RAW_BASELINE=0.7018   # from experiments/b_seed0/evaluate_stage1.log

echo "=== Stage 2b: QCNN architecture search (target-qubits=1 -> 2 outcomes) ==="
python qcnn_search.py --data "encoded/breast_seed${ENCODE_SEED}" \
    --target-qubits 1 --pop 12 --gen 10 --epochs 60 --lr 0.01 \
    --device auto --outdir experiments/qcnn_breast

echo "=== Stage 2c: held-out test evaluation ==="
python qcnn_evaluate.py --genome experiments/qcnn_breast/best_qcnn.json \
    --data "encoded/breast_seed${ENCODE_SEED}" --target-qubits 1 \
    --epochs 150 --seeds 5 --linear-baseline "$RAW_BASELINE"
