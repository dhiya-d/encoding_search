#!/usr/bin/env bash
# ----------------------------------------------------------------
# run_amplitude_pipeline.sh  —  the amplitude-encoder search, end to end,
# for every (dataset, mode) listed in search_amplitude/search_config.py RUNS:
#
#   1. encoding_search.py      retention-driven NSGA-II search (Pareto front)
#   2. select_encoder.py       re-score the front on held-out images, pick a circuit
#   3. diag_a_phase_check.py   encode the TEST split with the picked circuit
#                              (and with the hand-designed reference circuit)
#   4. diag_c_spectral.py      class-signal budget of the picked circuit
#   5. train_on_encoded_states.py   trained classifiers on the encoded states
#
# Usage (from the repository root):
#   bash search_amplitude/run_amplitude_pipeline.sh            # real run
#   bash search_amplitude/run_amplitude_pipeline.sh --quick    # tiny settings, pipeline test
# Results: experiments/<dataset>/search/<mode>/ (search and selection),
#          experiments_searched/ (test-split encodings, classifiers),
#          experiments_handref/ (reference circuit on the test split).
# Quick runs write to experiments_quick*/ instead.
# ----------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."                      # repository root

QUICK=""
SEARCH_OUT="experiments"
REPORT_OUT="experiments_searched"
HANDREF_OUT="experiments_handref"
DIAG_A_ARGS=""
CLF_ARGS=""
if [[ "${1:-}" == "--quick" ]]; then
    QUICK="--quick"
    SEARCH_OUT="experiments_quick"
    REPORT_OUT="experiments_quick_searched"
    HANDREF_OUT="experiments_quick_handref"
    DIAG_A_ARGS="--n-train 200 --n-test 100 --steps 60 --restarts 1"
    CLF_ARGS="--epochs 3"
fi

# 1-2: search and selection for all runs in search_config.RUNS
python search_amplitude/encoding_search.py $QUICK
python search_amplitude/select_encoder.py $QUICK

# the runs, as dataset:mode
RUNS=$(python -c "import sys; sys.path[:0] = ['core', 'search_amplitude']; import search_config as S; print(' '.join(f'{d}:{m}' for d, m in S.RUNS))")

for run in $RUNS; do
    ds="${run%%:*}"; mode="${run##*:}"
    # 3: encode the test split with the selected circuit
    python diagnostics/diag_a_phase_check.py --genome-root "$SEARCH_OUT" --out "$REPORT_OUT" \
        --datasets "$ds" --modes "$mode" $DIAG_A_ARGS
    # 3b: same test images with the reference circuit (skipped if already done)
    python diagnostics/diag_a_phase_check.py --out "$HANDREF_OUT" \
        --datasets "$ds" --modes "$mode" $DIAG_A_ARGS
    # 4: where the class signal goes for this circuit
    python diagnostics/diag_c_spectral.py --genome-root "$SEARCH_OUT" --out "$REPORT_OUT" \
        --datasets "$ds" --modes "$mode"
    # 5: trained classifiers on exact, searched and reference states
    python search_amplitude/train_on_encoded_states.py --runs "$run" --states-root "$REPORT_OUT" \
        --compare-root "$HANDREF_OUT" $CLF_ARGS
done

echo
echo "search results : $SEARCH_OUT/<dataset>/search/<mode>/"
echo "test-set report: $REPORT_OUT/summary_a_phase_check.csv"
