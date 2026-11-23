#!/usr/bin/env bash
# ----------------------------------------------------------------
# run_fourier_pipeline.sh  —  the Fourier (angle) encoding search, end to end,
# for every dataset in core/config.py RUNS:
#
#   1. search_encoding.py     NSGA-II over encoding genomes (kernel-SVM fitness),
#                             held-out selection of up to three picks
#   2. train_downstream.py    picks and literature references on the test split:
#                             quantum kernels, trained classifiers, classical baselines
#
# Usage (from the repository root):
#   bash search_fourier/run_fourier_pipeline.sh            # real run
#   bash search_fourier/run_fourier_pipeline.sh --quick    # tiny settings, pipeline test
# Results: experiments_fourier/<dataset>/{features,search,downstream}/
# (quick runs: experiments_fourier_quick/)
# ----------------------------------------------------------------
set -euo pipefail
cd "$(dirname "$0")/.."                      # repository root

Q=""
[[ "${1:-}" == "--quick" ]] && Q="--quick"

python search_fourier/search_encoding.py $Q
python search_fourier/train_downstream.py $Q

echo
echo "searches : experiments_fourier*/<dataset>/search/"
echo "test set : experiments_fourier*/summary_downstream.csv"
