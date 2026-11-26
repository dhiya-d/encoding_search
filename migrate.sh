#!/usr/bin/env bash
# ----------------------------------------------------------------
# Reorganise dhiya-d/encoding_search for the thesis.
# Run from the repository root, in YOUR clone, on a branch.
#
#   git switch -c reorg
#   bash migrate.sh
#   git commit -m "Reorganise: promote Frobenius search, add thesis map"
#
# Everything is a git mv, so history is preserved.
# ----------------------------------------------------------------
set -euo pipefail

[ -d .git ] || { echo "error: run this from the repository root"; exit 1; }
[ -d frobenius_measure ] || { echo "error: frobenius_measure/ not found -- already migrated?"; exit 1; }

echo "==> creating the new tree"
mkdir -p search_frobenius
mkdir -p results/frobenius/{bloodmnist,breastmnist,dermamnist,retinamnist}

echo "==> moving Frobenius results out of the code tree"
# bloodmnist
git mv frobenius_measure/blood_frobenius_results/blo_aggregate results/frobenius/bloodmnist/aggregate
for s in 0 1 2 3 4; do
  git mv "frobenius_measure/blood_frobenius_results/blo_seed$s" "results/frobenius/bloodmnist/seed$s"
done
# breastmnist
git mv frobenius_measure/breast_frobenius_results/b_aggregate results/frobenius/breastmnist/aggregate
for s in 0 1 2 3 4; do
  git mv "frobenius_measure/breast_frobenius_results/b_seed$s" "results/frobenius/breastmnist/seed$s"
done
git mv frobenius_measure/breast_frobenius_results/qcnn_breast results/frobenius/breastmnist/qcnn
# dermamnist
git mv frobenius_measure/derma_frobenius_results/der_aggregate results/frobenius/dermamnist/aggregate
for s in 0 1 2 3 4; do
  git mv "frobenius_measure/derma_frobenius_results/der_seed$s" "results/frobenius/dermamnist/seed$s"
done
git mv frobenius_measure/derma_frobenius_results/qcnn_derma results/frobenius/dermamnist/qcnn
# retinamnist
git mv frobenius_measure/retina_frobenius_results/r_aggregate results/frobenius/retinamnist/aggregate
for s in 0 1 2 3 4; do
  git mv "frobenius_measure/retina_frobenius_results/r_seed$s" "results/frobenius/retinamnist/seed$s"
done
git mv frobenius_measure/retina_frobenius_results/qcnn_retina4 results/frobenius/retinamnist/qcnn

echo "==> removing the empty placeholder and the old directory"
git rm -q --ignore-unmatch frobenius_measure/c.py
rm -rf frobenius_measure

echo
echo "Done. Remaining by hand:"
echo "  1. put the Frobenius search script in search_frobenius/"
echo "     (or keep the stub README explaining it is not recoverable)"
echo "  2. add \"frobenius\" to STAGES in run_all.py"
echo "  3. paste the thesis map into README.md"
echo
git status --short | head -20
echo "  ... $(git status --porcelain | wc -l) paths changed"
