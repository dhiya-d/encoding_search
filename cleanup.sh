#!/usr/bin/env bash
# ----------------------------------------------------------------
# encoding_search: repository hygiene after the Frobenius code landed.
#
# Nothing is deleted from your disk. Data and generated outputs are
# un-tracked with `git rm --cached`, so your local copies survive.
#
#   git switch -c cleanup
#   bash cleanup.sh
#   git commit -m "Untrack caches, datasets and generated output; add .gitignore"
# ----------------------------------------------------------------
set -euo pipefail
[ -d .git ] || { echo "error: run from the repository root"; exit 1; }

echo "==> 1. .gitignore"
if [ -f gitignore ] && [ ! -f .gitignore ]; then
  git mv gitignore .gitignore
fi
cat >> .gitignore <<'EOF'

# Frobenius suites: datasets and generated output are reproducible
frobenius_measure/*/__pycache__/
frobenius_measure/*/bloodmnist/
frobenius_measure/*/breastmnist/
frobenius_measure/*/breastmnist_original_split/
frobenius_measure/*/dermamnist/
frobenius_measure/*/retinamnist/
frobenius_measure/*/experiments/
frobenius_measure/*/encoded/
frobenius_measure/*/figures/
EOF
git add .gitignore

echo "==> 2. stop tracking byte-compiled caches"
git rm -r -q --cached --ignore-unmatch 'frobenius_measure/*/__pycache__/*'

echo "==> 3. stop tracking the MedMNIST copies (redownloadable: tools/prepare_dataset.py)"
for ds in bloodmnist breastmnist breastmnist_original_split dermamnist retinamnist; do
  git rm -r -q --cached --ignore-unmatch "frobenius_measure/*/$ds/*"
done

echo "==> 4. stop tracking generated output"
for g in experiments encoded figures; do
  git rm -r -q --cached --ignore-unmatch "frobenius_measure/*/$g/*"
done

echo "==> 5. remove the superseded scaffolding files"
# frobenius_measure/README.md now documents the search properly, and
# migrate.sh / run_all_patch.txt described work that is already done.
git rm -q --ignore-unmatch migrate.sh run_all_patch.txt search_frobenius_README.md

echo
echo "Tracked files: $(git ls-files | wc -l) (was 460)"
echo "Still to decide by hand:"
echo "  - README_thesis_map.md: merge into README.md, or link to it from there"
