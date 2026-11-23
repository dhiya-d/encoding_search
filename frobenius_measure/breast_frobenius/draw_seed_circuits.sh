#!/usr/bin/env bash
# Render each Stage 1 seed's winning encoding circuit to an image, so they can
# be compared side by side for structural motifs the search converges on
# repeatedly across independent seeds (as opposed to one-off quirks of a
# single run). Backfills seeds that finished before search.py auto-saved
# best_circuit.png itself; safe to re-run any time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

for d in experiments/b_seed*; do
    [ -f "$d/best_genome.json" ] || continue
    echo "=== $d ==="
    python draw_circuit.py --genome "$d/best_genome.json" --out "$d/best_circuit.png"
done
