# Frobenius-measure encoding + QCNN search (BreastMNIST)

Same two-stage pipeline as `../retina_frobenius`, run on BreastMNIST instead:
grayscale 28x28 images, binary labels (0=malignant, 1=benign), so there is no
ordinal grade to merge -- `merge` has a single "none" scheme, kept only for
interface parity with the RetinaMNIST config.

Stage 1's fitness function is label-blind, same as RetinaMNIST's: a genome's
fitness is its mean Frobenius reconstruction distance (how well Adam's
per-image angle fits reproduce the images) plus a gate penalty. Labels never
enter Stage 1 at all -- `evaluate.py` checks afterwards, as a diagnostic only,
whether the resulting encoding happened to preserve class-relevant signal.

## Pipeline order

```
Stage 0  resplit_data.py   ->  75/15/10 stratified train/val/test (one-off, done)
Stage 1  encoding search   (search.py, x5 seeds)  ->  best_genome.json
Stage 1  cost/signal curve (sweep_front.py, x5)   ->  cost_signal_curve.csv
Stage 1  aggregate         (aggregate_seeds.py)   ->  saturation point +/- sd
Stage 1  raw baseline      (evaluate.py)          ->  linear probe on raw targets
Stage 2a encode_states.py  ->  cached encoded states (this folder)
Stage 2b qcnn_search.py    ->  best_qcnn.json
Stage 2c qcnn_evaluate.py  ->  held-out test number
```

Run the whole thing with:

```bash
./run_pipeline.sh
```

or step through it manually -- see the docstring/usage example at the top of
each script (`search.py`, `sweep_front.py`, `aggregate_seeds.py`, `evaluate.py`,
`encode_states.py`, `qcnn_search.py`, `qcnn_evaluate.py`).

## What's different from RetinaMNIST

- **Grayscale, single channel.** `--channel` is accepted for interface parity
  but there is only one plane to select either way.
- **28x28 native, downsized to 16x16** (256 amplitudes -> 8 qubits), same as
  RetinaMNIST's `gray`/`green` settings.
- **Binary task, chance = 0.500.** No `retina4`/`retina2` split -- `merge`
  only has "none".
- **QCNN readout uses `--target-qubits 1`**, not the RetinaMNIST default of 2.
  One surviving qubit gives exactly 2 measurement outcomes, mapped directly
  onto the 2 classes -- no linear head needed.
- **Splits re-proportioned to 75/15/10** via `resplit_data.py` (the stock
  MedMNIST split is 546/78/156, i.e. 70/10/20). Pools all 780 images and
  re-splits stratified by class with a fixed seed; originals are preserved in
  `breastmnist_original_split/`. Current split: train 586 (158/428) / val 118
  (32/86) / test 76 (20/56). This exists because RetinaMNIST's 120-image
  validation set (smallest class n=12) produced a large winner's-curse gap
  between the QCNN search's validation-selected accuracy and the held-out test
  accuracy -- a bigger, better-balanced validation set should shrink that gap,
  but **only the `qcnn_evaluate.py` number should still be quoted**; it
  defaults to 5 independent training seeds reported as mean +/- sd.

## Notes and caveats

(carried over from the RetinaMNIST run; verify they still hold here)

- Single-seed results are not reportable -- always run and aggregate multiple
  seeds (`run_pipeline.sh` does 5 by default).
- The angle fit is non-convex; `--fit-steps` and `--restarts` both change the
  measured Frobenius distance for the *same* architecture. Keep them fixed
  when comparing genomes and report them with results.
- The QCNN simulator (`qsim_c.py`) is complex-valued, unlike the encoder's
  real-valued one (`qsim.py`), because the RZ/U3/CRZ components of the QCNN
  ansatze are otherwise inert.
- Training uses class-weighted loss and reports **balanced accuracy**;
  BreastMNIST is imbalanced (benign:malignant roughly 2.7:1 in train), and
  plain accuracy rewards a majority-class predictor.
- Exact simulation, no shot noise.

## Circuit diagrams

`search.py` and `qcnn_search.py` save a circuit image of the winning genome
automatically -- `best_circuit.png` next to `best_genome.json`, and
`best_qcnn_circuit.png` next to `best_qcnn.json`. Both failures are
non-fatal (`circuit image skipped: ...`) so a plotting hiccup never loses the
actual search result. The QCNN diagram draws discarded (pooled-away) wires as
grayed, terminated lines with an "x" where they leave the circuit, and marks
the surviving register with a measurement box.

To render any OTHER saved genome by hand (a specific finalist, a different
seed's winner, a figure at a chosen DPI, etc.), both drawers are also
standalone CLIs -- see the usage examples at the top of `draw_circuit.py` /
`draw_qcnn_circuit.py`.

## Crash-safe rendering

Every plotting/rendering call (`front.png`, `cost_signal_curve.png`,
`aggregate_curve.png`, both circuit images) runs through
`plot_safety.safe_render`, which does the render in a forked child process.
A plain `try/except` cannot catch a segfault -- it kills the process before
Python sees an exception -- and every stage here uses `set -euo pipefail`, so
an uncaught native-library crash (matplotlib/fontconfig under heavy load)
used to silently take down the rest of the pipeline after that point.
Isolating the render means a crash there only kills the forked child; the
parent prints a one-line skip message and keeps going. (This is exactly what
happened once in practice: `aggregate_seeds.py` segfaulted in matplotlib
after already saving its real output, silently killing Stage 2 for the rest
of that run.)

## Files

Identical to `../retina_frobenius` (minus `figures/`, which is just saved
output, not code):

| file | role |
|---|---|
| `config.py` | data/search config, gate pool, baseline table (BreastMNIST-specific) |
| `qsim.py` | real statevector simulator, per-sample batched angles |
| `genome.py` | encoding-architecture genome (Stage 1) |
| `encoder.py` | batched per-image AAE fitting -> Frobenius distances |
| `selection.py` | NSGA-II + knee, and penalised single-objective |
| `data_local.py` | offline `.npy` loader (BreastMNIST-specific) |
| `search.py` | Stage 1 CLI: encoding search, saves front + best genome + plot |
| `evaluate.py` | Stage 1 CLI: test reconstruction + class-signal probe |
| `sweep_front.py` | Stage 1 CLI: cost-vs-class-signal curve over a saved front |
| `aggregate_seeds.py` | Stage 1 CLI: pool curves across seeds, saturation point +/- sd |
| `qsim_c.py` | complex batched statevector simulator (shared parameters) |
| `ansatz.py` | Qcycle conv + Qmask pool gate library, wire patterns |
| `qcnn_genome.py` | 1-6 layer QCNN genome, repair to exactly REQUIRED_HALVINGS |
| `qcnn_model.py` | forward pass, training loop, balanced-accuracy scoring |
| `encode_states.py` | Stage 1->2 bridge: cache encoded states |
| `qcnn_search.py` | Stage 2b CLI: evolutionary search over QCNN architectures |
| `qcnn_evaluate.py` | Stage 2c CLI: final multi-seed held-out test evaluation |
| `run_pipeline.sh` | runs Stages 1-2c end to end |
| `resplit_data.py` | Stage 0 (one-off): re-splits breastmnist/ 75/15/10, stratified |
| `draw_circuit.py` | render a Stage 1 encoding genome to a circuit diagram (also standalone CLI) |
| `draw_qcnn_circuit.py` | render a Stage 2 QCNN genome to a circuit diagram (also standalone CLI) |
| `plot_safety.py` | subprocess-isolated rendering so a matplotlib crash can't kill the pipeline |
