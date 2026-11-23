# Frobenius-measure encoding + QCNN search (BloodMNIST)

Same two-stage pipeline as `../retina_frobenius` / `../breast_frobenius`, run
on BloodMNIST: RGB 28x28 blood-cell images, 8 classes, reasonably balanced
(largest class ~2.7x the smallest -- much better balanced than DermaMNIST's
~59x). No ordinal grade to merge -- `merge` has a single "none" scheme.

Stage 1's fitness function is label-blind, same as the other projects: a
genome's fitness is its mean Frobenius reconstruction distance plus a gate
penalty. Labels never enter Stage 1 -- `evaluate.py` checks afterwards, as a
diagnostic only, whether the resulting encoding happened to preserve
class-relevant signal.

## Readout: 8 classes into 16 outcomes

8 classes = 2**3, a clean power of two -- but 3 doesn't divide the 8-qubit
register evenly, so no `--target-qubits` value gives an exact 8-outcome fit
(that would need a 6-qubit register instead, i.e. more downsampled images).
This project uses `--target-qubits 4` (16 outcomes) to keep the same
16x16/8-qubit resolution as retina_frobenius/breast_frobenius.

Stage 2 uses **`--use-head`**: a small linear layer maps the 16-wide readout
onto exactly 8 classes. Without it, the direct (no-head) mapping treats all
16 outcomes as class indices via `nll_loss` -- the 8 that never appear in the
data just waste representational capacity and can produce "impossible"
predictions early in training. `qcnn_search.py` itself prints a warning if
this flag is skipped.

## `channel` = gray, not rgb

Kept at `gray` (8 qubits) to match retina_frobenius/breast_frobenius's
register size and simulation cost. `--channel rgb` is a one-flag change if
colour (e.g. staining) turns out to matter a lot here, at the cost of a
bigger register (768 amplitudes -> 10 qubits instead of 8).

## Validation split

BloodMNIST's stock split is train 11959 / val 1712 / test 3421, and it's
*well* balanced (smallest val class n=122) -- much better than
RetinaMNIST's (n=12) or DermaMNIST's (n=12). The winner's-curse gap between
the QCNN search's validation-selected accuracy and the held-out test accuracy
should plausibly be smaller here than in either of those, but still check it
with `qcnn_evaluate.py` rather than assuming.

## Pipeline order

```
Stage 1  encoding search   (search.py, x5 seeds, parallel)  ->  best_genome.json
Stage 1  cost/signal curve (sweep_front.py, x5, parallel)   ->  cost_signal_curve.csv
Stage 1  aggregate         (aggregate_seeds.py)             ->  saturation point +/- sd
Stage 1  raw baseline      (evaluate.py)                    ->  linear probe on raw targets
Stage 2a encode_states.py  ->  cached encoded states (this folder)
Stage 2b qcnn_search.py    ->  best_qcnn.json
Stage 2c qcnn_evaluate.py  ->  held-out test number
```

Run the whole thing with:

```bash
./run_pipeline.sh
```

The 5 Stage-1 seeds (and their 5 sweeps) run concurrently, each capped to
`nproc / 5` BLAS/OpenMP threads so the parallel jobs don't oversubscribe the
machine -- see the comment block at the top of `run_pipeline.sh`.

## Notes and caveats

(carried over from the other projects; verify they still hold here)

- Single-seed results are not reportable -- always run and aggregate multiple
  seeds (`run_pipeline.sh` does 5 by default).
- The angle fit is non-convex; `--fit-steps` and `--restarts` both change the
  measured Frobenius distance for the *same* architecture. Keep them fixed
  when comparing genomes and report them with results.
- The QCNN simulator (`qsim_c.py`) is complex-valued, unlike the encoder's
  real-valued one (`qsim.py`).
- Training uses class-weighted loss and reports **balanced accuracy**.
- Exact simulation, no shot noise.
- All plotting/rendering calls (`front.png`, `cost_signal_curve.png`,
  `aggregate_curve.png`, circuit images) run through `plot_safety.safe_render`,
  which isolates each render in a forked subprocess -- a native-library crash
  (matplotlib/fontconfig segfault, seen once under heavy machine load) can't
  take down the rest of the pipeline the way a plain try/except can't catch it.

## Files

Identical to `../breast_frobenius`, plus `resplit_data.py` is NOT included
here (no resplit has been done for BloodMNIST):

| file | role |
|---|---|
| `config.py` | data/search config, gate pool, baseline table (BloodMNIST-specific) |
| `qsim.py` | real statevector simulator, per-sample batched angles |
| `genome.py` | encoding-architecture genome (Stage 1) |
| `encoder.py` | batched per-image AAE fitting -> Frobenius distances |
| `selection.py` | NSGA-II + knee, and penalised single-objective |
| `data_local.py` | offline `.npy` loader (BloodMNIST-specific) |
| `search.py` | Stage 1 CLI: encoding search, saves front + best genome + plot + circuit image |
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
| `draw_circuit.py` | render a Stage 1 encoding genome to a circuit diagram (also standalone CLI) |
| `draw_qcnn_circuit.py` | render a Stage 2 QCNN genome to a circuit diagram (also standalone CLI) |
| `plot_safety.py` | subprocess-isolated rendering so a matplotlib crash can't kill the pipeline |
| `run_pipeline.sh` | runs Stages 1-2c end to end, parallelized across seeds |
