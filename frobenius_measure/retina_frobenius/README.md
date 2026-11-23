# QCNN architecture search (Stage 2)

Evolutionary search over **QCNN architectures**, run on the fixed encoding
chosen by the Frobenius encoding search (Stage 1). The register reduces
**8 → 2 qubits**, and the two surviving qubits give exactly **4 measurement
outcomes**, mapped one-to-one onto the 4 classes of the `retina4` grouping
(RetinaMNIST grades `{0}, {1}, {2}, {3,4}`).

## Pipeline order

```
Stage 1  encoding search   →  best_genome.json          (retina_frobenius/)
Stage 2a encode_states.py  →  cached encoded states     (this folder)
Stage 2b qcnn_search.py    →  best_qcnn.json
Stage 2c qcnn_evaluate.py  →  held-out test number
```

**Stage 2a is not optional.** Re-fitting per-image encoder angles inside every
QCNN fitness evaluation would be enormously wasteful *and* confounding: the QCNN
search must see a fixed input representation so that accuracy differences are
attributable to the QCNN architecture alone. `encode_states.py` encodes every
image once and caches the states to disk.

## Search space

A genome is **1 to 6 layers**. Each layer is a convolution followed by 0, 1 or 2
pooling halvings, with a repair step enforcing that the halvings total exactly
two — so the register always reduces `8 → 4 → 2` regardless of layer count. A
one-layer genome performs both halvings in that layer; a six-layer genome spreads
them out and uses the rest as convolution-only stages. Every layer must perform
some convolution work, which rules out the degenerate architectures seen
previously (near-empty conv layers delegating everything to pooling).

**Convolution ansatze (Qcycle)** — parameter counts per wire pair:

| gate | params | gate | params |
|---|---:|---|---:|
| `U_TTN` | 2 | `U_14` | 6 |
| `U_5` | 10 | `U_15` | 4 |
| `U_6` | 10 | `U_SO4` | 6 |
| `U_9` | 2 | `U_SU4` | 15 |
| `U_13` | 6 | `U_gate` | 9 |

**Wire patterns:** `linear` (disjoint pairs), `ring` (cyclic), `brick`
(alternating offsets).

**Pooling ansatze (Qmask):** `pool_U` (6 params, controlled-U3 on both branches),
`pool_ansatz` (2 params, CRZ on the |1> branch and CRX on the |0> branch).

## Selection modes

- **single-objective (default)** — `fitness = bal_acc − l1·gates − l2·params`,
  with `l1 = 5e-4` and `l2 = 1e-3`. Breaks accuracy ties toward simpler circuits.
- **`--pareto-front`** — NSGA-II on (balanced accuracy ↑, two-qubit gates ↓),
  mirroring the encoding search.

## Usage

Run the whole retina4 pipeline (Stage 1 x5 seeds through Stage 2c) end to end:

```bash
./run_pipeline.sh
```

`run_pipeline.sh` uses the seed-0 front by default -- for a different encoding
seed, run the relevant stage manually, following the steps below. `retina2`
and `none` are not available in this project at all (see `config.py`).

**2a — cache the encoded states** (once per encoder × grouping):

```bash
python encode_states.py \
    --frob-dir ../retina_frobenius \
    --genome ../retina_frobenius/experiments/r_seed0/best_genome.json \
    --data-dir ../retina_frobenius/retinamnist \
    --merge retina4 --fit-steps 400 --restarts 3 \
    --out encoded/retina4_seed0
```

**2b — search:**

```bash
python qcnn_search.py --data encoded/retina4_seed0 \
    --pop 12 --gen 10 --epochs 40 --lr 0.003 \
    --device auto --outdir experiments/qcnn_retina4
```

**2c — held-out test number** (the one that goes in the thesis):

```bash
python qcnn_evaluate.py --genome experiments/qcnn_retina4/best_qcnn.json \
    --data encoded/retina4_seed0 --epochs 150 --seeds 5 \
    --linear-baseline 0.4557
```

## What the QCNN has to beat

For `retina4` at 16×16 greyscale:

| reference | value |
|---|---|
| chance | 0.250 |
| linear probe on **raw amplitude targets** | 0.4557 |
| linear probe on **encoded states** | printed by `qcnn_evaluate.py` |

The second row is the number that justifies the quantum pipeline at all; the
third is the sharper test, since it uses *exactly* the same input the QCNN sees.
If the QCNN cannot beat a logistic regression on its own input states, the
convolution/pooling stack is not contributing.

## Notes and caveats

- **Validation selection bias is real and was measured at ~0.10** on comparably
  sized splits earlier in this project. `qcnn_search.py` selects on validation,
  so its reported number is optimistic by construction. Only the
  `qcnn_evaluate.py` test figure should be quoted, and it defaults to 5
  independent training seeds reported as mean ± sd.
- The simulator is **complex** (unlike the encoder's real-valued one), because
  the RZ/U3/CRZ components of these ansatze are otherwise inert.
- Training uses class-weighted loss throughout and reports **balanced accuracy**;
  RetinaMNIST grades are heavily imbalanced and plain accuracy rewards a
  majority-class predictor.
- Exact simulation, no shot noise.

## Files

| file | role |
|---|---|
| `qsim_c.py` | complex batched statevector simulator (shared parameters) |
| `ansatz.py` | Qcycle conv + Qmask pool gate library, wire patterns |
| `qcnn_genome.py` | 1–6 layer genome, repair to exactly two halvings (8→4→2) |
| `qcnn_model.py` | forward pass, training loop, balanced-accuracy scoring |
| `encode_states.py` | Stage 1→2 bridge: cache encoded states |
| `qcnn_search.py` | evolutionary search CLI (single-objective or Pareto) |
| `qcnn_evaluate.py` | final multi-seed test evaluation with baselines |
| `draw_circuit.py` | render a Stage 1 encoding genome to a circuit diagram (also standalone CLI) |
| `draw_qcnn_circuit.py` | render a Stage 2 QCNN genome to a circuit diagram (also standalone CLI) |
| `plot_safety.py` | subprocess-isolated rendering so a matplotlib crash can't kill the pipeline |

## Circuit diagrams

`search.py` and `qcnn_search.py` now save a circuit image of the winning
genome automatically -- `best_circuit.png` next to `best_genome.json`, and
`best_qcnn_circuit.png` next to `best_qcnn.json`. Both failures are
non-fatal (`circuit image skipped: ...`) so a plotting hiccup never loses the
actual search result. The QCNN diagram draws discarded (pooled-away) wires as
grayed, terminated lines with an "x" where they leave the circuit, and marks
the surviving register with a measurement box.

## Crash-safe rendering

Every plotting/rendering call (`front.png`, `cost_signal_curve.png`,
`aggregate_curve.png`, both circuit images) runs through
`plot_safety.safe_render`, which does the render in a forked child process.
A plain `try/except` cannot catch a segfault -- it kills the process before
Python sees an exception -- and every stage here uses `set -euo pipefail`, so
an uncaught native-library crash (matplotlib/fontconfig under heavy load, seen
once in practice) used to silently take down the rest of the pipeline after
that point. Isolating the render means a crash there only kills the forked
child; the parent prints a one-line skip message and keeps going.

To render any OTHER saved genome by hand (a specific finalist, a different
seed's winner, a thesis figure at a chosen DPI, etc.), both drawers are also
standalone CLIs -- see the usage examples at the top of `draw_circuit.py` /
`draw_qcnn_circuit.py`.
