# search_fourier/ — Search 4: Fourier (angle) encoding search

Images are reduced to low-frequency 2-D DCT coefficients per colour channel, ranked by
Fisher score on the training split and scaled to rotation angles. A genome is a list of
layers acting on 8 qubits:

| Layer | Effect |
|---|---|
| `enc` | rotation R_axis(scale · x_f) on every qubit q, feature f = (start + q) mod d |
| `zz` | ZZ rotations with angle scale · x_f · x_g on the edges of a ring |
| `ent` | fixed CNOT/CZ ring (stride, step, offset) |
| `H` | Hadamard on every qubit |

The scales of the rotations fix the frequency spectrum of the model (Schuld, Sweke &
Meyer 2021). NSGA-II maximises the cross-validated AUC of a fidelity-kernel SVM on a
training batch and minimises the two-qubit cost; the final front and the literature
reference encodings are re-evaluated on a disjoint held-out batch and up to three picks
are made: *best* (highest held-out AUC), *knee* (cheapest within `SELECT_TOLERANCE` of
the best) and *efficient* (cheapest that beats every reference).

## Steps (from the repository root)

```bash
python search_fourier/search_encoding.py  --dataset bloodmnist
python search_fourier/train_downstream.py --dataset bloodmnist
```

or `bash search_fourier/run_fourier_pipeline.sh` for every dataset in `core/config.py`
(`RUNS`). `--quick` gives a fast pipeline test; `--seed s --out-root <dir>` reruns with
another seed.

`train_downstream.py` evaluates the picks and the references on the test split with
fidelity and projected quantum kernels, the re-uploading classifier, QCNN and HEA
classifiers (several seeds), and classical baselines: an RBF SVM on the same features
and a classical model with exactly the same frequency spectrum.

## Files

| File | Role |
|---|---|
| `search_encoding.py` | features, search (resumes from `state.json`), held-out selection, picks |
| `train_downstream.py` | downstream evaluation of picks and references |
| `nsga2.py` | non-dominated sorting, crowding distance, survivor selection |
| `run_fourier_pipeline.sh` | both steps for every dataset |

Settings are in `core/config.py` (shared with the classifiers). The genome operators and
spectrum calculations are in `core/encoding_genome.py`.

## Outputs

`experiments_fourier/<ds>/`

* `features/`: feature matrices of every split and `feature_meta.json`
* `search/`: `evaluations.csv`, `history.csv`, `front.json`, `selection.csv`, `picks.json`, plots
* `downstream/`: one JSON per encoding (all models, per-seed test probabilities),
  `encodings/<name>.json` (standalone export, see `tools/pennylane_encoding.py`),
  `summary.csv`, `results.png`
