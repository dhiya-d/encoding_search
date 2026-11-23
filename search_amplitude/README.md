# search_amplitude/ — Search 1: amplitude-encoder search

Approximate amplitude encoding: every 16×16 grayscale image (256 values, normalised)
is prepared by its own fitted RY/CNOT circuit. The *structure* of that circuit is searched
with NSGA-II over two objectives:

1. **retention** (maximised): cross-validated AUC of a fidelity-kernel SVM on the fitted
   states of a fixed batch of 400 training images, divided by the same AUC on the exact
   states (the ceiling). Mean fidelity is logged but not optimised.
2. **number of two-qubit gates** (minimised).

The final front is re-evaluated on a disjoint held-out batch with a larger fitting budget,
and the cheapest circuit within 0.01 of the best retention is selected.

## Steps (from the repository root)

```bash
python search_amplitude/encoding_search.py --dataset bloodmnist --mode centered
python search_amplitude/select_encoder.py  --dataset bloodmnist --mode centered
# encode the test split with the selected circuit (and the reference circuit, for comparison)
python diagnostics/diag_a_phase_check.py --genome-root experiments --out experiments_searched \
       --datasets bloodmnist --modes centered
python diagnostics/diag_a_phase_check.py --out experiments_handref --datasets bloodmnist --modes centered
python search_amplitude/train_on_encoded_states.py --runs bloodmnist:centered
```

or all runs in `search_config.RUNS` at once: `bash search_amplitude/run_amplitude_pipeline.sh`.
Add `--quick` to any search command for a fast pipeline test; `--seed s --out-root <dir>`
reruns the search with another seed; `--objective fidelity` optimises mean fidelity
instead of retention (objective study).

## Files

| File | Role |
|---|---|
| `search_config.py` | all settings (batch size, fitting budget, search space, NSGA-II, selection) |
| `search_core.py` | batches, exact-state ceiling, parallel fitness evaluation, genome operators, NSGA-II |
| `encoding_search.py` | the search (resumes from `state.json`) |
| `select_encoder.py` | held-out re-evaluation of the final front and selection |
| `train_on_encoded_states.py` | HEA, HEA-8/12, HEA-8 (probabilities) and QCNN trained on the encoded test-split states |
| `literature_ansatze.py` | literature baselines at matched CNOT counts (objective study) |
| `run_amplitude_pipeline.sh` | all steps for every run in `search_config.RUNS` |

## Outputs

* `experiments/<ds>/search/<mode>/`: `evaluations.csv`, `history.csv`, `front_final.json`,
  `selection.json`, `selected_genome.json`, plots.
* `experiments_searched/<ds>/a_phase_check/<mode>/`: test-split states of the selected circuit.
* `experiments_searched/<ds>/downstream_classifiers/<mode>/`: one JSON per (state source,
  classifier) with per-seed metrics and test probabilities.

Genome format (shared with the circuit code in `core/aae_sim.py`): a list alternating
rotation and entangler layers, e.g.
`[{"kind": "rot", "gate": "RY"}, {"kind": "ent", "gate": "CNOT_ring", "stride": 1, "step": 1, "offset": 0}, ...]`.
