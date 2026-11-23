# frobenius_measure/ — Search 1: Frobenius (reconstruction) search

Approximate amplitude encoding with a label-blind reconstruction objective
(paper Sec. IV B, results Sec. V B, Appendix C). One self-contained PyTorch
suite per dataset; the four copies run the same algorithm and differ only in
dataset defaults (data folder, class merge, QCNN readout width):

| Suite | Data | Classes | Notes |
|---|---|---|---|
| `blood_frobenius/` | bloodmnist | 8 | QCNN stage not run; train/test images not in the repository |
| `breast_frobenius/` | breastmnist | 2 | stratified 75/15/10 resplit (`resplit_data.py`, originals in `breastmnist_original_split/`) |
| `derma_frobenius/` | dermamnist | 7 | QCNN readout: 4 qubits + linear head |
| `retina_frobenius/` | retinamnist | 4 | grades {0},{1},{2},{3,4} merged (`retina4`) |

Each suite's `run_pipeline.sh` runs, for search seeds 0–4:

1. `search.py` — NSGA-II on (mean Frobenius distance ↓, two-qubit gates ↓), per-image
   angles fitted with Adam; knee = cheapest circuit within 0.015 of the best distance.
   (`--no-pareto-front` runs the scalarized variant: distance + 0.004 × gates.)
2. `sweep_front.py` — refits every front circuit on train/test images and compares a
   linear probe on the encoded states with the probe on the exact targets.
3. `aggregate_seeds.py` — saturation point and binned curves across seeds.
4. `evaluate.py`, `encode_states.py`, `qcnn_search.py`, `qcnn_evaluate.py` — QCNN stage on
   the seed-0 encoder (pipe `qcnn_evaluate.py` into `qcnn_evaluate.log` in the result folder,
   so that `reporting/frobenius_summary.py` picks up the test numbers).

Results are copied to `<ds>_frobenius_results/`. The paper's table and figure are made by

```bash
python reporting/frobenius_summary.py      # from the repository root
```
