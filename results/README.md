# Results reported in the paper

This folder is a snapshot of the results behind the paper's tables and figures. Its
subfolders are named after the code folders that produced the files.

The scripts never read from or write to `results/`: a rerun writes to `reports/` and
`experiments*/`, which `.gitignore` excludes. After a rerun, compare the new files with
these, and copy them here once the paper has been updated. The right-hand column
gives each file's path in a rerun.

The Frobenius search (search 1) is the exception. Its raw results stay in
`frobenius_measure/<ds>_frobenius_results/`, and only its summary is stored here.

## Files

| File | Paper | Script → path in a rerun |
|---|---|---|
| **Diagnostics** (amplitude encoding, hand-designed 31-CNOT reference circuit) | | |
| `diagnostics/summary_a_phase_check.csv` | Sec. V A, Fig. 2: sign artefact, six readouts (4 datasets × 3 preprocessing modes) | `diag_a_phase_check.py` → `experiments_handref/summary_a_phase_check.csv` |
| `diagnostics/summary_b_ceiling.csv` | Sec. V A, Fig. 2: exact-state ceilings, classical references, kernel concentration | `diag_b_ceiling.py` → `experiments/summary_b_ceiling.csv` |
| `diagnostics/summary_c_spectral.csv` | Sec. V A, Fig. 2: shared / between-class / within-class split, compressibility | `diag_c_spectral.py` → `experiments_handref/summary_c_spectral.csv` |
| `diagnostics/lowpass_check.log` | Sec. V A: low-pass sentence (between-class share 0.254 → 0.322 at radius 3) | one-off `snippet.py`, **not in the repository** |
| **Search 1: Frobenius** | | |
| `frobenius_measure/frobenius_summary.{md,json}` | Sec. V B, Table III | `reporting/frobenius_summary.py` → `reports/frobenius_summary.{md,json}` |
| `frobenius_measure/frobenius_table.tex` | Table III | → `reports/paper/frobenius_table.tex` |
| `frobenius_measure/fig7_frobenius.pdf` | Fig. 3 (the file name follows the paper's `\includegraphics`) | → `paper_figures/fig7_frobenius.pdf` |
| **Searches 2 and 3: fidelity and retention** (amplitude) | | |
| `search_amplitude/selection_bloodmnist_centered.csv` | Sec. V C: held-out re-scoring of the retention front (pick: 19 CNOTs, retention 0.980) | `select_encoder.py` → `experiments/bloodmnist/search/centered/selection.csv` |
| `search_amplitude/summary_a_phase_check.csv` | Table V: selected circuits on the test split (bloodmnist fidelity 0.666, breastmnist 0.953) | `diag_a_phase_check.py --genome-root experiments` → `experiments_searched/summary_a_phase_check.csv` |
| `search_amplitude/train_on_encoded_states.log` | Table V: trained classifiers on exact, searched and reference states (e.g. hea8_probs 0.637 / 0.573 / 0.457) | console output of `train_on_encoded_states.py`; a rerun also writes `experiments_searched/summary_downstream_classifiers.csv` |
| `search_amplitude/diag_a_phase_check_searched.log` | console output of the selected circuits' test-split run | — |
| **Search 4: Fourier** | | |
| `search_fourier/run_fourier_pipeline.log` | Sec. V E, Fig. 5: picks on four datasets; bloodmnist downstream (kernel 0.788, re-uploading 0.687, classical RBF 0.813) | console output of `search_encoding.py` + `train_downstream.py`; structured results → `experiments_fourier/<ds>/{search,downstream}/` |
| **Studies and report** | | |
| `studies/objective_bloodmnist.md` | Sec. V C, Table IV, Fig. 4 | `objective_study.py` → `reports/objective_bloodmnist.md` |
| `studies/multiseed_bloodmnist.md` | Table VI, Fig. 6 | `multiseed.py` → `reports/multiseed_bloodmnist.md` |
| `studies/noise_bloodmnist.md` | Sec. V F, Table VII, Fig. 7 | `noise_study.py` → `reports/noise_bloodmnist.md` |
| `studies/{objective,multiseed,noise}_pathmnist.md` | Sec. V G (second dataset) | same scripts → `reports/*_pathmnist.md` |
| `studies/product_kernel_bloodmnist.md` | Prop. 4, Sec. V E | `product_kernel_check.py` → `reports/product_kernel_bloodmnist.md` |
| `report.md` | Table V (section "Where the class information goes"), Fourier sections of Sec. V E, claims | `make_report.py` → `reports/report.md` |
| `tests/test_verification.log` | Appendix B | `python tests/test_verification.py` (console) |

## Caveats

- **Markdown only.** The study reports here are Markdown. `paper_tables.py` and
  `paper_figures.py` read the matching `reports/*.json` files. If the JSON files from the
  original runs still exist (`~/fourier/reports/*.json`), copy them next to the `.md`
  files. Otherwise the figures are redrawn when the studies are rerun.
- **Older code version.** `multiseed_bloodmnist.md` and the two `search_amplitude`
  logs come from an earlier version of the code, which called the hand-designed
  reference circuit `default`. The numbers are unaffected.
- **Product kernel, `identical: NO`.** This flag on one row of
  `product_kernel_bloodmnist.md` comes from the old check, which compared the chosen SVM
  constant C. The metrics on that row are identical. The current script compares the
  metrics, so it reports `yes`.
- **No classifier CSV.** The `summary_downstream_classifiers.csv` in the chat predates the
  retrained hea8_probs classifier: it gives 0.566 instead of 0.637 on exact bloodmnist
  states. For that reason the log is stored here instead of the CSV.
- **Fourier log is incomplete.** It was pasted only up to the start of the breastmnist
  downstream run. The downstream results of breastmnist, dermamnist and retinamnist are
  in `report.md`.
- **Frobenius summary gaps.**
  - B is `n/a` for bloodmnist because its train/test arrays are not in the repository.
  - QCNN test accuracies are `pending` until `qcnn_evaluate.py` has been rerun.
