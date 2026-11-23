# Fidelity, spectra, and classical simulability: data encodings for quantum image classification

Code for the paper *"Fidelity, spectra, and classical simulability: what matters in data
encodings for quantum image classification"*. It searches for 8-qubit data-encoding circuits
for MedMNIST images with four evolutionary (NSGA-II) searches, each trading a quality
measure against the number of two-qubit gates, and evaluates them with linear probes,
quantum kernels and trained quantum classifiers, with paired bootstrap statistics:

| Search | Encoding | Quality objective | Code |
|---|---|---|---|
| 1. Frobenius | approximate amplitude encoding, circuit fitted per image | mean Frobenius distance to the target state | `frobenius_measure/` (PyTorch) |
| 2. Fidelity | approximate amplitude encoding | mean fidelity | `search_amplitude/` with `--objective fidelity` |
| 3. Retention | approximate amplitude encoding | retained class signal (CV kernel AUC, encoded / exact states) | `search_amplitude/` (default) |
| 4. Fourier | angle encoding of colour DCT features | CV kernel AUC | `search_fourier/` |

Everything is simulated exactly in NumPy (state vectors, adjoint gradients, depolarizing
noise by trajectories, finite-shot estimators).

## Repository layout

```
run_all.py               reproduces every result of the paper, stage by stage
core/                    shared code: simulators, classifiers, data, settings
  config.py                settings of the Fourier pipeline and of all trained classifiers
  common.py                data loading, DCT features, metrics, kernels, SVM helpers
  qsim.py                  batched state-vector simulator with adjoint gradients
  aae_sim.py               amplitude-encoding circuits fitted per image (+ reference circuit)
  encoding_genome.py       Fourier-encoding genomes, frequency spectra, reference encodings
  classifiers.py           HEA, QCNN and re-uploading classifiers, training loop
  noise.py                 depolarizing noise (trajectories), mixed-state kernels, swap test
  diag_common.py           preprocessing modes and helpers of the amplitude pipeline
  repo.py                  makes all folders importable (see "Notes")
diagnostics/             amplitude-encoding diagnostics (paper Sec. V A)
  diag_a_phase_check.py    (a) global-sign artefact; also encodes the test split with a circuit
  diag_b_ceiling.py        (b) exact-state ceilings vs. classical references
  diag_c_spectral.py       (c) where the class signal sits; class-signal budget vs. fidelity
search_amplitude/        SEARCHES 2 and 3: amplitude-encoder search (fidelity or retention)
  search_config.py         its settings
  search_core.py           fitness (retention), genome operators, NSGA-II
  encoding_search.py       the search
  select_encoder.py        held-out selection of the final circuit
  train_on_encoded_states.py  trained classifiers on the encoded test states
  literature_ansatze.py    literature baselines (HEA chain, RealAmplitudes, ALT)
  run_amplitude_pipeline.sh   the whole amplitude chain for search_config.RUNS
search_fourier/          SEARCH 4: Fourier-encoding search
  search_encoding.py       features -> NSGA-II search -> held-out picks
  train_downstream.py      picks + references: kernels, trained classifiers, classical baselines
  nsga2.py                 NSGA-II sorting and selection
  run_fourier_pipeline.sh  search + downstream for config.RUNS
studies/                 experiments built on both searches
  multiseed.py             search seeds 1, 2: robustness of both searches
  objective_study.py       retention vs. fidelity objective vs. literature and random circuits
  noise_study.py           gate noise and finite measurement budgets
  product_kernel_check.py  entanglement-free encodings have an exact classical kernel
reporting/               reports, LaTeX tables and figures
  make_report.py           claims report with paired bootstrap tests (reports/report.md)
  frobenius_summary.py     table and figure of the Frobenius search (reads frobenius_measure/)
  paper_tables.py          LaTeX tables (reports/paper/tables.tex)
  paper_figures.py         vector figures of the paper
frobenius_measure/       SEARCH 1: Frobenius (reconstruction) search, one suite per dataset
  <ds>_frobenius/          search, front sweep (linear probe), aggregation, QCNN stage;
                           run_pipeline.sh runs it all (PyTorch, see its README)
  <ds>_frobenius_results/  results of five search seeds per dataset
tools/
  prepare_dataset.py       downloads a MedMNIST dataset into data/
  pennylane_encoding.py    rebuilds an exported encoding (and classifier) in PennyLane
tests/
  test_verification.py     simulators vs. PennyLane/Qiskit, gradients, noise, kernels
data/                    datasets (not in the repository, see data/README.md)
```

Each code folder has its own README with the scripts' inputs and outputs.

## Installation

Python ≥ 3.10.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pennylane qiskit pytest        # optional: verification tests, PennyLane export
python tests/test_verification.py          # about 5 s
```

## Reproducing the paper

Run every command **from the repository root**; all results are written relative to it.

```bash
python run_all.py --dry-run          # list all commands in order
python run_all.py --quick            # tiny settings in quick_run/: tests the installation
python run_all.py                    # everything (datasets are downloaded first)
python run_all.py --stages noise report tables figures   # selected stages
```

Every stage is resumable: finished work is detected and skipped, so after an interruption
simply rerun the same command. `--keep-going` continues after a failed command.
`--quick` runs the full chain with tiny settings in a separate folder `quick_run/`; its
numbers are meaningless. Rough runtimes on 8 cores: diagnostics 1 h; per main dataset
amplitude search 1 h, Fourier search 20 min, multiseed 1.5 h, objective study 3–4 h,
noise study 1 h.

| Stage | Script(s) | Paper |
|---|---|---|
| `data` | `tools/prepare_dataset.py` | Table II |
| `diagnostics` | `diagnostics/diag_{a,b,c}_*.py` on blood-, breast-, derma-, retinamnist | Sec. V A, Fig. 2 |
| `amplitude` | reference circuit, `encoding_search.py`, `select_encoder.py`, `diag_a_phase_check.py --genome-root`, `train_on_encoded_states.py` | Sec. IV C |
| `fourier` | `search_encoding.py`, `train_downstream.py` | Secs. IV D, V E, Fig. 5 |
| `multiseed` | `studies/multiseed.py` | Table VI, Fig. 6 |
| `objective` | `studies/objective_study.py` (fidelity search + comparison) | Sec. V C, Table IV, Fig. 4 |
| `noise` | `studies/noise_study.py` | Sec. V F, Table VII, Fig. 7 |
| `kernel` | `studies/product_kernel_check.py` | Prop. 4, Sec. V E |
| `report` | `reporting/make_report.py` | loss budget (Table V), claims |
| `frobenius` | `reporting/frobenius_summary.py` (reads `frobenius_measure/*_results`) | Sec. V B, Table III, Fig. 3 |
| `tables`, `figures` | `reporting/paper_tables.py`, `reporting/paper_figures.py` | all tables/figures |

The Frobenius search itself (Sec. IV B, Appendix C) runs per dataset with
`frobenius_measure/<ds>_frobenius/run_pipeline.sh` (PyTorch; search, front sweep,
aggregation, QCNN stage); `run_all.py` only summarizes its results.

Datasets: bloodmnist and pathmnist (centered preprocessing) run the full pipeline
(`--main`); breastmnist (uncentered) additionally runs the amplitude search
(`--amp-extra`); breast-, derma- and retinamnist additionally run the Fourier search
(`--fourier-extra`).

## Running one search by hand

```bash
# amplitude search (settings: search_amplitude/search_config.py)
bash search_amplitude/run_amplitude_pipeline.sh            # [--quick]

# Fourier search (settings: core/config.py)
bash search_fourier/run_fourier_pipeline.sh                # [--quick]
```

See `search_amplitude/README.md` and `search_fourier/README.md` for the individual steps.

## Where the results go

| Folder | Content |
|---|---|
| `experiments_handref/` | diagnostics (a), (c) and test-split states of the hand-designed 31-CNOT reference circuit |
| `experiments/` | diagnostic (b); amplitude search and selection (`<ds>/search/<mode>/`) |
| `experiments_searched/` | test-split states and trained classifiers of the selected amplitude circuit |
| `experiments_fourier/` | Fourier features, search, picks and downstream results (`<ds>/{features,search,downstream}/`) |
| `experiments*_seed<s>/` | the same for search seed s (multiseed and objective studies) |
| `experiments_fidsearch_seed<s>/` | fidelity-driven amplitude search (objective study) |
| `experiments_objective/`, `experiments_noise/` | per-circuit results of the studies |
| `reports/` | Markdown/JSON reports of every study, `report.md`, `claims.json`, `paper/tables.tex` |
| `paper_figures/<ds>/` | `fig1_pipeline.pdf` … `fig6_noise.pdf` |

## Notes

* **Imports.** The modules import each other by plain name (`import common`). Every
  entry-point script first imports `core/repo.py`, which puts all code folders on
  `sys.path`; scripts that launch other scripts locate them with `repo.script(name)`.
* **Reference circuit.** The hand-designed 31-CNOT reference circuit is `DEFAULT_GENOME` in
  `core/aae_sim.py`. Its outputs are in `experiments_handref/` (earlier versions of the code
  used `experiments_default_genome_old/`; rename the folder to reuse old results). In
  classifier result files it appears as source `"default"`.
* **Hard-coded figure values.** Figures 2 and 5 (`fig2_diagnostics.pdf`,
  `fig5_fourier.pdf`) use values copied from the bloodmnist reports (`SIGN`, `BUDGET`,
  `JOURNEY`, `FOURIER` at the top of `reporting/paper_figures.py`). Update them if you rerun.
* **Determinism.** All randomness is seeded (`SEED`, `DATA_SEED` in the configs, `--seed`
  on the searches). Results can still differ in the last digits between BLAS libraries.

## Citation

```bibtex
@article{TODO,
  title  = {Fidelity, spectra, and classical simulability: what matters in data encodings
            for quantum image classification},
  author = {TODO},
  year   = {TODO}
}
```

MedMNIST v2: J. Yang et al., *Sci. Data* **10**, 41 (2023).

## License

TODO: choose a license (e.g. MIT) before making the repository public.
