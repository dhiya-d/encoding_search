# Reorganising `encoding_search` for the thesis

## What a newcomer currently hits

| | count | share |
|---|---|---|
| Files tracked in `frobenius_measure/` | 182 | **81%** |
| Files in every code directory combined | 43 | 19% |

`frobenius_measure/` is 9.7 MB of the repository's 10 MB, contains no code
(`c.py` is 1 byte), uses four different seed-directory prefixes
(`blo_`, `b_`, `der_`, `r_`), and is never mentioned in `run_all.py`. It
sits at top level beside the real code directories.

Meanwhile the main pipeline commits **no** results at all — nothing from
`experiments*/` is tracked, and there is no `.gitignore` saying so. So one
study ships all of its outputs and the other three ship none, with nothing
explaining the difference.

The thesis presents the Frobenius search **first**, in §3.8.2, as the study
whose finding motivates everything after it. A reader who goes looking for
it finds a results dump.

## Recommendation: keep the layout, fix three things

Do **not** rename directories to thesis numbers (`s382_frobenius/`). The
functional layout is sound, `run_all.py` depends on it, every code folder
already has its own README, and thesis numbering will move. What is missing
is a map, not a different tree.

### 1. Add the thesis map to the README (highest value, costs nothing)

Paste the table below under a `## Where each part of the thesis lives`
heading, directly after the repository layout block.

### 2. Make Frobenius a first-class search

```
search_frobenius/            # new: a peer of search_amplitude/ and search_fourier/
  README.md                  #   inputs, outputs, how to run it
  frobenius_search.py        #   THE MISSING SCRIPT
  run_frobenius_pipeline.sh
results/
  frobenius/
    bloodmnist/{seed0..seed4,aggregate}/
    breastmnist/{seed0..seed4,aggregate,qcnn}/
    dermamnist/{seed0..seed4,aggregate,qcnn}/
    retinamnist/{seed0..seed4,aggregate,qcnn}/
```

Three changes in one: the code becomes present, the results move out of the
code tree, and the dataset and seed directories get one naming scheme
instead of four. Then add `"frobenius"` to `STAGES` in `run_all.py` so the
study is reproducible by the same command as everything else.

If the script genuinely cannot be recovered, say so in
`search_frobenius/README.md` in one line — an acknowledged gap is far
better than an empty `c.py`.

### 3. Add a `.gitignore` and state the results policy

```gitignore
# data and generated results
data/*
!data/README.md
experiments*/
quick_run/
reports/
paper_figures/

# python
__pycache__/
*.py[cod]
.venv/
```

Then add one line to the README: *"Generated results are not tracked;
`run_all.py` reproduces them. The exception is `results/frobenius/`, which
is committed because its search code predates this repository."* Or, if the
script is recovered, commit nothing and delete that sentence.

---

## Paste-ready: thesis map for the README

### Where each part of the thesis lives

Section numbers refer to Chapter 3, *Method*, §3.8 *Encoding search*.

| Thesis | What it describes | Code | Outputs |
|---|---|---|---|
| §3.8.1 The amplitude target | 16×16 grayscale → 256 amplitudes on 8 qubits; per-image fitting | `core/aae_sim.py`, `core/diag_common.py` | — |
| §3.8.2 Frobenius-based search | First search: minimise reconstruction residual vs CNOT count | `search_frobenius/` | `results/frobenius/` |
| §3.8.3 The global-sign artefact | Why a linear probe on raw amplitudes misleads, and why the kernel does not | `diagnostics/diag_a_phase_check.py` | `experiments_handref/` |
| §3.8.4 The shared amplitude search space | Genome, NSGA-II settings, selection rule | `search_amplitude/search_config.py`, `search_core.py` | — |
| §3.8.5 Fidelity-based search | Comparison arm: maximise mean fidelity | `search_amplitude/encoding_search.py --objective fidelity` | `experiments_fidsearch_seed<s>/` |
| §3.8.6 Retention-based search | The proposed objective: kernel AUC against the exact-state ceiling | `search_amplitude/encoding_search.py`, `select_encoder.py` | `experiments/<ds>/search/<mode>/` |
| — exact-state ceilings | The denominator retention is measured against | `diagnostics/diag_b_ceiling.py` | `experiments/` |
| — where the class signal sits | Spectral budget behind the ceiling | `diagnostics/diag_c_spectral.py` | `experiments_handref/` |
| §3.8.7 Is the objective what makes the difference? | Six-arm comparison with paired bootstrap tests | `studies/objective_study.py` | `experiments_objective/` |
| §3.8.8 Do the searched encodings help a trained classifier? | HEA and QCNN trained on the encoded test states | `search_amplitude/train_on_encoded_states.py` | `experiments_searched/` |
| §3.8.9 Fourier angles search | DCT features as rotation angles; NSGA-II on kernel AUC vs cost | `search_fourier/search_encoding.py`, `train_downstream.py` | `experiments_fourier/<ds>/` |
| §3.8.10 Limitations | Noise and shot budgets; the classical kernel of entanglement-free encodings | `studies/noise_study.py`, `studies/product_kernel_check.py` | `experiments_noise/`, `reports/` |
| Robustness across seeds | Both searches rerun under new seeds | `studies/multiseed.py` | `experiments*_seed<s>/` |
| Tables and figures | — | `reporting/` | `reports/`, `paper_figures/` |

### Reading order for a first visit

1. `README.md` — this file
2. `core/config.py` and `search_amplitude/search_config.py` — every setting
   quoted in the thesis is in one of these two files
3. `search_amplitude/search_core.py` — the retention fitness, which is the
   methodological contribution
4. `search_fourier/nsga2.py` — 70 lines, the whole selection mechanism
5. `run_all.py --dry-run` — every command in order, nothing executed

---

## Two smaller things

**`core/repo.py` surprises people.** Modules import each other by bare
name (`import common`) and work only because every entry point imports
`repo.py` first to put the code folders on `sys.path`. It is in the README
under *Notes*, but a newcomer opening `search_core.py` meets `import
common` long before reaching Notes. One comment line at the top of each
entry point would save that confusion.

**The README maps stages to paper sections, not thesis sections.** If the
repository is cited by the thesis, that table currently points at a
document the reader does not have. Keep it if the paper is going out, but
put the thesis table first.
