# studies/ — experiments built on both searches

Run after the main searches (seed 0) of the dataset have finished; every study resumes
and only computes what is missing. Reports go to `reports/`.

| Script | What it does | Paper |
|---|---|---|
| `multiseed.py` | reruns both searches with seeds 1 and 2 (whole chain), compares the picks across seeds with paired bootstrap tests | Table IV, Fig. 5 |
| `objective_study.py` | retention-driven vs. fidelity-driven amplitude search, the literature ansätze (best chosen on held-out data within the CNOT budget, plus a high-budget circuit), random circuits in the same CNOT range and the reference circuit, all evaluated identically | Table II, Fig. 3 |
| `noise_study.py` | depolarizing gate noise (trajectories, p = 0 … 0.02), finite-shot swap-test kernels and finite-shot classifier readout for the amplitude and Fourier encoders | Table V, Fig. 6 |
| `product_kernel_check.py` | for entanglement-free Fourier encodings, computes the kernel classically (product of single-qubit overlaps), checks it against the state vector and retrains the SVM with it | Sec. III D |

```bash
python studies/multiseed.py        --aae-run bloodmnist:centered --fourier-dataset bloodmnist --seeds 1 2
python studies/objective_study.py  --run bloodmnist:centered --seeds 0 1 2 --jobs 2
python studies/noise_study.py      --run bloodmnist:centered --fourier-dataset bloodmnist
python studies/product_kernel_check.py --dataset bloodmnist --seeds 0 1 2
```

`--report-only` (multiseed, noise) or `--stage report` (objective) rebuilds the report from
existing results. `--quick` uses tiny settings for a pipeline test.
