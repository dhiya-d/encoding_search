# Search robustness across seeds

## Amplitude encoder search (pathmnist, centered)

Each seed's selected circuit, evaluated on the same test images; Δ = searched − hand-designed 31-CNOT reference circuit under the same classifier (paired bootstrap, 95% CI, * = excludes 0).

| seed | 2q gates | held-out retention | test fidelity | kernel bal. acc. / AUC | hea: bal. acc. (Δ vs hand-designed ref.) | hea8: bal. acc. (Δ vs hand-designed ref.) | hea8_probs: bal. acc. (Δ vs hand-designed ref.) | qcnn: bal. acc. (Δ vs hand-designed ref.) |
|---|---|---|---|---|---|---|---|---|
| 0 | 15 | 0.953 | 0.627 | 0.289 / 0.652 | 0.234 (-0.008 [-0.028, +0.012]) | 0.217 (-0.014 [-0.034, +0.006]) | 0.253 (+0.026 [+0.003, +0.050] *) | 0.258 (-0.062 [-0.083, -0.040] *) |
| 1 | 15 | 0.990 | 0.641 | 0.258 / 0.637 | 0.286 (+0.043 [+0.018, +0.067] *) | 0.234 (+0.003 [-0.018, +0.022]) | 0.224 (-0.002 [-0.026, +0.020]) | 0.231 (-0.088 [-0.112, -0.065] *) |
| 2 | 12 | 0.981 | 0.623 | 0.275 / 0.659 | 0.251 (+0.009 [-0.017, +0.033]) | 0.250 (+0.019 [-0.000, +0.038]) | 0.246 (+0.020 [-0.001, +0.042]) | 0.309 (-0.010 [-0.035, +0.016]) |

Across 3 seeds: 2q gates 14.0 ± 1.4; kernel balanced accuracy 0.274 ± 0.013.

| seed | entangler layers | share with step 2 | entanglers |
|---|---|---|---|
| 0 | 3 | 0.67 | CNOT_ring |
| 1 | 3 | 0.67 | CNOT_ring, CZ_ring |
| 2 | 2 | 0.50 | CZ_ring |

**Searched circuit beats the hand-designed reference under most classifiers (none worse): 0/3 seeds.**

![](fig_multiseed_aae_pathmnist.png)

## Fourier encoder search (pathmnist)

Each seed's search pick on the full test split; Δ = pick − the best literature reference (chosen on that seed's held-out batch), same classifier and training data (paired bootstrap, 95% CI, * = excludes 0).

| seed | pick | 2q cost | d | scales | held-out AUC (ref) | kernel bal. acc. | reupload: bal. acc. (Δ vs ref) | qcnn: bal. acc. (Δ vs ref) | hea8: bal. acc. (Δ vs ref) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | search_best | 4 | 4 | 0.5 | 0.891 (0.885 ref_parallel_d4) | 0.542 | 0.481 (+0.029 [+0.022, +0.037] *) | 0.477 (-0.002 [-0.010, +0.006]) | 0.516 (+0.024 [+0.016, +0.033] *) |
| 1 | search_best_efficient | 4 | 4 | 0.5 | 0.897 (0.895 ref_parallel_d4) | 0.559 | 0.499 (+0.048 [+0.041, +0.054] *) | 0.331 (-0.148 [-0.158, -0.139] *) | 0.488 (-0.003 [-0.010, +0.004]) |
| 2 | search_best | 18 | 4 | 0.5, 1 | 0.893 (0.891 ref_parallel_d4) | 0.509 | 0.492 (+0.040 [+0.034, +0.047] *) | 0.494 (+0.015 [+0.009, +0.021] *) | 0.528 (+0.037 [+0.031, +0.042] *) |

Pick circuits:

- seed 0: `d=4: RY(0.5x[0+q]) > H > CZ(s2/t2/o1) > RY(0.5x[2+q]) > H > RY(0.5x[0+q])`
- seed 1: `d=4: H > CZ(s2/t2/o0) > RZ(0.5x[3+q]) > RX(0.5x[0+q])`
- seed 2: `d=4: RY(1x[3+q]) > CNOT(s4/t2/o0) > ZZ(0.5,s3/t1/o1)`

**Search pick beats the best reference under most classifiers (none worse): 2/3 seeds.**

![](fig_multiseed_fourier_pathmnist.png)

## How to read this

- A search is reliable if every seed supports the claim, not just seed 0.
- If seeds pick different circuits that perform alike, report the common structure (gate count, entangler pattern, frequency scales) rather than one specific circuit.
- Confidence intervals reflect the test images only; seed-to-seed spread is the extra variation a reader should expect when rerunning the method.
