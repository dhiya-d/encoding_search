# Search robustness across seeds

## Amplitude encoder search (bloodmnist, centered)

Each seed's selected circuit, evaluated on the same test images; Δ = searched − default 31-CNOT circuit under the same classifier (paired bootstrap, 95% CI, * = excludes 0).

| seed | 2q gates | held-out retention | test fidelity | kernel bal. acc. / AUC | hea: bal. acc. (Δ vs default) | hea8: bal. acc. (Δ vs default) | hea8_probs: bal. acc. (Δ vs default) | qcnn: bal. acc. (Δ vs default) |
|---|---|---|---|---|---|---|---|---|
| 0 | 19 | 0.980 | 0.666 | 0.659 / 0.904 | 0.552 (+0.078 [+0.049, +0.106] *) | 0.574 (+0.054 [+0.024, +0.082] *) | 0.573 (+0.116 [+0.087, +0.143] *) | 0.496 (+0.049 [+0.021, +0.078] *) |
| 1 | 20 | 0.978 | 0.685 | 0.688 / 0.909 | 0.551 (+0.078 [+0.051, +0.104] *) | 0.614 (+0.094 [+0.066, +0.121] *) | 0.586 (+0.129 [+0.102, +0.157] *) | 0.534 (+0.088 [+0.056, +0.116] *) |
| 2 | 15 | 0.944 | 0.551 | 0.624 / 0.886 | 0.509 (+0.036 [+0.007, +0.064] *) | 0.552 (+0.032 [+0.003, +0.059] *) | 0.542 (+0.085 [+0.052, +0.117] *) | 0.494 (+0.047 [+0.016, +0.076] *) |

Across 3 seeds: 2q gates 18.0 ± 2.2; kernel balanced accuracy 0.657 ± 0.026.

| seed | entangler layers | share with step 2 | entanglers |
|---|---|---|---|
| 0 | 4 | 0.75 | CNOT_ring, CZ_ring |
| 1 | 5 | 1.00 | CZ_ring |
| 2 | 3 | 0.67 | CNOT_ring, CZ_ring |

**Searched circuit beats the default under most classifiers (none worse): 3/3 seeds.**

![](fig_multiseed_aae.png)

## Fourier encoder search (bloodmnist)

Each seed's search pick on the full test split; Δ = pick − the best literature reference (chosen on that seed's held-out batch), same classifier and training data (paired bootstrap, 95% CI, * = excludes 0).

| seed | pick | 2q cost | d | scales | held-out AUC (ref) | kernel bal. acc. | reupload: bal. acc. (Δ vs ref) | qcnn: bal. acc. (Δ vs ref) | hea8: bal. acc. (Δ vs ref) |
|---|---|---|---|---|---|---|---|---|---|
| 0 | search_best | 16 | 32 | 0.5, 1 | 0.937 (0.930 ref_two_layer_d16) | 0.788 | 0.687 (+0.052 [+0.041, +0.063] *) | 0.648 (+0.007 [-0.007, +0.021]) | 0.710 (+0.016 [+0.006, +0.026] *) |
| 1 | search_best | 12 | 16 | 0.5, 1 | 0.939 (0.933 ref_two_layer_d16) | 0.788 | 0.670 (+0.035 [+0.023, +0.046] *) | 0.625 (-0.016 [-0.028, -0.003] *) | 0.703 (+0.008 [-0.002, +0.018]) |
| 2 | search_best_knee_efficient | 0 | 32 | 0.5 | 0.933 (0.926 ref_two_layer_d16) | 0.807 | 0.672 (+0.037 [+0.025, +0.049] *) | 0.626 (-0.015 [-0.027, -0.002] *) | 0.725 (+0.031 [+0.020, +0.041] *) |

Pick circuits:

- seed 0: `d=32: RY(1x[0+q]) > CNOT(s1/t1/o0) > RY(0.5x[9+q]) > ZZ(0.5,s1/t2/o0) > RX(0.5x[0+q]) > H`
- seed 1: `d=16: RX(0.5x[5+q]) > CZ(s1/t2/o0) > RY(1x[13+q]) > CZ(s3/t2/o0) > CNOT(s2/t2/o0)`
- seed 2: `d=32: RZ(0.5x[6+q]) > RX(0.5x[10+q]) > H > RY(0.5x[0+q])`

**Search pick beats the best reference under most classifiers (none worse): 1/3 seeds.**

![](fig_multiseed_fourier.png)

## How to read this

- A search is reliable if every seed supports the claim, not just seed 0.
- If seeds pick different circuits that perform alike, report the common structure (gate count, entangler pattern, frequency scales) rather than one specific circuit.
- Confidence intervals reflect the test images only; seed-to-seed spread is the extra variation a reader should expect when rerunning the method.
