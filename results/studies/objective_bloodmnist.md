# Objective study: retention vs fidelity, literature and random circuits (bloodmnist, centered)

All circuits are amplitude encoders from the same search space, evaluated on the same test images with the same classifiers. The retention and fidelity searches differ only in the objective.

## Circuits compared

| circuit | type | CNOTs | held-out retention | test fidelity | kernel AUC | hea8 bal. acc. | hea8_probs bal. acc. | qcnn bal. acc. |
|---|---|---|---|---|---|---|---|---|
| retention_s2 | retention search (ours) | 15 | 0.944 | 0.551 | 0.886 | 0.552 | 0.542 | 0.494 |
| retention_s0 | retention search (ours) | 19 | 0.980 | 0.666 | 0.904 | 0.574 | 0.573 | 0.496 |
| retention_s1 | retention search (ours) | 20 | 0.978 | 0.685 | 0.909 | 0.614 | 0.586 | 0.534 |
| fidelity_matched_s2 | fidelity search, matched CNOTs | 15 | — | 0.610 | 0.892 | 0.570 | 0.542 | 0.504 |
| fidelity_matched_s0 | fidelity search, matched CNOTs | 19 | — | 0.657 | 0.905 | 0.594 | 0.585 | 0.531 |
| fidelity_matched_s1 | fidelity search, matched CNOTs | 20 | — | 0.692 | 0.905 | 0.559 | 0.551 | 0.474 |
| fidelity_best_s1 | fidelity search, own selection | 35 | — | 0.827 | 0.916 | 0.623 | 0.571 | 0.541 |
| fidelity_best_s2 | fidelity search, own selection | 38 | — | 0.847 | 0.918 | 0.610 | 0.603 | 0.553 |
| fidelity_best_s0 | fidelity search, own selection | 41 | — | 0.829 | 0.917 | 0.612 | 0.587 | 0.566 |
| lit_alt_2 | literature ansatz | 14 | 0.922 | 0.484 | 0.876 | 0.542 | 0.528 | 0.481 |
| lit_hea_ry_2 | literature ansatz | 14 | 0.921 | 0.532 | 0.874 | 0.509 | 0.509 | 0.492 |
| lit_realamp_2 | literature ansatz | 14 | 0.937 | 0.531 | 0.867 | 0.523 | 0.510 | 0.481 |
| lit_hea_ry_3 | literature ansatz | 21 | 0.977 | 0.625 | 0.895 | 0.586 | 0.575 | 0.530 |
| lit_realamp_3 | literature ansatz | 21 | 0.972 | 0.625 | 0.897 | 0.579 | 0.535 | 0.517 |
| lit_alt_3 | literature ansatz | 22 | 0.953 | 0.620 | 0.896 | 0.570 | 0.560 | 0.515 |
| lit_alt_4 | literature ansatz | 28 | 0.970 | 0.705 | 0.900 | 0.584 | 0.568 | 0.523 |
| lit_hea_ry_4 | literature ansatz | 28 | 0.966 | 0.690 | 0.905 | 0.599 | 0.581 | 0.522 |
| lit_hea_ry_8 | literature ansatz | 56 | 0.997 | 0.822 | 0.915 | 0.625 | 0.581 | 0.561 |
| handref | hand-designed 31-CNOT reference | 31 | — | 0.578 | 0.875 | 0.520 | 0.457 | 0.447 |

Best literature ansatz within the CNOT budget of the searched circuits (≤ 23), chosen on the held-out batch, never on test: **lit_hea_ry_3** (21 CNOTs). Deeper literature circuits (lit_alt_4, lit_hea_ry_4, lit_hea_ry_8) are reported as high-budget references.

Kernel AUC of the exact states (ceiling): 0.927.

## Paired comparisons (retention pick − other circuit)

Δ balanced accuracy, paired bootstrap on the same test images, 95% CI, * = excludes 0.

| comparison | hea8 | hea8_probs | qcnn |
|---|---|---|---|
| seed 0: retention (19) − fidelity search, matched CNOTs (19) | -0.021 [-0.046, +0.004] | -0.012 [-0.037, +0.011] | -0.034 [-0.058, -0.010] * |
| seed 0: retention (19) − fidelity search, own selection (41) | -0.038 [-0.062, -0.013] * | -0.014 [-0.041, +0.011] | -0.070 [-0.094, -0.043] * |
| seed 0: retention (19) − best literature ansatz within budget (lit_hea_ry_3) (21) | -0.013 [-0.039, +0.014] | -0.002 [-0.028, +0.023] | -0.034 [-0.060, -0.006] * |
| seed 0: retention (19) − matched-cost literature (lit_hea_ry_3) (21) | -0.013 [-0.039, +0.014] | -0.002 [-0.028, +0.023] | -0.034 [-0.060, -0.006] * |
| seed 0: retention (19) − matched-cost literature (lit_realamp_3) (21) | -0.005 [-0.030, +0.020] | +0.037 [+0.010, +0.064] * | -0.021 [-0.044, +0.004] |
| seed 0: retention (19) − matched-cost literature (lit_alt_3) (22) | +0.004 [-0.021, +0.029] | +0.012 [-0.014, +0.038] | -0.019 [-0.044, +0.006] |
| seed 0: retention (19) − high-budget literature (lit_alt_4) (28) | -0.010 [-0.037, +0.016] | +0.004 [-0.023, +0.031] | -0.027 [-0.052, -0.003] * |
| seed 0: retention (19) − high-budget literature (lit_hea_ry_4) (28) | -0.025 [-0.052, +0.000] | -0.009 [-0.032, +0.015] | -0.026 [-0.048, -0.004] * |
| seed 0: retention (19) − high-budget literature (lit_hea_ry_8) (56) | -0.051 [-0.076, -0.027] * | -0.009 [-0.034, +0.018] | -0.064 [-0.090, -0.038] * |
| seed 0: retention (19) − hand-designed 31-CNOT reference (31) | +0.054 [+0.024, +0.082] * | +0.116 [+0.087, +0.143] * | +0.049 [+0.021, +0.078] * |
| seed 1: retention (20) − fidelity search, matched CNOTs (20) | +0.054 [+0.029, +0.079] * | +0.036 [+0.010, +0.062] * | +0.061 [+0.035, +0.090] * |
| seed 1: retention (20) − fidelity search, own selection (35) | -0.009 [-0.034, +0.015] | +0.015 [-0.011, +0.042] | -0.006 [-0.032, +0.020] |
| seed 1: retention (20) − best literature ansatz within budget (lit_hea_ry_3) (21) | +0.027 [+0.003, +0.053] * | +0.011 [-0.015, +0.038] | +0.004 [-0.020, +0.030] |
| seed 1: retention (20) − matched-cost literature (lit_hea_ry_3) (21) | +0.027 [+0.003, +0.053] * | +0.011 [-0.015, +0.038] | +0.004 [-0.020, +0.030] |
| seed 1: retention (20) − matched-cost literature (lit_realamp_3) (21) | +0.035 [+0.011, +0.059] * | +0.051 [+0.023, +0.078] * | +0.018 [-0.008, +0.043] |
| seed 1: retention (20) − matched-cost literature (lit_alt_3) (22) | +0.044 [+0.019, +0.069] * | +0.026 [-0.002, +0.054] | +0.019 [-0.008, +0.046] |
| seed 1: retention (20) − high-budget literature (lit_alt_4) (28) | +0.030 [+0.005, +0.054] * | +0.018 [-0.008, +0.044] | +0.011 [-0.014, +0.038] |
| seed 1: retention (20) − high-budget literature (lit_hea_ry_4) (28) | +0.015 [-0.010, +0.042] | +0.005 [-0.020, +0.031] | +0.012 [-0.012, +0.036] |
| seed 1: retention (20) − high-budget literature (lit_hea_ry_8) (56) | -0.011 [-0.034, +0.012] | +0.005 [-0.021, +0.031] | -0.026 [-0.053, -0.000] * |
| seed 1: retention (20) − hand-designed 31-CNOT reference (31) | +0.094 [+0.066, +0.121] * | +0.129 [+0.102, +0.157] * | +0.088 [+0.056, +0.116] * |
| seed 2: retention (15) − fidelity search, matched CNOTs (15) | -0.018 [-0.045, +0.009] | +0.000 [-0.031, +0.030] | -0.010 [-0.039, +0.019] |
| seed 2: retention (15) − fidelity search, own selection (38) | -0.058 [-0.083, -0.031] * | -0.061 [-0.089, -0.033] * | -0.059 [-0.086, -0.031] * |
| seed 2: retention (15) − best literature ansatz within budget (lit_hea_ry_3) (21) | -0.034 [-0.061, -0.008] * | -0.033 [-0.063, -0.004] * | -0.036 [-0.064, -0.008] * |
| seed 2: retention (15) − matched-cost literature (lit_hea_ry_2) (14) | +0.042 [+0.014, +0.071] * | +0.032 [+0.001, +0.064] * | +0.003 [-0.027, +0.030] |
| seed 2: retention (15) − matched-cost literature (lit_realamp_2) (14) | +0.029 [+0.003, +0.056] * | +0.032 [+0.002, +0.063] * | +0.013 [-0.017, +0.043] |
| seed 2: retention (15) − matched-cost literature (lit_alt_2) (14) | +0.009 [-0.017, +0.038] | +0.014 [-0.015, +0.045] | +0.014 [-0.013, +0.041] |
| seed 2: retention (15) − high-budget literature (lit_alt_4) (28) | -0.032 [-0.059, -0.007] * | -0.026 [-0.056, +0.004] | -0.029 [-0.055, -0.002] * |
| seed 2: retention (15) − high-budget literature (lit_hea_ry_4) (28) | -0.047 [-0.074, -0.019] * | -0.039 [-0.068, -0.012] * | -0.028 [-0.056, -0.000] * |
| seed 2: retention (15) − high-budget literature (lit_hea_ry_8) (56) | -0.073 [-0.096, -0.048] * | -0.039 [-0.068, -0.010] * | -0.066 [-0.095, -0.038] * |
| seed 2: retention (15) − hand-designed 31-CNOT reference (31) | +0.032 [+0.003, +0.059] * | +0.085 [+0.052, +0.117] * | +0.047 [+0.016, +0.076] * |

## Random circuits (12 circuits, CNOTs 12–23)

| metric | random mean ± std | random range | retention picks (seeds) | share of random circuits beaten |
|---|---|---|---|---|
| kernel AUC | 0.820 ± 0.032 | 0.767 – 0.868 | 0.904, 0.909, 0.886 | 100%, 100%, 100% |
| hea8 bal. acc. | 0.408 ± 0.047 | 0.341 – 0.498 | 0.574, 0.614, 0.552 | 100%, 100%, 100% |
| hea8_probs bal. acc. | 0.410 ± 0.050 | 0.352 – 0.482 | 0.573, 0.586, 0.542 | 100%, 100%, 100% |
| qcnn bal. acc. | 0.356 ± 0.043 | 0.297 – 0.432 | 0.496, 0.534, 0.494 | 100%, 100%, 100% |

## Does fidelity predict downstream quality?

| Spearman correlation | value |
|---|---|
| fidelity vs kernel auc | 0.98 |
| fidelity vs hea8 | 0.95 |
| fidelity vs hea8 probs | 0.93 |
| fidelity vs qcnn | 0.92 |
| random only fidelity vs kernel auc | 0.97 |


![](fig_objective_scatter_bloodmnist.png)

## Claims

| claim | status |
|---|---|
| Retention pick beats the fidelity pick at matched CNOT count | not supported (3 better, 1 worse of 9 seed×classifier tests) |
| Retention pick beats the fidelity search's own selection | not supported (0 better, 5 worse of 9 seed×classifier tests) |
| Retention pick beats the best literature ansatz within budget (held-out selection) | not supported (1 better, 4 worse of 9 seed×classifier tests) |
| Retention pick beats matched-cost literature ansaetze | not supported (9 better, 1 worse of 27 seed×classifier tests) |
| Retention pick beats deeper (high-budget) literature ansaetze | not supported (1 better, 13 worse of 27 seed×classifier tests) |
| Retention pick beats the hand-designed 31-CNOT reference | SUPPORTED (9 better, 0 worse of 9 seed×classifier tests) |
| Retention picks beat ≥ 90% of random circuits of similar cost (kernel AUC) | SUPPORTED (100%, 100%, 100%) |
| Test fidelity predicts class information across circuits (|Spearman| ≥ 0.7) | yes (ρ = 0.98) |

