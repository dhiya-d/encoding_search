# Objective study: retention vs fidelity, literature and random circuits (pathmnist, centered)

All circuits are amplitude encoders from the same search space, evaluated on the same test images with the same classifiers. The retention and fidelity searches differ only in the objective.

## Circuits compared

| circuit | type | CNOTs | held-out retention | test fidelity | kernel AUC | hea8 bal. acc. | hea8_probs bal. acc. | qcnn bal. acc. |
|---|---|---|---|---|---|---|---|---|
| retention_s2 | retention search (ours) | 12 | 0.981 | 0.623 | 0.659 | 0.250 | 0.246 | 0.309 |
| retention_s0 | retention search (ours) | 15 | 0.953 | 0.627 | 0.652 | 0.217 | 0.253 | 0.258 |
| retention_s1 | retention search (ours) | 15 | 0.990 | 0.641 | 0.637 | 0.234 | 0.224 | 0.231 |
| fidelity_matched_s1 | fidelity search, matched CNOTs | 12 | — | 0.644 | 0.654 | 0.214 | 0.239 | 0.269 |
| fidelity_matched_s2 | fidelity search, matched CNOTs | 12 | — | 0.643 | 0.654 | 0.227 | 0.253 | 0.270 |
| fidelity_matched_s0 | fidelity search, matched CNOTs | 15 | — | 0.663 | 0.634 | 0.236 | 0.243 | 0.303 |
| fidelity_best_s0 | fidelity search, own selection | 35 | — | 0.842 | 0.657 | 0.325 | 0.275 | 0.365 |
| fidelity_best_s2 | fidelity search, own selection | 35 | — | 0.847 | 0.656 | 0.308 | 0.261 | 0.344 |
| fidelity_best_s1 | fidelity search, own selection | 39 | — | 0.846 | 0.682 | 0.296 | 0.261 | 0.360 |
| lit_alt_2 | literature ansatz | 14 | 0.971 | 0.604 | 0.654 | 0.235 | 0.255 | 0.223 |
| lit_hea_ry_2 | literature ansatz | 14 | 0.974 | 0.641 | 0.630 | 0.262 | 0.247 | 0.298 |
| lit_realamp_2 | literature ansatz | 14 | 0.978 | 0.644 | 0.644 | 0.256 | 0.250 | 0.310 |
| lit_hea_ry_3 | literature ansatz | 21 | 0.959 | 0.682 | 0.657 | 0.274 | 0.253 | 0.333 |
| lit_realamp_3 | literature ansatz | 21 | 0.962 | 0.685 | 0.663 | 0.242 | 0.259 | 0.323 |
| lit_alt_3 | literature ansatz | 22 | 0.980 | 0.682 | 0.654 | 0.263 | 0.261 | 0.310 |
| lit_alt_4 | literature ansatz | 28 | 0.989 | 0.745 | 0.646 | 0.263 | 0.266 | 0.297 |
| lit_hea_ry_4 | literature ansatz | 28 | 0.976 | 0.728 | 0.649 | 0.254 | 0.249 | 0.330 |
| lit_hea_ry_8 | literature ansatz | 56 | 0.988 | 0.833 | 0.649 | 0.332 | 0.238 | 0.344 |
| handref | hand-designed 31-CNOT reference | 31 | — | 0.684 | 0.655 | 0.231 | 0.227 | 0.319 |

Best literature ansatz within the CNOT budget of the searched circuits (≤ 18), chosen on the held-out batch, never on test: **lit_realamp_2** (14 CNOTs). Deeper literature circuits (lit_alt_3, lit_alt_4, lit_hea_ry_3, lit_hea_ry_4, lit_hea_ry_8, lit_realamp_3) are reported as high-budget references.

Kernel AUC of the exact states (ceiling): 0.664.

## Paired comparisons (retention pick − other circuit)

Δ balanced accuracy, paired bootstrap on the same test images, 95% CI, * = excludes 0.

| comparison | hea8 | hea8_probs | qcnn |
|---|---|---|---|
| seed 0: retention (15) − fidelity search, matched CNOTs (15) | -0.019 [-0.038, -0.001] * | +0.009 [-0.014, +0.033] | -0.045 [-0.066, -0.026] * |
| seed 0: retention (15) − fidelity search, own selection (35) | -0.108 [-0.129, -0.086] * | -0.022 [-0.043, +0.002] | -0.107 [-0.129, -0.085] * |
| seed 0: retention (15) − best literature ansatz within budget (lit_realamp_2) (14) | -0.039 [-0.058, -0.019] * | +0.003 [-0.020, +0.025] | -0.052 [-0.070, -0.032] * |
| seed 0: retention (15) − matched-cost literature (lit_hea_ry_2) (14) | -0.044 [-0.067, -0.023] * | +0.006 [-0.018, +0.029] | -0.040 [-0.060, -0.020] * |
| seed 0: retention (15) − matched-cost literature (lit_realamp_2) (14) | -0.039 [-0.058, -0.019] * | +0.003 [-0.020, +0.025] | -0.052 [-0.070, -0.032] * |
| seed 0: retention (15) − matched-cost literature (lit_alt_2) (14) | -0.017 [-0.037, +0.002] | -0.002 [-0.026, +0.020] | +0.035 [+0.019, +0.051] * |
| seed 0: retention (15) − high-budget literature (lit_alt_3) (22) | -0.046 [-0.065, -0.026] * | -0.008 [-0.030, +0.014] | -0.052 [-0.072, -0.034] * |
| seed 0: retention (15) − high-budget literature (lit_alt_4) (28) | -0.046 [-0.066, -0.027] * | -0.013 [-0.036, +0.008] | -0.039 [-0.056, -0.022] * |
| seed 0: retention (15) − high-budget literature (lit_hea_ry_3) (21) | -0.056 [-0.077, -0.035] * | +0.000 [-0.024, +0.024] | -0.075 [-0.096, -0.052] * |
| seed 0: retention (15) − high-budget literature (lit_hea_ry_4) (28) | -0.037 [-0.054, -0.020] * | +0.003 [-0.019, +0.025] | -0.072 [-0.093, -0.052] * |
| seed 0: retention (15) − high-budget literature (lit_hea_ry_8) (56) | -0.114 [-0.137, -0.092] * | +0.015 [-0.008, +0.037] | -0.086 [-0.108, -0.064] * |
| seed 0: retention (15) − high-budget literature (lit_realamp_3) (21) | -0.025 [-0.043, -0.006] * | -0.006 [-0.029, +0.017] | -0.065 [-0.088, -0.044] * |
| seed 0: retention (15) − hand-designed 31-CNOT reference (31) | -0.014 [-0.034, +0.006] | +0.026 [+0.003, +0.050] * | -0.062 [-0.083, -0.040] * |
| seed 1: retention (15) − fidelity search, matched CNOTs (12) | +0.020 [-0.000, +0.040] | -0.015 [-0.038, +0.007] | -0.038 [-0.059, -0.017] * |
| seed 1: retention (15) − fidelity search, own selection (39) | -0.062 [-0.084, -0.041] * | -0.037 [-0.061, -0.015] * | -0.129 [-0.152, -0.106] * |
| seed 1: retention (15) − best literature ansatz within budget (lit_realamp_2) (14) | -0.022 [-0.042, -0.003] * | -0.026 [-0.049, -0.003] * | -0.078 [-0.098, -0.056] * |
| seed 1: retention (15) − matched-cost literature (lit_hea_ry_2) (14) | -0.027 [-0.048, -0.005] * | -0.023 [-0.047, +0.001] | -0.067 [-0.087, -0.045] * |
| seed 1: retention (15) − matched-cost literature (lit_realamp_2) (14) | -0.022 [-0.042, -0.003] * | -0.026 [-0.049, -0.003] * | -0.078 [-0.098, -0.056] * |
| seed 1: retention (15) − matched-cost literature (lit_alt_2) (14) | -0.001 [-0.019, +0.018] | -0.031 [-0.054, -0.008] * | +0.008 [-0.009, +0.028] |
| seed 1: retention (15) − high-budget literature (lit_alt_3) (22) | -0.029 [-0.049, -0.009] * | -0.037 [-0.058, -0.014] * | -0.079 [-0.100, -0.058] * |
| seed 1: retention (15) − high-budget literature (lit_alt_4) (28) | -0.029 [-0.050, -0.008] * | -0.042 [-0.065, -0.022] * | -0.065 [-0.084, -0.046] * |
| seed 1: retention (15) − high-budget literature (lit_hea_ry_3) (21) | -0.039 [-0.061, -0.019] * | -0.028 [-0.052, -0.005] * | -0.101 [-0.123, -0.079] * |
| seed 1: retention (15) − high-budget literature (lit_hea_ry_4) (28) | -0.020 [-0.039, -0.001] * | -0.025 [-0.047, -0.004] * | -0.099 [-0.120, -0.078] * |
| seed 1: retention (15) − high-budget literature (lit_hea_ry_8) (56) | -0.097 [-0.120, -0.075] * | -0.014 [-0.036, +0.009] | -0.113 [-0.134, -0.089] * |
| seed 1: retention (15) − high-budget literature (lit_realamp_3) (21) | -0.008 [-0.028, +0.011] | -0.035 [-0.059, -0.009] * | -0.092 [-0.113, -0.071] * |
| seed 1: retention (15) − hand-designed 31-CNOT reference (31) | +0.003 [-0.018, +0.022] | -0.002 [-0.026, +0.020] | -0.088 [-0.112, -0.065] * |
| seed 2: retention (12) − fidelity search, matched CNOTs (12) | +0.024 [+0.008, +0.040] * | -0.007 [-0.029, +0.015] | +0.040 [+0.018, +0.062] * |
| seed 2: retention (12) − fidelity search, own selection (35) | -0.058 [-0.080, -0.035] * | -0.014 [-0.035, +0.008] | -0.034 [-0.058, -0.011] * |
| seed 2: retention (12) − best literature ansatz within budget (lit_realamp_2) (14) | -0.006 [-0.023, +0.011] | -0.004 [-0.026, +0.019] | -0.000 [-0.022, +0.023] |
| seed 2: retention (12) − matched-cost literature (lit_hea_ry_2) (14) | -0.011 [-0.032, +0.011] | -0.000 [-0.023, +0.024] | +0.011 [-0.011, +0.032] |
| seed 2: retention (12) − matched-cost literature (lit_realamp_2) (14) | -0.006 [-0.023, +0.011] | -0.004 [-0.026, +0.019] | -0.000 [-0.022, +0.023] |
| seed 2: retention (12) − matched-cost literature (lit_alt_2) (14) | +0.016 [-0.003, +0.034] | -0.009 [-0.031, +0.014] | +0.087 [+0.066, +0.108] * |
| seed 2: retention (12) − high-budget literature (lit_alt_3) (22) | -0.013 [-0.030, +0.004] | -0.014 [-0.037, +0.008] | -0.001 [-0.022, +0.022] |
| seed 2: retention (12) − high-budget literature (lit_alt_4) (28) | -0.013 [-0.033, +0.007] | -0.020 [-0.041, -0.000] * | +0.013 [-0.008, +0.034] |
| seed 2: retention (12) − high-budget literature (lit_hea_ry_3) (21) | -0.023 [-0.043, -0.004] * | -0.006 [-0.029, +0.016] | -0.023 [-0.045, -0.000] * |
| seed 2: retention (12) − high-budget literature (lit_hea_ry_4) (28) | -0.004 [-0.017, +0.011] | -0.003 [-0.025, +0.018] | -0.020 [-0.044, +0.004] |
| seed 2: retention (12) − high-budget literature (lit_hea_ry_8) (56) | -0.081 [-0.103, -0.058] * | +0.008 [-0.014, +0.031] | -0.035 [-0.058, -0.010] * |
| seed 2: retention (12) − high-budget literature (lit_realamp_3) (21) | +0.008 [-0.009, +0.025] | -0.013 [-0.035, +0.012] | -0.014 [-0.037, +0.009] |
| seed 2: retention (12) − hand-designed 31-CNOT reference (31) | +0.019 [-0.000, +0.038] | +0.020 [-0.001, +0.042] | -0.010 [-0.035, +0.016] |

## Random circuits (12 circuits, CNOTs 11–18)

| metric | random mean ± std | random range | retention picks (seeds) | share of random circuits beaten |
|---|---|---|---|---|
| kernel AUC | 0.654 ± 0.009 | 0.643 – 0.673 | 0.652, 0.637, 0.659 | 58%, 0%, 75% |
| hea8 bal. acc. | 0.224 ± 0.016 | 0.206 – 0.252 | 0.217, 0.234, 0.250 | 42%, 58%, 92% |
| hea8_probs bal. acc. | 0.230 ± 0.023 | 0.192 – 0.263 | 0.253, 0.224, 0.246 | 75%, 58%, 75% |
| qcnn bal. acc. | 0.222 ± 0.014 | 0.191 – 0.240 | 0.258, 0.231, 0.309 | 100%, 58%, 100% |

## Does fidelity predict downstream quality?

| Spearman correlation | value |
|---|---|
| fidelity vs kernel auc | 0.01 |
| fidelity vs hea8 | 0.75 |
| fidelity vs hea8 probs | 0.56 |
| fidelity vs qcnn | 0.88 |
| random only fidelity vs kernel auc | -0.15 |


![](fig_objective_scatter_pathmnist.png)

## Claims

| claim | status |
|---|---|
| Retention pick beats the fidelity pick at matched CNOT count | not supported (2 better, 3 worse of 9 seed×classifier tests) |
| Retention pick beats the fidelity search's own selection | not supported (0 better, 7 worse of 9 seed×classifier tests) |
| Retention pick beats the best literature ansatz within budget (held-out selection) | not supported (0 better, 5 worse of 9 seed×classifier tests) |
| Retention pick beats matched-cost literature ansaetze | not supported (2 better, 10 worse of 27 seed×classifier tests) |
| Retention pick beats deeper (high-budget) literature ansaetze | not supported (0 better, 33 worse of 54 seed×classifier tests) |
| Retention pick beats the hand-designed 31-CNOT reference | not supported (1 better, 2 worse of 9 seed×classifier tests) |
| Retention picks beat ≥ 90% of random circuits of similar cost (kernel AUC) | not supported (58%, 0%, 75%) |
| Test fidelity predicts class information across circuits (|Spearman| ≥ 0.7) | no (ρ = 0.01) |

