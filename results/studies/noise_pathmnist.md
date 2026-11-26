# Noise study (pathmnist)

Two-qubit depolarizing noise p after every two-qubit gate of the encoding, single-qubit noise 0.1·p after every single-qubit gate; 4 noise trajectories per image; classifier noise-free. Amplitude encoders: pathmnist:centered, 800 test images. Fourier encoders: pathmnist, full test split.

## hea8 balanced accuracy (mean over seeds)

| encoder | CNOTs | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|
| aae_retention | 15 | 0.217 | 0.227 | 0.221 | 0.211 | 0.206 |
| aae_handref | 31 | 0.231 | 0.234 | 0.230 | 0.224 | 0.203 |
| aae_fidelity | 15 | 0.236 | 0.232 | 0.263 | 0.230 | 0.205 |
| aae_literature | 14 | 0.256 | 0.239 | 0.249 | 0.219 | 0.220 |
| aae_fidelity_deep | 35 | 0.325 | 0.305 | 0.302 | 0.256 | 0.236 |
| aae_literature_deep | 56 | 0.332 | 0.274 | 0.268 | 0.251 | 0.188 |
| fourier_pick | 4 | 0.516 | 0.514 | 0.510 | 0.514 | 0.510 |
| fourier_zero | 0 | 0.472 | 0.490 | 0.463 | 0.499 | 0.515 |
| fourier_ref | 8 | 0.491 | 0.490 | 0.499 | 0.497 | 0.491 |

## kernel SVM balanced accuracy

| encoder | CNOTs | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|
| aae_retention | 15 | 0.207 | 0.240 | 0.234 | 0.237 | 0.209 |
| aae_handref | 31 | 0.237 | 0.233 | 0.233 | 0.235 | 0.200 |
| aae_fidelity | 15 | 0.240 | 0.249 | 0.236 | 0.233 | 0.208 |
| aae_literature | 14 | 0.211 | 0.233 | 0.233 | 0.211 | 0.206 |
| aae_fidelity_deep | 35 | 0.280 | 0.273 | 0.267 | 0.267 | 0.231 |
| aae_literature_deep | 56 | 0.265 | 0.260 | 0.246 | 0.235 | 0.200 |
| fourier_pick | 4 | 0.561 | 0.554 | 0.499 | 0.517 | 0.519 |
| fourier_zero | 0 | 0.566 | 0.567 | 0.556 | 0.546 | 0.523 |
| fourier_ref | 8 | 0.558 | 0.511 | 0.500 | 0.501 | 0.482 |

## mean fidelity to the exact state

| encoder | CNOTs | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|
| aae_retention | 15 | 0.642 | 0.626 | 0.596 | 0.553 | 0.488 |
| aae_handref | 31 | 0.696 | 0.655 | 0.587 | 0.517 | 0.381 |
| aae_fidelity | 15 | 0.678 | 0.656 | 0.638 | 0.591 | 0.502 |
| aae_literature | 14 | 0.658 | 0.640 | 0.614 | 0.566 | 0.501 |
| aae_fidelity_deep | 35 | 0.849 | 0.794 | 0.707 | 0.592 | 0.411 |
| aae_literature_deep | 56 | 0.840 | 0.743 | 0.631 | 0.468 | 0.272 |

## Paired comparisons under noise (hea8, Δ balanced accuracy, 95% CI, * = excludes 0)

| comparison | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|
| aae_retention (15) − aae_literature (14) | -0.039 [-0.058, -0.019] * | -0.012 [-0.033, +0.010] | -0.028 [-0.053, -0.004] * | -0.007 [-0.031, +0.016] | -0.015 [-0.036, +0.008] |
| aae_retention (15) − aae_fidelity (15) | -0.019 [-0.038, -0.001] * | -0.005 [-0.025, +0.015] | -0.042 [-0.065, -0.019] * | -0.019 [-0.043, +0.004] | +0.001 [-0.021, +0.024] |
| aae_retention (15) − aae_fidelity_deep (35) | -0.108 [-0.129, -0.086] * | -0.078 [-0.101, -0.055] * | -0.081 [-0.105, -0.057] * | -0.044 [-0.066, -0.020] * | -0.030 [-0.055, -0.003] * |
| aae_retention (15) − aae_literature_deep (56) | -0.114 [-0.137, -0.092] * | -0.047 [-0.073, -0.022] * | -0.047 [-0.071, -0.024] * | -0.040 [-0.064, -0.016] * | +0.018 [-0.007, +0.045] |
| aae_retention (15) − aae_handref (31) | -0.014 [-0.034, +0.006] | -0.007 [-0.030, +0.014] | -0.008 [-0.032, +0.016] | -0.012 [-0.035, +0.010] | +0.003 [-0.021, +0.027] |
| fourier_pick (4) − fourier_ref (8) | +0.024 [+0.016, +0.033] * | +0.024 [+0.016, +0.033] * | +0.010 [+0.002, +0.019] * | +0.018 [+0.009, +0.026] * | +0.019 [+0.009, +0.028] * |

## Finite number of measurements (hea8 and kernel SVM)

Readouts estimated from S measurement shots of one setting (same trained models); kernel entries estimated with S swap-test shots each.

Entries: classifier balanced accuracy / kernel SVM balanced accuracy.

| encoder | CNOTs | S | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|---|
| aae_retention | 15 | exact | 0.217 / 0.207 | 0.227 / 0.240 | 0.221 / 0.234 | 0.211 / 0.237 | 0.206 / 0.209 |
| aae_retention | 15 | 1000 | 0.217 / 0.244 | 0.220 / 0.212 | 0.206 / 0.216 | 0.215 / 0.207 | 0.210 / 0.211 |
| aae_retention | 15 | 100 | 0.212 / 0.229 | 0.196 / 0.227 | 0.196 / 0.201 | 0.201 / 0.210 | 0.191 / 0.194 |
| aae_handref | 31 | exact | 0.231 / 0.237 | 0.234 / 0.233 | 0.230 / 0.233 | 0.224 / 0.235 | 0.203 / 0.200 |
| aae_handref | 31 | 1000 | 0.228 / 0.237 | 0.221 / 0.222 | 0.206 / 0.210 | 0.208 / 0.238 | 0.204 / 0.195 |
| aae_handref | 31 | 100 | 0.208 / 0.228 | 0.188 / 0.240 | 0.198 / 0.218 | 0.217 / 0.202 | 0.185 / 0.185 |
| aae_fidelity | 15 | exact | 0.236 / 0.240 | 0.232 / 0.249 | 0.263 / 0.236 | 0.230 / 0.233 | 0.205 / 0.208 |
| aae_fidelity | 15 | 1000 | 0.242 / 0.220 | 0.235 / 0.220 | 0.257 / 0.230 | 0.227 / 0.202 | 0.209 / 0.203 |
| aae_fidelity | 15 | 100 | 0.222 / 0.225 | 0.210 / 0.236 | 0.208 / 0.208 | 0.212 / 0.236 | 0.202 / 0.206 |
| aae_literature | 14 | exact | 0.256 / 0.211 | 0.239 / 0.233 | 0.249 / 0.233 | 0.219 / 0.211 | 0.220 / 0.206 |
| aae_literature | 14 | 1000 | 0.239 / 0.224 | 0.220 / 0.212 | 0.224 / 0.237 | 0.214 / 0.217 | 0.214 / 0.223 |
| aae_literature | 14 | 100 | 0.204 / 0.229 | 0.194 / 0.235 | 0.202 / 0.213 | 0.185 / 0.223 | 0.206 / 0.212 |
| aae_fidelity_deep | 35 | exact | 0.325 / 0.280 | 0.305 / 0.273 | 0.302 / 0.267 | 0.256 / 0.267 | 0.236 / 0.231 |
| aae_fidelity_deep | 35 | 1000 | 0.300 / 0.262 | 0.286 / 0.231 | 0.284 / 0.252 | 0.254 / 0.231 | 0.240 / 0.178 |
| aae_fidelity_deep | 35 | 100 | 0.251 / 0.228 | 0.233 / 0.238 | 0.238 / 0.215 | 0.234 / 0.212 | 0.226 / 0.181 |
| aae_literature_deep | 56 | exact | 0.332 / 0.265 | 0.274 / 0.260 | 0.268 / 0.246 | 0.251 / 0.235 | 0.188 / 0.200 |
| aae_literature_deep | 56 | 1000 | 0.305 / 0.261 | 0.269 / 0.228 | 0.253 / 0.239 | 0.243 / 0.217 | 0.190 / 0.163 |
| aae_literature_deep | 56 | 100 | 0.242 / 0.210 | 0.218 / 0.247 | 0.223 / 0.219 | 0.224 / 0.222 | 0.189 / 0.151 |
| fourier_pick | 4 | exact | 0.516 / 0.561 | 0.514 / 0.554 | 0.510 / 0.499 | 0.514 / 0.517 | 0.510 / 0.519 |
| fourier_pick | 4 | 1000 | 0.493 / 0.478 | 0.490 / 0.477 | 0.483 / 0.468 | 0.486 / 0.498 | 0.489 / 0.454 |
| fourier_pick | 4 | 100 | 0.388 / 0.430 | 0.391 / 0.437 | 0.395 / 0.407 | 0.380 / 0.409 | 0.387 / 0.400 |
| fourier_zero | 0 | exact | 0.472 / 0.566 | 0.490 / 0.567 | 0.463 / 0.556 | 0.499 / 0.546 | 0.515 / 0.523 |
| fourier_zero | 0 | 1000 | 0.453 / 0.461 | 0.467 / 0.483 | 0.446 / 0.459 | 0.468 / 0.482 | 0.483 / 0.472 |
| fourier_zero | 0 | 100 | 0.384 / 0.428 | 0.375 / 0.438 | 0.381 / 0.445 | 0.364 / 0.446 | 0.381 / 0.435 |
| fourier_ref | 8 | exact | 0.491 / 0.558 | 0.490 / 0.511 | 0.499 / 0.500 | 0.497 / 0.501 | 0.491 / 0.482 |
| fourier_ref | 8 | 1000 | 0.480 / 0.480 | 0.477 / 0.498 | 0.485 / 0.466 | 0.479 / 0.470 | 0.473 / 0.477 |
| fourier_ref | 8 | 100 | 0.417 / 0.435 | 0.421 / 0.431 | 0.406 / 0.420 | 0.411 / 0.415 | 0.373 / 0.405 |

Paired comparisons with S = 100 shots (hea8, Δ balanced accuracy, 95% CI):

| comparison | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|
| aae_retention − aae_literature | +0.008 [-0.015, +0.030] | +0.002 [-0.022, +0.025] | -0.006 [-0.030, +0.018] | +0.016 [-0.008, +0.040] | -0.015 [-0.038, +0.008] |
| aae_retention − aae_fidelity | -0.010 [-0.031, +0.010] | -0.014 [-0.037, +0.010] | -0.013 [-0.037, +0.011] | -0.012 [-0.036, +0.013] | -0.010 [-0.034, +0.012] |
| aae_retention − aae_fidelity_deep | -0.039 [-0.064, -0.015] * | -0.037 [-0.062, -0.010] * | -0.042 [-0.067, -0.016] * | -0.033 [-0.057, -0.008] * | -0.035 [-0.060, -0.009] * |
| aae_retention − aae_literature_deep | -0.030 [-0.055, -0.006] * | -0.022 [-0.048, +0.003] | -0.027 [-0.053, -0.002] * | -0.023 [-0.048, +0.003] | +0.002 [-0.023, +0.028] |
| aae_retention − aae_handref | +0.004 [-0.018, +0.024] | +0.009 [-0.016, +0.033] | -0.003 [-0.027, +0.022] | -0.016 [-0.041, +0.008] | +0.006 [-0.018, +0.030] |
| fourier_pick − fourier_ref | -0.029 [-0.039, -0.018] * | -0.030 [-0.041, -0.020] * | -0.010 [-0.021, +0.000] | -0.031 [-0.042, -0.021] * | +0.014 [+0.004, +0.024] * |


![](fig_noise_pathmnist.png)

## Claims

| claim | status |
|---|---|
| aae_retention better than aae_literature at every noise level | not supported (0/5 significant, 2 worse); Δ -0.039 at p = 0 → -0.015 at p = 0.02 |
| aae_retention better than aae_fidelity at every noise level | not supported (0/5 significant, 2 worse); Δ -0.019 at p = 0 → +0.001 at p = 0.02 |
| aae_retention better than aae_fidelity_deep at every noise level | not supported (0/5 significant, 5 worse); Δ -0.108 at p = 0 → -0.030 at p = 0.02 |
| aae_retention better than aae_literature_deep at every noise level | not supported (0/5 significant, 4 worse); Δ -0.114 at p = 0 → +0.018 at p = 0.02 |
| aae_retention better than aae_handref at every noise level | not supported (0/5 significant, 0 worse); Δ -0.014 at p = 0 → +0.003 at p = 0.02 |
| fourier_pick better than fourier_ref at every noise level | SUPPORTED (5/5 significant, 0 worse); Δ +0.024 at p = 0 → +0.019 at p = 0.02 |

Accuracy lost between the lowest and highest noise level: aae_retention +0.012, aae_handref +0.028, aae_fidelity +0.032, aae_literature +0.036, aae_fidelity_deep +0.090, aae_literature_deep +0.144, fourier_pick +0.005, fourier_zero -0.043, fourier_ref -0.000.
