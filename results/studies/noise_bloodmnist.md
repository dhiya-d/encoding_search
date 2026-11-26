# Noise study (bloodmnist)

Two-qubit depolarizing noise p after every two-qubit gate of the encoding, single-qubit noise 0.1·p after every single-qubit gate; 4 noise trajectories per image; classifier noise-free. Amplitude encoders: bloodmnist:centered, 800 test images. Fourier encoders: bloodmnist, full test split.

## hea8 balanced accuracy (mean over seeds)

| encoder | CNOTs | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|
| aae_retention | 19 | 0.574 | 0.582 | 0.574 | 0.560 | 0.539 |
| aae_handref | 31 | 0.520 | 0.492 | 0.478 | 0.451 | 0.396 |
| aae_fidelity | 19 | 0.594 | 0.559 | 0.571 | 0.538 | 0.538 |
| aae_literature | 21 | 0.586 | 0.575 | 0.565 | 0.533 | 0.515 |
| aae_fidelity_deep | 41 | 0.612 | 0.591 | 0.553 | 0.529 | 0.459 |
| aae_literature_deep | 56 | 0.625 | 0.594 | 0.571 | 0.496 | 0.378 |
| fourier_pick | 16 | 0.710 | 0.712 | 0.690 | 0.687 | 0.677 |
| fourier_zero | 0 | 0.709 | 0.707 | 0.699 | 0.717 | 0.694 |
| fourier_ref | 16 | 0.694 | 0.699 | 0.697 | 0.675 | 0.660 |

## kernel SVM balanced accuracy

| encoder | CNOTs | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|
| aae_retention | 19 | 0.634 | 0.630 | 0.624 | 0.620 | 0.589 |
| aae_handref | 31 | 0.580 | 0.552 | 0.526 | 0.538 | 0.481 |
| aae_fidelity | 19 | 0.636 | 0.631 | 0.625 | 0.600 | 0.573 |
| aae_literature | 21 | 0.634 | 0.617 | 0.603 | 0.606 | 0.571 |
| aae_fidelity_deep | 41 | 0.672 | 0.667 | 0.636 | 0.624 | 0.566 |
| aae_literature_deep | 56 | 0.655 | 0.660 | 0.635 | 0.609 | 0.521 |
| fourier_pick | 16 | 0.767 | 0.759 | 0.754 | 0.745 | 0.710 |
| fourier_zero | 0 | 0.771 | 0.767 | 0.765 | 0.764 | 0.759 |
| fourier_ref | 16 | 0.759 | 0.753 | 0.744 | 0.735 | 0.704 |

## mean fidelity to the exact state

| encoder | CNOTs | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|
| aae_retention | 19 | 0.665 | 0.637 | 0.602 | 0.548 | 0.449 |
| aae_handref | 31 | 0.577 | 0.542 | 0.486 | 0.422 | 0.308 |
| aae_fidelity | 19 | 0.658 | 0.631 | 0.601 | 0.533 | 0.444 |
| aae_literature | 21 | 0.627 | 0.600 | 0.564 | 0.515 | 0.411 |
| aae_fidelity_deep | 41 | 0.829 | 0.758 | 0.664 | 0.541 | 0.352 |
| aae_literature_deep | 56 | 0.823 | 0.727 | 0.613 | 0.449 | 0.260 |

## Paired comparisons under noise (hea8, Δ balanced accuracy, 95% CI, * = excludes 0)

| comparison | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|
| aae_retention (19) − aae_literature (21) | -0.013 [-0.039, +0.014] | +0.007 [-0.018, +0.032] | +0.009 [-0.018, +0.036] | +0.027 [-0.001, +0.055] | +0.024 [-0.008, +0.055] |
| aae_retention (19) − aae_fidelity (19) | -0.021 [-0.046, +0.004] | +0.023 [-0.006, +0.049] | +0.003 [-0.023, +0.029] | +0.022 [-0.006, +0.050] | +0.001 [-0.027, +0.030] |
| aae_retention (19) − aae_fidelity_deep (41) | -0.038 [-0.062, -0.013] * | -0.009 [-0.035, +0.016] | +0.021 [-0.006, +0.047] | +0.031 [+0.003, +0.061] * | +0.080 [+0.048, +0.113] * |
| aae_retention (19) − aae_literature_deep (56) | -0.051 [-0.076, -0.027] * | -0.013 [-0.038, +0.014] | +0.002 [-0.027, +0.030] | +0.064 [+0.034, +0.094] * | +0.161 [+0.126, +0.195] * |
| aae_retention (19) − aae_handref (31) | +0.054 [+0.024, +0.082] * | +0.090 [+0.060, +0.119] * | +0.096 [+0.066, +0.126] * | +0.109 [+0.078, +0.139] * | +0.143 [+0.106, +0.176] * |
| fourier_pick (16) − fourier_ref (16) | +0.016 [+0.006, +0.026] * | +0.013 [+0.002, +0.024] * | -0.006 [-0.017, +0.004] | +0.012 [+0.000, +0.023] * | +0.017 [+0.004, +0.029] * |

## Finite number of measurements (hea8 and kernel SVM)

Readouts estimated from S measurement shots of one setting (same trained models); kernel entries estimated with S swap-test shots each.

Entries: classifier balanced accuracy / kernel SVM balanced accuracy.

| encoder | CNOTs | S | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|---|---|
| aae_retention | 19 | exact | 0.574 / 0.634 | 0.582 / 0.630 | 0.574 / 0.624 | 0.560 / 0.620 | 0.539 / 0.589 |
| aae_retention | 19 | 1000 | 0.556 / 0.604 | 0.569 / 0.610 | 0.559 / 0.586 | 0.545 / 0.576 | 0.512 / 0.539 |
| aae_retention | 19 | 100 | 0.514 / 0.509 | 0.492 / 0.496 | 0.464 / 0.465 | 0.449 / 0.441 | 0.435 / 0.354 |
| aae_handref | 31 | exact | 0.520 / 0.580 | 0.492 / 0.552 | 0.478 / 0.526 | 0.451 / 0.538 | 0.396 / 0.481 |
| aae_handref | 31 | 1000 | 0.501 / 0.541 | 0.472 / 0.502 | 0.461 / 0.486 | 0.452 / 0.453 | 0.377 / 0.325 |
| aae_handref | 31 | 100 | 0.414 / 0.425 | 0.402 / 0.401 | 0.392 / 0.349 | 0.358 / 0.314 | 0.338 / 0.290 |
| aae_fidelity | 19 | exact | 0.594 / 0.636 | 0.559 / 0.631 | 0.571 / 0.625 | 0.538 / 0.600 | 0.538 / 0.573 |
| aae_fidelity | 19 | 1000 | 0.586 / 0.600 | 0.544 / 0.603 | 0.563 / 0.568 | 0.514 / 0.539 | 0.522 / 0.514 |
| aae_fidelity | 19 | 100 | 0.456 / 0.501 | 0.486 / 0.483 | 0.474 / 0.439 | 0.441 / 0.376 | 0.411 / 0.389 |
| aae_literature | 21 | exact | 0.586 / 0.634 | 0.575 / 0.617 | 0.565 / 0.603 | 0.533 / 0.606 | 0.515 / 0.571 |
| aae_literature | 21 | 1000 | 0.569 / 0.587 | 0.555 / 0.573 | 0.562 / 0.580 | 0.524 / 0.546 | 0.507 / 0.499 |
| aae_literature | 21 | 100 | 0.522 / 0.480 | 0.463 / 0.471 | 0.483 / 0.430 | 0.431 / 0.395 | 0.418 / 0.320 |
| aae_fidelity_deep | 41 | exact | 0.612 / 0.672 | 0.591 / 0.667 | 0.553 / 0.636 | 0.529 / 0.624 | 0.459 / 0.566 |
| aae_fidelity_deep | 41 | 1000 | 0.603 / 0.640 | 0.581 / 0.599 | 0.547 / 0.586 | 0.503 / 0.539 | 0.436 / 0.329 |
| aae_fidelity_deep | 41 | 100 | 0.519 / 0.526 | 0.474 / 0.489 | 0.449 / 0.407 | 0.411 / 0.352 | 0.354 / 0.304 |
| aae_literature_deep | 56 | exact | 0.625 / 0.655 | 0.594 / 0.660 | 0.571 / 0.635 | 0.496 / 0.609 | 0.378 / 0.521 |
| aae_literature_deep | 56 | 1000 | 0.602 / 0.632 | 0.586 / 0.605 | 0.554 / 0.573 | 0.463 / 0.312 | 0.373 / 0.224 |
| aae_literature_deep | 56 | 100 | 0.514 / 0.499 | 0.476 / 0.465 | 0.438 / 0.358 | 0.372 / 0.328 | 0.294 / 0.232 |
| fourier_pick | 16 | exact | 0.710 / 0.767 | 0.712 / 0.759 | 0.690 / 0.754 | 0.687 / 0.745 | 0.677 / 0.710 |
| fourier_pick | 16 | 1000 | 0.697 / 0.697 | 0.693 / 0.693 | 0.673 / 0.695 | 0.672 / 0.683 | 0.660 / 0.657 |
| fourier_pick | 16 | 100 | 0.604 / 0.633 | 0.606 / 0.626 | 0.610 / 0.628 | 0.586 / 0.623 | 0.574 / 0.606 |
| fourier_zero | 0 | exact | 0.709 / 0.771 | 0.707 / 0.767 | 0.699 / 0.765 | 0.717 / 0.764 | 0.694 / 0.759 |
| fourier_zero | 0 | 1000 | 0.700 / 0.703 | 0.696 / 0.703 | 0.686 / 0.704 | 0.703 / 0.695 | 0.679 / 0.688 |
| fourier_zero | 0 | 100 | 0.600 / 0.632 | 0.613 / 0.630 | 0.607 / 0.631 | 0.600 / 0.620 | 0.595 / 0.621 |
| fourier_ref | 16 | exact | 0.694 / 0.759 | 0.699 / 0.753 | 0.697 / 0.744 | 0.675 / 0.735 | 0.660 / 0.704 |
| fourier_ref | 16 | 1000 | 0.676 / 0.711 | 0.681 / 0.705 | 0.683 / 0.697 | 0.664 / 0.687 | 0.638 / 0.655 |
| fourier_ref | 16 | 100 | 0.584 / 0.642 | 0.585 / 0.654 | 0.582 / 0.615 | 0.553 / 0.645 | 0.523 / 0.613 |

Paired comparisons with S = 100 shots (hea8, Δ balanced accuracy, 95% CI):

| comparison | p = 0 | p = 0.002 | p = 0.005 | p = 0.01 | p = 0.02 |
|---|---|---|---|---|---|
| aae_retention − aae_literature | -0.008 [-0.036, +0.019] | +0.029 [-0.001, +0.057] | -0.019 [-0.048, +0.011] | +0.019 [-0.010, +0.049] | +0.018 [-0.015, +0.050] |
| aae_retention − aae_fidelity | +0.058 [+0.029, +0.088] * | +0.006 [-0.024, +0.034] | -0.009 [-0.040, +0.021] | +0.008 [-0.021, +0.036] | +0.024 [-0.008, +0.054] |
| aae_retention − aae_fidelity_deep | -0.005 [-0.034, +0.023] | +0.018 [-0.013, +0.047] | +0.015 [-0.015, +0.046] | +0.039 [+0.008, +0.071] * | +0.081 [+0.049, +0.112] * |
| aae_retention − aae_literature_deep | +0.000 [-0.031, +0.028] | +0.016 [-0.014, +0.044] | +0.026 [-0.002, +0.056] | +0.077 [+0.045, +0.108] * | +0.141 [+0.110, +0.171] * |
| aae_retention − aae_handref | +0.100 [+0.069, +0.130] * | +0.089 [+0.058, +0.121] * | +0.073 [+0.043, +0.104] * | +0.091 [+0.060, +0.122] * | +0.098 [+0.065, +0.129] * |
| fourier_pick − fourier_ref | +0.020 [+0.006, +0.033] * | +0.021 [+0.006, +0.036] * | +0.029 [+0.014, +0.043] * | +0.032 [+0.017, +0.048] * | +0.052 [+0.036, +0.067] * |


![](fig_noise_bloodmnist.png)

## Claims

| claim | status |
|---|---|
| aae_retention better than aae_literature at every noise level | not supported (0/5 significant, 0 worse); Δ -0.013 at p = 0 → +0.024 at p = 0.02 |
| aae_retention better than aae_fidelity at every noise level | not supported (0/5 significant, 0 worse); Δ -0.021 at p = 0 → +0.001 at p = 0.02 |
| aae_retention better than aae_fidelity_deep at every noise level | not supported (2/5 significant, 1 worse); Δ -0.038 at p = 0 → +0.080 at p = 0.02 |
| aae_retention better than aae_literature_deep at every noise level | not supported (2/5 significant, 1 worse); Δ -0.051 at p = 0 → +0.161 at p = 0.02 |
| aae_retention better than aae_handref at every noise level | SUPPORTED (5/5 significant, 0 worse); Δ +0.054 at p = 0 → +0.143 at p = 0.02 |
| fourier_pick better than fourier_ref at every noise level | not supported (4/5 significant, 0 worse); Δ +0.016 at p = 0 → +0.017 at p = 0.02 |

Accuracy lost between the lowest and highest noise level: aae_retention +0.035, aae_handref +0.124, aae_fidelity +0.057, aae_literature +0.071, aae_fidelity_deep +0.153, aae_literature_deep +0.247, fourier_pick +0.034, fourier_zero +0.015, fourier_ref +0.035.
