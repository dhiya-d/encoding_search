# Automated encoding design: results report

## Claims

| run | claim | status |
|---|---|---|
| bloodmnist:centered | Amplitude: fidelity is a poor fitness for encoder search | SUPPORTED |
| bloodmnist:centered | Amplitude: searched encoder is cheaper than the default | SUPPORTED |
| bloodmnist:centered | Amplitude: searched encoder beats the default under trained classifiers | SUPPORTED (5/5 classifiers significant) |
| bloodmnist:centered | Fourier: search pick beats the best reference encoding (ref_two_layer_d16) | SUPPORTED (2 better, 0 worse of 3) |
| bloodmnist:centered | Fourier encoding beats the searched amplitude encoding (same classifier) | SUPPORTED (2/2) |
| bloodmnist:centered | Best trained quantum model is within 0.02 of the best classical model | not supported (gap 0.105) |
| bloodmnist:centered | Fourier: best quantum model beats classical models on the same features | not supported (0.788 vs 0.813) |
| breastmnist:uncentered | Amplitude: fidelity is a poor fitness for encoder search | not supported |
| breastmnist:uncentered | Amplitude: searched encoder is cheaper than the default | SUPPORTED |
| breastmnist:uncentered | Amplitude: searched encoder beats the default under trained classifiers | not supported (0/5 classifiers significant) |
| breastmnist:uncentered | Fourier: search pick beats the best reference encoding (ref_two_layer_d16) | not supported (0 better, 0 worse of 3) |
| breastmnist:uncentered | Fourier encoding beats the searched amplitude encoding (same classifier) | not supported (0/2) |
| breastmnist:uncentered | Best trained quantum model is within 0.02 of the best classical model | not supported (gap 0.035) |
| breastmnist:uncentered | Fourier: best quantum model beats classical models on the same features | not supported (0.761 vs 0.751) |
| dermamnist | Fourier: search pick beats the best reference encoding (ref_two_layer_d16) | not supported (0 better, 0 worse of 3) |
| dermamnist | Fourier: best quantum model beats classical models on the same features | not supported (0.546 vs 0.532) |
| retinamnist | Fourier: search pick beats the best reference encoding (ref_parallel_d4) | not supported (0 better, 0 worse of 3) |
| retinamnist | Fourier: best quantum model beats classical models on the same features | not supported (0.334 vs 0.393) |

"Supported" is determined by the thresholds in make_report.py; read the numbers below before using a claim.

## bloodmnist (centered)

### Encoder search front (bloodmnist, centered)

| 2q gates | held-out AUC | retention | fidelity |
|---|---|---|---|
| 4 | 0.780 ± 0.026 | 0.886 | 0.373 |
| 8 | 0.786 ± 0.024 | 0.893 | 0.434 |
| 11 | 0.828 ± 0.040 | 0.942 | 0.462 |
| 12 | 0.838 ± 0.021 | 0.953 | 0.548 |
| 15 | 0.832 ± 0.022 | 0.946 | 0.552 |
| 19 | 0.862 ± 0.016 | 0.980 | 0.665 |
| 23 | 0.863 ± 0.018 | 0.981 | 0.698 |
| 26 | 0.863 ± 0.019 | 0.981 | 0.704 |
| 27 | 0.854 ± 0.029 | 0.971 | 0.763 |
| 38 | 0.861 ± 0.029 | 0.979 | 0.813 |

Retention reaches its plateau (within 0.02 of the best) at **19 gates**. Beyond that, fidelity rises by +0.148 while retention varies by only 0.010. Spearman correlation of fidelity and retention along the front: 0.81.

![](fig_search_front_bloodmnist_centered.png)

### Encoders on the test split, sign-blind kernel SVM (bloodmnist, centered)

| encoder | 2q gates | fidelity | bal. acc. | AUC |
|---|---|---|---|---|
| exact states (ceiling) | — | — | 0.718 | 0.927 |
| searched | 19 | 0.666 | 0.659 | 0.904 |
| default | 31 | 0.578 | 0.617 | 0.875 |

### Trained classifiers on amplitude-encoded states (bloodmnist, centered)

Balanced accuracy (mean ± std over seeds) / AUC, test split.

| classifier | exact | searched | default |
|---|---|---|---|
| hea | 0.601 ± 0.017 / 0.898 | 0.552 ± 0.012 / 0.866 | 0.474 ± 0.006 / 0.838 |
| hea8 | 0.633 ± 0.008 / 0.910 | 0.574 ± 0.022 / 0.889 | 0.520 ± 0.020 / 0.849 |
| hea12 | 0.543 ± 0.028 / 0.880 | 0.557 ± 0.016 / 0.881 | 0.473 ± 0.008 / 0.833 |
| hea8_probs | 0.637 ± 0.013 / 0.916 | 0.573 ± 0.001 / 0.890 | 0.457 ± 0.008 / 0.833 |
| qcnn | 0.572 ± 0.009 / 0.886 | 0.496 ± 0.045 / 0.850 | 0.447 ± 0.017 / 0.817 |

**Searched − default** (paired bootstrap on the same test images, 95% CI; * = CI excludes 0):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| hea | +0.078 [+0.049, +0.106] * | +0.028 [+0.017, +0.039] * |
| hea8 | +0.054 [+0.024, +0.082] * | +0.040 [+0.029, +0.051] * |
| hea12 | +0.085 [+0.057, +0.112] * | +0.047 [+0.036, +0.058] * |
| hea8_probs | +0.116 [+0.087, +0.143] * | +0.056 [+0.044, +0.068] * |
| qcnn | +0.049 [+0.021, +0.078] * | +0.033 [+0.021, +0.045] * |

![](fig_searched_vs_default_bloodmnist_centered.png)

### Where the class information goes (bloodmnist, centered)

Balanced accuracy at each stage, same test images; classifier = the trained model that is best on exact states (hea8_probs).

| stage | bal. acc. | change |
|---|---|---|
| classical RBF, RGB 16×16 | 0.815 |  |
| classical RBF, grayscale 16×16 | 0.756 | -0.059 |
| kernel SVM, exact amplitude states | 0.718 | -0.039 |
| hea8_probs, exact states | 0.637 | -0.080 |
| hea8_probs, searched encoder | 0.573 | -0.065 |
| hea8_probs, default encoder | 0.457 | -0.116 |

![](fig_loss_budget_bloodmnist_centered.png)

### Fourier (angle) encodings, full test split (bloodmnist)

Classical models on all features: logistic regression 0.797, RBF SVM 0.813 (balanced accuracy).

| encoding | 2q cost | d | kernel fid. | kernel proj. | reupload | qcnn | hea8 | classical RBF (same d) | classical Fourier (same spectrum) |
|---|---|---|---|---|---|---|---|---|---|
| search_knee_efficient | 0 | 32 | 0.787 | 0.781 | 0.649 | 0.601 | 0.709 | 0.813 | 0.766 |
| search_best | 16 | 32 | 0.788 | 0.742 | 0.687 | 0.648 | 0.710 | 0.813 | 0.766 |
| ref_angle_RY_d8 | 0 | 8 | 0.737 | 0.730 | 0.632 | 0.636 | 0.658 | 0.730 | 0.724 |
| ref_angle_RY_ring_d8 | 8 | 8 | 0.736 | 0.643 | 0.630 | 0.618 | 0.665 | 0.730 | 0.724 |
| ref_parallel_d4 | 8 | 4 | 0.630 | 0.558 | 0.581 | 0.552 | 0.589 | 0.628 | 0.616 |
| ref_two_layer_d16 | 16 | 16 | 0.778 | 0.643 | 0.635 | 0.640 | 0.694 | 0.785 | 0.746 |
| ref_reupload3_d8 | 16 | 8 | 0.726 | 0.662 | 0.608 | 0.615 | 0.667 | 0.730 | 0.636 |
| ref_iqp_zz_d8 | 32 | 8 | 0.712 | 0.712 | 0.609 | 0.563 | 0.647 | 0.730 | 0.679 |

For the search pick **search_best** (chosen on held-out data, not on test), the best quantum model reaches 0.788 and the best classical model on the same features 0.813 (not significance-tested: classical predictions are not saved).

**Search pick (search_best, cost 16) − best reference (ref_two_layer_d16, cost 16)**, reference chosen on held-out data; same classifier, same 3421 test images (paired bootstrap, 95% CI):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| reupload | +0.052 [+0.041, +0.063] * | +0.020 [+0.017, +0.023] * |
| qcnn | +0.007 [-0.007, +0.021] | +0.011 [+0.007, +0.014] * |
| hea8 | +0.016 [+0.006, +0.026] * | +0.003 [+0.001, +0.005] * |

**Fourier encoding (search_best) − searched amplitude encoding**, same classifier, same 800 test images (paired bootstrap, 95% CI):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| qcnn | +0.144 [+0.110, +0.175] * | +0.075 [+0.062, +0.086] * |
| hea8 | +0.136 [+0.105, +0.167] * | +0.056 [+0.046, +0.067] * |

## breastmnist (uncentered)

### Encoder search front (breastmnist, uncentered)

| 2q gates | held-out AUC | retention | fidelity |
|---|---|---|---|
| 4 | 0.673 ± 0.104 | 0.838 | 0.938 |
| 8 | 0.695 ± 0.097 | 0.866 | 0.940 |
| 12 | 0.763 ± 0.123 | 0.950 | 0.953 |
| 19 | 0.779 ± 0.117 | 0.969 | 0.952 |
| 27 | 0.740 ± 0.121 | 0.921 | 0.951 |
| 35 | 0.770 ± 0.123 | 0.959 | 0.970 |

Retention reaches its plateau (within 0.02 of the best) at **12 gates**. Beyond that, fidelity rises by +0.018 while retention varies by only 0.048. Spearman correlation of fidelity and retention along the front: 0.83.

![](fig_search_front_breastmnist_uncentered.png)

### Encoders on the test split, sign-blind kernel SVM (breastmnist, uncentered)

| encoder | 2q gates | fidelity | bal. acc. | AUC |
|---|---|---|---|---|
| exact states (ceiling) | — | — | 0.741 | 0.808 |
| searched | 19 | 0.953 | 0.742 | 0.806 |
| default | 31 | 0.950 | 0.731 | 0.781 |

### Trained classifiers on amplitude-encoded states (breastmnist, uncentered)

Balanced accuracy (mean ± std over seeds) / AUC, test split.

| classifier | exact | searched | default |
|---|---|---|---|
| hea | 0.672 ± 0.014 / 0.738 | 0.666 ± 0.022 / 0.732 | 0.683 ± 0.007 / 0.713 |
| hea8 | 0.694 ± 0.024 / 0.743 | 0.680 ± 0.017 / 0.726 | 0.695 ± 0.002 / 0.724 |
| hea12 | 0.713 ± 0.006 / 0.779 | 0.701 ± 0.003 / 0.771 | 0.685 ± 0.013 / 0.733 |
| hea8_probs | 0.720 ± 0.011 / 0.805 | 0.721 ± 0.037 / 0.787 | 0.717 ± 0.022 / 0.774 |
| qcnn | 0.672 ± 0.030 / 0.704 | 0.685 ± 0.004 / 0.717 | 0.676 ± 0.004 / 0.689 |

**Searched − default** (paired bootstrap on the same test images, 95% CI; * = CI excludes 0):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| hea | -0.017 [-0.059, +0.028] | +0.019 [-0.015, +0.058] |
| hea8 | -0.014 [-0.068, +0.045] | +0.002 [-0.021, +0.026] |
| hea12 | +0.016 [-0.015, +0.048] | +0.037 [+0.009, +0.068] * |
| hea8_probs | +0.003 [-0.054, +0.063] | +0.013 [-0.028, +0.054] |
| qcnn | +0.009 [-0.034, +0.057] | +0.028 [+0.006, +0.055] * |

![](fig_searched_vs_default_breastmnist_uncentered.png)

### Where the class information goes (breastmnist, uncentered)

Balanced accuracy at each stage, same test images; classifier = the trained model that is best on exact states (hea8_probs).

| stage | bal. acc. | change |
|---|---|---|
| classical RBF, grayscale 16×16 | 0.756 |  |
| kernel SVM, exact amplitude states | 0.741 | -0.014 |
| hea8_probs, exact states | 0.720 | -0.022 |
| hea8_probs, searched encoder | 0.721 | +0.001 |
| hea8_probs, default encoder | 0.717 | -0.003 |

![](fig_loss_budget_breastmnist_uncentered.png)

### Fourier (angle) encodings, full test split (breastmnist)

Classical models on all features: logistic regression 0.666, RBF SVM 0.734 (balanced accuracy).

| encoding | 2q cost | d | kernel fid. | kernel proj. | reupload | qcnn | hea8 | classical RBF (same d) | classical Fourier (same spectrum) |
|---|---|---|---|---|---|---|---|---|---|
| search_best_knee | 0 | 24 | 0.758 | 0.761 | 0.581 | 0.661 | 0.693 | 0.727 | 0.576 |
| ref_angle_RY_d8 | 0 | 8 | 0.746 | 0.744 | 0.654 | 0.706 | 0.735 | 0.749 | 0.742 |
| ref_angle_RY_ring_d8 | 8 | 8 | 0.746 | 0.602 | 0.687 | 0.720 | 0.737 | 0.749 | 0.742 |
| ref_parallel_d4 | 8 | 4 | 0.699 | 0.605 | 0.642 | 0.709 | 0.689 | 0.694 | 0.595 |
| ref_two_layer_d16 | 16 | 16 | 0.721 | 0.666 | 0.552 | 0.663 | 0.669 | 0.765 | 0.764 |
| ref_reupload3_d8 | 16 | 8 | 0.765 | 0.672 | 0.658 | 0.686 | 0.704 | 0.749 | 0.712 |
| ref_iqp_zz_d8 | 32 | 8 | 0.734 | 0.701 | 0.712 | 0.692 | 0.707 | 0.749 | 0.723 |

For the search pick **search_best_knee** (chosen on held-out data, not on test), the best quantum model reaches 0.761 and the best classical model on the same features 0.751 (not significance-tested: classical predictions are not saved).

**Search pick (search_best_knee, cost 0) − best reference (ref_two_layer_d16, cost 16)**, reference chosen on held-out data; same classifier, same 156 test images (paired bootstrap, 95% CI):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| reupload | +0.030 [-0.061, +0.121] | +0.065 [-0.028, +0.157] |
| qcnn | -0.002 [-0.056, +0.049] | -0.001 [-0.051, +0.046] |
| hea8 | +0.024 [-0.041, +0.085] | +0.021 [-0.045, +0.086] |

**Fourier encoding (search_best_knee) − searched amplitude encoding**, same classifier, same 156 test images (paired bootstrap, 95% CI):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| qcnn | -0.024 [-0.120, +0.070] | +0.010 [-0.084, +0.109] |
| hea8 | +0.013 [-0.077, +0.098] | +0.054 [-0.029, +0.138] |

## dermamnist (Fourier encodings only)

### Fourier (angle) encodings, full test split (dermamnist)

Classical models on all features: logistic regression 0.527, RBF SVM 0.519 (balanced accuracy).

| encoding | 2q cost | d | kernel fid. | kernel proj. | reupload | qcnn | hea8 | classical RBF (same d) | classical Fourier (same spectrum) |
|---|---|---|---|---|---|---|---|---|---|
| search_knee_efficient | 7 | 16 | 0.533 | 0.454 | 0.401 | 0.381 | 0.473 | 0.528 | 0.483 |
| search_best | 16 | 24 | 0.546 | 0.514 | 0.384 | 0.343 | 0.421 | 0.532 | 0.482 |
| ref_angle_RY_d8 | 0 | 8 | 0.444 | 0.463 | 0.383 | 0.350 | 0.364 | 0.437 | 0.427 |
| ref_angle_RY_ring_d8 | 8 | 8 | 0.444 | 0.362 | 0.374 | 0.331 | 0.374 | 0.437 | 0.427 |
| ref_parallel_d4 | 8 | 4 | 0.362 | 0.268 | 0.321 | 0.306 | 0.277 | 0.370 | 0.326 |
| ref_two_layer_d16 | 16 | 16 | 0.527 | 0.342 | 0.361 | 0.354 | 0.436 | 0.528 | 0.458 |
| ref_reupload3_d8 | 16 | 8 | 0.426 | 0.380 | 0.331 | 0.341 | 0.395 | 0.437 | 0.375 |
| ref_iqp_zz_d8 | 32 | 8 | 0.425 | 0.430 | 0.356 | 0.273 | 0.359 | 0.437 | 0.426 |

For the search pick **search_best** (chosen on held-out data, not on test), the best quantum model reaches 0.546 and the best classical model on the same features 0.532 (not significance-tested: classical predictions are not saved).

**Search pick (search_best, cost 16) − best reference (ref_two_layer_d16, cost 16)**, reference chosen on held-out data; same classifier, same 2005 test images (paired bootstrap, 95% CI):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| reupload | +0.023 [-0.007, +0.051] | +0.025 [+0.013, +0.039] * |
| qcnn | -0.011 [-0.041, +0.020] | -0.019 [-0.032, -0.005] * |
| hea8 | -0.015 [-0.044, +0.014] | -0.012 [-0.022, -0.003] * |

## retinamnist (Fourier encodings only)

### Fourier (angle) encodings, full test split (retinamnist)

Classical models on all features: logistic regression 0.393, RBF SVM 0.324 (balanced accuracy).

| encoding | 2q cost | d | kernel fid. | kernel proj. | reupload | qcnn | hea8 | classical RBF (same d) | classical Fourier (same spectrum) |
|---|---|---|---|---|---|---|---|---|---|
| search_best_knee | 0 | 32 | 0.334 | 0.329 | 0.300 | 0.240 | 0.286 | 0.324 | 0.237 |
| ref_angle_RY_d8 | 0 | 8 | 0.290 | 0.301 | 0.285 | 0.319 | 0.331 | 0.299 | 0.323 |
| ref_angle_RY_ring_d8 | 8 | 8 | 0.290 | 0.246 | 0.285 | 0.323 | 0.333 | 0.299 | 0.323 |
| ref_parallel_d4 | 8 | 4 | 0.299 | 0.213 | 0.252 | 0.291 | 0.318 | 0.263 | 0.252 |
| ref_two_layer_d16 | 16 | 16 | 0.292 | 0.244 | 0.281 | 0.281 | 0.303 | 0.310 | 0.308 |
| ref_reupload3_d8 | 16 | 8 | 0.304 | 0.307 | 0.292 | 0.277 | 0.285 | 0.299 | 0.267 |
| ref_iqp_zz_d8 | 32 | 8 | 0.274 | 0.297 | 0.266 | 0.275 | 0.295 | 0.299 | 0.295 |

For the search pick **search_best_knee** (chosen on held-out data, not on test), the best quantum model reaches 0.334 and the best classical model on the same features 0.393 (not significance-tested: classical predictions are not saved).

**Search pick (search_best_knee, cost 0) − best reference (ref_parallel_d4, cost 8)**, reference chosen on held-out data; same classifier, same 400 test images (paired bootstrap, 95% CI):

| classifier | Δ bal. acc. | Δ AUC |
|---|---|---|
| reupload | +0.048 [-0.006, +0.100] | -0.018 [-0.054, +0.016] |
| qcnn | -0.051 [-0.107, +0.004] | -0.123 [-0.163, -0.083] * |
| hea8 | -0.032 [-0.089, +0.028] | -0.061 [-0.098, -0.025] * |

## Limitations to state explicitly

- Noise-free statevector simulation on 8 qubits; no hardware or noise model.
- Amplitude encoding needs a separate optimisation for every image; its cost grows with the data.
  The Fourier encodings do not have this problem.
- Each encoder search was run with a single random seed. Rerun with different `SEED` values
  before claiming that the search reliably finds these circuits.
- The classifier results average only a few training seeds. Confidence intervals account for the
  test images, not for variation between training runs.
- breastmnist has 156 test images; differences below about 0.05 AUC cannot be resolved there.
- Classical models on the same inputs remain clearly ahead. The contribution is automated encoder
  design and an analysis of where information is lost, not an advantage over classical methods.
