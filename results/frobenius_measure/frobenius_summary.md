# Frobenius (reconstruction) search: summary

Saturation rule: suffix-mean probe within 0.03 of the probe on exact targets.

| dataset | K | B (test) | probe raw | g* per seed | probe 1-5 gates | probe >= 26 gates | Frobenius >= 26 gates | knee gates | hours/seed | QCNN val | QCNN test |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bloodmnist (none) | 8 | n/a | 0.6160 | [7, 4, 5, 4, 7] | 0.501 ± 0.037 | 0.607 ± 0.007 | 0.443 ± 0.052 | [45, 83, 45, 45, 31] | 0.88 | -- | not run |
| breastmnist (none) | 2 | 0.0273 | 0.7018 | [1, 2, 1, 1, 1] | 0.660 ± 0.024 | 0.691 ± 0.010 | 0.306 ± 0.053 | [73, 51, 24, 25, 39] | 0.86 | 0.735 | pending |
| dermamnist (none) | 7 | 0.0443 | 0.3250 | [13, 8, None, 3, 3] | 0.255 ± 0.019 | 0.315 ± 0.012 | 0.299 ± 0.043 | [36, 34, 18, 30, 43] | 0.80 | 0.303 | pending |
| retinamnist (retina4) | 4 | 0.0704 | 0.4557 | [8, 1, 11, 8, 9] | 0.390 ± 0.017 | 0.438 ± 0.012 | 0.294 ± 0.035 | [54, 55, 34, 54, 26] | 0.77 | 0.401 | pending |
