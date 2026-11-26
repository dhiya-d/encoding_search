# Exact classical kernel for entanglement-free encodings (bloodmnist)

Encodings with only single-qubit gates produce product states, so their fidelity kernel is a product of single-qubit overlaps, k(x,x') = Π_q |⟨φ_q(x)|φ_q(x')⟩|², computable classically in O(n) per entry. Kernel check on 300 training images; the SVM is retrained with the classical kernel on exactly the same data and C grid as the quantum kernel.

| encoding | circuit | max |K_quantum − K_classical| | quantum kernel bal. acc. / AUC (saved) | classical kernel bal. acc. / AUC | identical | time: state vector / classical |
|---|---|---|---|---|---|---|
| seed0:ref_angle_RY_d8 | d=8: RY(1x[0+q]) | 3.0e-15 | 0.737 / 0.933 | 0.737 / 0.933 | yes | 119 ms / 27 ms |
| seed0:search_knee_efficient | d=32: RY(0.5x[9+q]) > RX(0.5x[0+q]) > RX(0.5x[0+q]) > RY(0.5x[0+q]) | 5.6e-15 | 0.787 / 0.952 | 0.787 / 0.952 | NO | 103 ms / 27 ms |
| seed1:search_knee_efficient | d=16: RY(1x[13+q]) > RX(0.5x[6+q]) | 4.6e-15 | 0.786 / 0.952 | 0.786 / 0.952 | yes | 45 ms / 80 ms |
| seed2:search_best_knee_efficient | d=32: RZ(0.5x[6+q]) > RX(0.5x[10+q]) > H > RY(0.5x[0+q]) | 3.8e-15 | 0.807 / 0.956 | 0.807 / 0.956 | yes | 195 ms / 52 ms |


**Some results differ; see the table.**

Classical product kernel, 200 × 200 entries, one RY layer:

| qubits | seconds | memory for ONE state vector |
|---|---|---|
| 8 | 0.002 | 10^-5 GB |
| 32 | 0.004 | 10^2 GB |
| 128 | 0.019 | 10^31 GB |
| 512 | 0.066 | 10^146 GB |
| 1024 | 0.319 | 10^300 GB |
