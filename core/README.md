# core/ — shared code

| File | Content |
|---|---|
| `config.py` | settings of the Fourier pipeline (features, search space, NSGA-II, selection) and of all trained classifiers (epochs, seeds, learning rate) |
| `common.py` | dataset loading, DCT features, metrics (balanced accuracy, macro AUC), kernels, kernel-SVM helpers |
| `qsim.py` | batched state-vector simulator: data-encoding gates, trainable gates, diagonal readout, adjoint gradients (also for noisy trajectories) |
| `aae_sim.py` | amplitude-encoding circuits from genomes (hierarqcal connectivity), per-image Adam fitting with adjoint gradients, the hand-designed reference circuit `DEFAULT_GENOME` |
| `encoding_genome.py` | Fourier-encoding genomes: gate lists, two-qubit cost, frequency spectra, literature references, mutation and crossover |
| `classifiers.py` | HEA, QCNN and data re-uploading classifiers; training with early stopping over several seeds; finite-shot evaluation |
| `noise.py` | depolarizing noise by trajectories, mixed-state fidelity and kernel, swap-test kernel estimates |
| `diag_common.py` | preprocessing modes (uncentered / centered / whitened), amplitude targets, helpers of the amplitude pipeline |
| `repo.py` | puts every code folder on `sys.path` and locates scripts (`repo.script(name)`) |

Conventions (all simulators): wire 0 is the most significant bit, R_P(θ) = exp(−iθP/2),
as in PennyLane. `tests/test_verification.py` checks the simulators against PennyLane
and Qiskit.
