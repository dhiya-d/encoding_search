# ----------------------------------------------------------------
# config.py  —  every setting for the Fourier-encoding pipeline and the
# downstream classifiers.
#
#   search_encoding.py           features -> evolutionary search over
#                                data-encoding circuits -> held-out selection
#   train_downstream.py          picked + reference encodings -> quantum kernels,
#                                re-uploading classifier and QCNN on the TEST
#                                split, next to classical baselines
#   train_on_encoded_states.py   HEA and QCNN trained on the amplitude-encoded
#                                states of the earlier AAE search
# ----------------------------------------------------------------
import numpy as np

RUNS = ["bloodmnist", "breastmnist", "dermamnist", "retinamnist"]
DATA_ROOT = "data"
OUT_ROOT = "experiments_fourier"         # experiments_fourier/<dataset>/{features,search,downstream}/
SEED = 0                                 # search randomness and search/selection batches
DATA_SEED = 0                            # feature pipeline and downstream train/val samples;
                                         # kept fixed so different search seeds are compared
                                         # on identical data
N_QUBITS = 8

# ── Classical preprocessing -> features ─────────────────────────────
# Low-frequency 2D DCT coefficients at native resolution, per colour
# channel. The spectral analysis showed 75-99% of the class-mean signal
# sits at the lowest spatial frequencies, and colour carried more signal
# than amplitude encoding lost (dermamnist, bloodmnist).
# Coefficients are ranked by class-balanced Fisher score on TRAIN only;
# a genome uses the top-d of that ranking.
FEATURE_METHOD = "dct"                   # "dct" | "pca"
DCT_MAX_ORDER = 6                        # keep (u, v) with u + v <= this, per channel
USE_RGB = True                           # use colour when the images have it
MAX_FEATURES = 32                        # size of the ranked pool
ANGLE_CLIP = 3.0                         # standardised features clipped to +-ANGLE_CLIP ...
ANGLE_RANGE = np.pi / 2                  # ... then mapped linearly to +-ANGLE_RANGE (radians)

# ── Encoding search (fitness needs no circuit training) ───────────
# Fitness: stratified-CV AUC of an SVM on the encoding's quantum kernel,
# computed on a fixed class-balanced TRAIN batch.
#   "fidelity"  : k(x,x') = |<psi(x)|psi(x')>|^2    (Schuld & Killoran feature space)
#   "projected" : RBF on single-qubit Pauli expectations (local measurements only)
FITNESS_KERNEL = "fidelity"
SEARCH_N = 600
CV_FOLDS = 5
SVM_C_GRID = (0.1, 1.0, 10.0, 100.0)

# Genome grammar: an ordered list of layers acting on all 8 qubits.
#   enc : e^{-i s x_f P/2} on every qubit, qubit q gets feature (start + q) mod d
#   zz  : e^{-i s x_f x_g ZZ/2} on hierarqcal Qcycle pairs (IQP / Havlicek-type)
#   ent : fixed CNOT / CZ on hierarqcal Qcycle pairs
#   H   : Hadamard on every qubit (the Z_2^n Fourier transform)
# Repeating a feature (serially or in parallel) raises its Fourier degree.
FEATURE_COUNTS = [4, 8, 12, 16, 24, 32]
ENC_AXES = ["X", "Y", "Z"]
ENC_SCALES = [0.5, 1.0, 2.0, 3.0]
ZZ_SCALES = [0.5, 1.0]
ENTANGLERS = ["CNOT", "CZ"]
STRIDES = [1, 2, 3, 4]
STEPS = [1, 2]
OFFSETS = [0, 1]
LAYER_WEIGHTS = {"enc": 4, "ent": 3, "zz": 1, "H": 1}   # sampling weights for new layers
MIN_LAYERS = 1
MAX_LAYERS = 14
INIT_MAX_LAYERS = 7
MAX_TWO_QUBIT_COST = 48                  # CNOT/CZ count 1, each ZZ rotation counts 2

POP_SIZE = 40
N_GENERATIONS = 30
TOURNAMENT_K = 2
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.9
N_WORKERS = None                         # None = min(8, CPU count); 1 = serial

# ── Selection on a held-out batch (val split + unused train) ────────
SELECT_N = 800
SELECT_TOLERANCE = 0.005                 # AUC tolerance for "cheapest near-best"
N_PICKS = 3                              # front members taken to downstream training

# ── Quantum classifiers (classifiers.py) ──────────────────────────
#   reupload : W0, then [S(x), W] x VQC_REUPLOADS; W = RY, RZ on every
#              qubit + CNOT ring; readout <Z_q>, <Z_q Z_q+1> -> linear layer
#              (Schuld, Sweke & Meyer 2021 layout; the linear layer makes the
#              observable trainable, their Sec. V B)
#   qcnn     : S(x) once, then a hierarqcal-style QCNN (conv on Qcycle pairs,
#              pooling halves the active qubits); readout = basis
#              probabilities of the remaining 2 (<= 4 classes) or 4 qubits
#   hea      : CLF_HEA_LAYERS of [RY, RZ, CNOT ring] (amplitude-encoded inputs)
#              variants: hea<L> = L layers; *_probs = readout of all 256 basis
#              probabilities (one measurement setting), scaled by 256
VQC_REUPLOADS = 2
CLF_HEA_LAYERS = 4
VQC_EPOCHS = 40
VQC_BATCH = 64
VQC_LR = 0.03
VQC_PATIENCE = 8                         # early stopping on validation balanced accuracy
CLF_SEEDS = [0, 1]                       # results are mean +- std over these seeds
CLF_WEIGHT_DECAY = 1e-3                  # L2 on the linear readout, applied only to readouts
                                         # with more than 16 features (the *_probs variants)

# ── Downstream evaluation of Fourier encodings (train_downstream.py) ─
CLASSIFIERS = ["reupload", "qcnn", "hea8"]   # trained classifiers per encoding
KERNEL_N_TRAIN = 2000                    # kernel SVM / classical baselines: TRAIN images
VQC_N_TRAIN = 2000                       # trained classifiers: TRAIN images
VQC_N_VAL = 500                          # early-stopping images (VAL split)
TEST_MAX = None                          # cap on test images (None = full test split)
RFF_N_FEATURES = 512                     # classical Fourier model restricted to the
                                         # encoding's own spectrum (dequantisation check)
