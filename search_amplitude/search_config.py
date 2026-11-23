# ----------------------------------------------------------------
# search_config.py  —  Settings for the retention-driven encoding search
# (encoding_search.py) and the final selection (select_encoder.py).
# ----------------------------------------------------------------

# (dataset, preprocessing mode) pairs to search. Modes: "uncentered" | "centered".
# Chosen from the diagnostics:
#   bloodmnist  centered    strongest ceiling (AUC ~0.93), clear encoding loss to close
#   breastmnist uncentered  centering makes the two class means antiparallel,
#                           which quantum readouts cannot separate
# Add ("dermamnist", "centered") if you want it; grayscale already costs
# dermamnist more than encoding does. retinamnist is at noise level.
RUNS = [
    ("bloodmnist", "centered"),
    ("breastmnist", "uncentered"),
]

DATA_ROOT = "data"
OUT_ROOT = "experiments"          # results in experiments/<dataset>/search/<mode>/
SEED = 0

# ── Fitness (evaluated for every genome during the search) ─────────
# Each genome encodes a fixed, class-balanced batch of TRAIN images
# (per-image fidelity training, as before), then a kernel SVM on the
# sign-blind fidelity kernel |<psi_i|psi_j>|^2 is scored by stratified
# k-fold CV on that batch. Fitness = CV AUC / CV AUC of the exact states
# (the physical ceiling), i.e. the fraction of class signal retained.
# Mean fidelity is logged but NOT optimised.
SEARCH_N_IMAGES = 400             # same batch for every genome
CV_FOLDS = 5
SVM_C_GRID = (0.1, 1.0, 10.0, 100.0)   # C is chosen once on the exact states, then fixed
FIT_STEPS = 100                   # Adam steps per image during the search
FIT_LR = 0.15
FIT_RESTARTS = 1

# What NSGA-II maximises (second objective is always two-qubit gates):
#   "retention" : class signal kept (the method of this project)
#   "fidelity"  : mean fidelity to the target states (the conventional objective;
#                 used by objective_study.py as the comparison arm)
# Both are computed and logged for every genome either way.
OBJECTIVE = "retention"

# ── Search space (genome format identical to the benchmark repo) ───
# A genome alternates rotation and entangler layers:
#   rot, ent, rot, ent, ..., rot       (one "block" = one ent + one rot)
ROT_GATES = ["RY"]                # add "RYRZ" / "RXRYRZ" for complex circuits (slower)
ENTANGLERS = ["CNOT_ring", "CZ_ring"]
STRIDES = [1, 2, 3, 4]
STEPS = [1, 2]
OFFSETS = [0, 1]
MIN_BLOCKS = 1
MAX_BLOCKS = 8
INIT_MAX_BLOCKS = 5
MAX_TWO_QUBIT_GATES = 64

# ── NSGA-II (objectives: maximise retention, minimise two-qubit gates)
POP_SIZE = 24
N_GENERATIONS = 15
TOURNAMENT_K = 2
CROSSOVER_RATE = 0.5
MUTATION_RATE = 0.9
N_WORKERS = None                  # None = min(8, CPU count); 1 = no multiprocessing

# ── Selection (select_encoder.py) ─────────────────────────────────
# The final Pareto front is re-evaluated on a DIFFERENT batch (val split
# plus train images not used in the search) with a bigger fitting budget.
# The cheapest genome within SELECT_TOLERANCE of the best retention wins.
SELECT_N_IMAGES = 600
SELECT_FIT_STEPS = 250
SELECT_FIT_RESTARTS = 2
SELECT_TOLERANCE = 0.01                   # for OBJECTIVE = "retention"
SELECT_TOLERANCE_FIDELITY = 0.01          # for OBJECTIVE = "fidelity"
