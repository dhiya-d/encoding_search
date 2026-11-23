"""Configuration for the Frobenius-measure encoding search on BreastMNIST.

Differences from the RetinaMNIST version:
  * BreastMNIST is grayscale 28x28, so `channel` is a no-op (kept only for
    interface parity -- there is only one plane to select).
  * Labels are already binary (0=malignant, 1=benign), so there is no ordinal
    grade to merge. `merge` keeps a single "none" scheme for the same reason.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Two-qubit gates available to the genome.
# 'cry' is parameterised (one trainable angle per gate); 'cnot'/'cz' are not.
ENTANGLER_POOL = ("cnot", "cz", "cry")

MERGE_SCHEMES = {
    "none": None,
}

# Linear-probe baseline on raw amplitude targets (16px, TEST split), as
# measured by evaluate.py. Filled in after the first run -- reporting only,
# nothing in the pipeline reads this dict.
LINEAR_BASELINES: dict = {}


@dataclass
class DataConfig:
    data_dir: str = "breastmnist"
    channel: str = "gray"          # gray | green | rgb (breast is single-plane either way)
    size: int = 16                 # 16x16 -> 256 amplitudes -> 8 qubits
    center: bool = True            # subtract train-mean image before L2-normalise
    merge: str = "none"            # only scheme available for a binary task


@dataclass
class SearchConfig:
    # ---- the switch --------------------------------------------------------
    pareto_front: bool = True
    gate_penalty: float = 0.004    # only used when pareto_front=False
    knee_tolerance: float = 0.015  # only used when pareto_front=True

    # ---- GA ---------------------------------------------------------------
    pop: int = 24
    gen: int = 12
    p_crossover: float = 0.7
    min_gates: int = 1
    max_layers: int = 10
    elite: int = 2

    # ---- per-image angle fitting (AAE) -------------------------------------
    fit_steps: int = 150
    fit_lr: float = 0.08
    batch_train: int = 256
    batch_final: int = 512
    restarts: int = 1

    # ---- misc -------------------------------------------------------------
    seed: int = 0
    device: str = "auto"
    outdir: str = "experiments/frob_search"

    def to_dict(self):
        return asdict(self)


@dataclass
class RunConfig:
    data: DataConfig = field(default_factory=DataConfig)
    search: SearchConfig = field(default_factory=SearchConfig)

    def to_dict(self):
        return {"data": asdict(self.data), "search": asdict(self.search)}
