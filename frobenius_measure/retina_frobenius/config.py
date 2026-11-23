"""Configuration for the Frobenius-measure encoding search on RetinaMNIST.

Restricted to the `retina4` grouping only (4 classes: {0},{1},{2},{3,4},
chance 0.250) -- retina2 (binary) and none (5-class) are intentionally not
available here. RetinaMNIST is RGB, so `channel` genuinely matters
(gray / green / rgb).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# Two-qubit gates available to the genome.
# 'cry' is parameterised (one trainable angle per gate); 'cnot'/'cz' are not.
ENTANGLER_POOL = ("cnot", "cz", "cry")

MERGE_SCHEMES = {
    "retina4": {0: 0, 1: 1, 2: 2, 3: 3, 4: 3},
}

# Linear-probe baselines measured on the raw amplitude targets (16px, TEST split).
# Used only for reporting context -- the scripts recompute the probe themselves.
LINEAR_BASELINES = {
    ("retina4", "gray"): 0.4557, ("retina4", "rgb"): 0.4855,
}


@dataclass
class DataConfig:
    data_dir: str = "retinamnist"
    channel: str = "gray"          # gray | green | rgb
    size: int = 16                 # 16x16 -> 256 amplitudes -> 8 qubits (gray/green)
    center: bool = True            # subtract train-mean image before L2-normalise
    merge: str = "retina4"         # only scheme available -- see module docstring


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
