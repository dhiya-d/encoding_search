"""Configuration for the Frobenius-measure encoding search on DermaMNIST.

DermaMNIST is RGB 28x28, 7 diagnostic classes (heavily imbalanced -- the
majority class is ~67% of train), no ordinal grade to merge -- `merge` has a
single "none" scheme, kept only for interface parity with the other configs.

7 classes don't divide the 8-qubit register as a clean power of two, so the
QCNN's readout (--target-qubits 4 -> 16 outcomes) has 9 unused outcomes; see
README.md for why 8 qubits / 16x16 was still chosen over a tighter-fitting
but much more heavily downsampled alternative.

`channel` defaults to "gray" to keep the register at 8 qubits, matching
retina_frobenius/breast_frobenius. DermaMNIST diagnosis genuinely leans on
colour (pigmentation patterns), so this is a real fidelity tradeoff, not a
free choice -- --channel rgb is a one-flag change if colour turns out to
matter a lot here, at the cost of a bigger register (768 amplitudes -> 10
qubits instead of 8).
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
    data_dir: str = "dermamnist"
    channel: str = "gray"          # gray | green | rgb -- see module docstring
    size: int = 16                 # 16x16 -> 256 amplitudes -> 8 qubits
    center: bool = True            # subtract train-mean image before L2-normalise
    merge: str = "none"            # only scheme available -- 7 classes, no merge


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
