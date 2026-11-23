"""QCNN architecture genome.

A genome is a list of 1..MAX_LAYERS layers. Each layer is

    {"conv": <conv ansatz name>, "pattern": <linear|ring|brick>, "pool": 0|1|2,
     "pool_type": <pool_U|pool_ansatz>}

`pool` is the number of halvings performed at the end of that layer. Starting
from 8 active qubits and requiring exactly TARGET_QUBITS surviving qubits at
readout (set via `configure()` / --target-qubits), the halvings must satisfy

    sum(layer["pool"] for layer in genome) == REQUIRED_HALVINGS

A repair step enforces this, which is what lets the layer count range over
1..6 while the readout width stays fixed: a single-layer genome performs all
the halvings inside that layer, a six-layer genome spreads them out and uses
the remaining layers as convolution-only stages.

For DermaMNIST (7 classes), --target-qubits 4 gives REQUIRED_HALVINGS=1
(8 -> 4) and 16 measurement outcomes -- 7 get assigned a label, 9 are unused
(7 isn't a power of two, so no target_qubits value gives an exact fit).
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from ansatz import CONV_PARAMS, CONV_TYPES, PATTERNS, POOL_PARAMS, POOL_TYPES, conv_pairs, pool_pairs

MIN_LAYERS = 1
MAX_LAYERS = 6
N_QUBITS = 8
TARGET_QUBITS = 2
REQUIRED_HALVINGS = 2          # 8 -> 4 -> 2
FREE_POOLING = False           # if True: pooling is unconstrained and the readout
                               # takes the first TARGET_QUBITS of whatever remains


def configure(n_qubits=8, target_qubits=2, min_layers=1, max_layers=6,
              free_pooling=False):
    """Set register size, readout width, layer bounds and the pooling regime.

    free_pooling=False : the repair step forces the halvings to total exactly
        log2(n_qubits/target_qubits), so every genome reduces to precisely
        target_qubits active wires.
    free_pooling=True  : pooling is unconstrained (a genome may pool not at all);
        the readout measures the first target_qubits of whatever survives and
        traces out the rest at measurement.
    """
    global N_QUBITS, TARGET_QUBITS, MIN_LAYERS, MAX_LAYERS, REQUIRED_HALVINGS, FREE_POOLING
    import math
    if target_qubits < 1 or n_qubits % target_qubits != 0:
        raise ValueError("target_qubits must divide n_qubits")
    h = math.log2(n_qubits / target_qubits)
    if abs(h - round(h)) > 1e-9:
        raise ValueError("n_qubits / target_qubits must be a power of two")
    N_QUBITS, TARGET_QUBITS = n_qubits, target_qubits
    MIN_LAYERS, MAX_LAYERS = min_layers, max_layers
    REQUIRED_HALVINGS = int(round(h))
    FREE_POOLING = free_pooling
    return REQUIRED_HALVINGS


def n_outcomes() -> int:
    return 1 << TARGET_QUBITS


# ---------------------------------------------------------------------------
@dataclass
class QCNNGenome:
    n_qubits: int = N_QUBITS
    layers: list = field(default_factory=list)

    # -- structure ---------------------------------------------------------
    def total_pool(self) -> int:
        return sum(l["pool"] for l in self.layers)

    def schedule(self):
        """Walk the genome, returning per-layer (active_before, conv_pairs,
        pool_stages, active_after) and the final surviving wires."""
        active = list(range(self.n_qubits))
        out = []
        for l in self.layers:
            pairs = conv_pairs(active, l["pattern"])
            stages = []
            for _ in range(l["pool"]):
                if len(active) <= TARGET_QUBITS:
                    break
                pp = pool_pairs(active)
                stages.append(pp)
                active = [b for (_, b) in pp]
            out.append({"conv_pairs": pairs, "pool_stages": stages,
                        "active_after": list(active)})
        return out, active

    def n_params(self) -> int:
        sched, _ = self.schedule()
        tot = 0
        for l, s in zip(self.layers, sched):
            tot += CONV_PARAMS[l["conv"]] * len(s["conv_pairs"])
            for pp in s["pool_stages"]:
                tot += POOL_PARAMS[l["pool_type"]] * len(pp)
        return tot

    def two_qubit_gates(self) -> int:
        """Entangling-gate count, used as the complexity objective."""
        ARITY = {"U_TTN": 1, "U_5": 2, "U_6": 2, "U_9": 1, "U_13": 2, "U_14": 2,
                 "U_15": 2, "U_SO4": 2, "U_SU4": 3, "U_gate": 1}
        PARITY = {"pool_U": 2, "pool_ansatz": 2}
        sched, _ = self.schedule()
        tot = 0
        for l, s in zip(self.layers, sched):
            tot += ARITY[l["conv"]] * len(s["conv_pairs"])
            for pp in s["pool_stages"]:
                tot += PARITY[l["pool_type"]] * len(pp)
        return tot

    def final_wires(self):
        _, active = self.schedule()
        return active

    def readout_wires(self):
        """Wires actually measured: the first TARGET_QUBITS of the survivors.

        Under strict reduction the survivors are exactly TARGET_QUBITS, so this
        is the whole set. Under FREE_POOLING the circuit may leave more qubits
        active, and everything past the first TARGET_QUBITS is traced out at
        measurement rather than pooled away.
        """
        return self.final_wires()[:TARGET_QUBITS]

    def is_valid(self) -> bool:
        if not (MIN_LAYERS <= len(self.layers) <= MAX_LAYERS):
            return False
        if len(self.final_wires()) < TARGET_QUBITS:
            return False
        if not FREE_POOLING and len(self.final_wires()) != TARGET_QUBITS:
            return False
        # every layer must actually do some convolution work
        sched, _ = self.schedule()
        return all(len(s["conv_pairs"]) > 0 for s in sched)

    def summary(self) -> str:
        chain = [self.n_qubits]
        sched, _ = self.schedule()
        for s in sched:
            chain.append(len(s["active_after"]))
        convs = "+".join(l["conv"] for l in self.layers)
        meas = f" meas{len(self.readout_wires())}/{len(self.final_wires())}" \
            if len(self.final_wires()) != TARGET_QUBITS else ""
        return (f"{len(self.layers)}L [{'→'.join(map(str, chain))}]{meas} {convs} "
                f"| {self.n_params()} params | {self.two_qubit_gates()} 2q")

    def signature(self) -> str:
        return "|".join(f"{l['conv']}/{l['pattern']}/{l['pool']}/{l['pool_type']}"
                        for l in self.layers)

    def clone(self):
        return QCNNGenome(self.n_qubits, copy.deepcopy(self.layers))

    def to_dict(self):
        return {"n_qubits": self.n_qubits, "layers": self.layers}

    @staticmethod
    def from_dict(d):
        return QCNNGenome(d["n_qubits"], copy.deepcopy(d["layers"]))


# ---------------------------------------------------------------------------
def _rand_layer(rng, pool=0):
    return {"conv": rng.choice(CONV_TYPES), "pattern": rng.choice(PATTERNS),
            "pool": pool, "pool_type": rng.choice(POOL_TYPES)}


def repair(g: QCNNGenome, rng: random.Random) -> QCNNGenome:
    """Enforce the layer bounds and the pooling regime.

    Under forced reduction the halvings are made to total exactly
    REQUIRED_HALVINGS. Under FREE_POOLING nothing is enforced beyond the clamp:
    the schedule already refuses to reduce below TARGET_QUBITS.
    """
    g = g.clone()
    if not g.layers:
        g.layers = [_rand_layer(rng)]
    if len(g.layers) > MAX_LAYERS:
        g.layers = g.layers[:MAX_LAYERS]
    for l in g.layers:
        l["pool"] = max(0, min(REQUIRED_HALVINGS, int(l.get("pool", 0))))

    if FREE_POOLING:
        return g

    while g.total_pool() > REQUIRED_HALVINGS:
        for l in g.layers:
            if l["pool"] > 0:
                l["pool"] -= 1
                break
    while g.total_pool() < REQUIRED_HALVINGS:
        need = REQUIRED_HALVINGS - g.total_pool()
        cand = [l for l in reversed(g.layers) if l["pool"] < REQUIRED_HALVINGS]
        if not cand:
            break
        cand[0]["pool"] = min(REQUIRED_HALVINGS, cand[0]["pool"] + need)
    return g


def random_genome(rng: random.Random, min_layers=None,
                  max_layers=None) -> QCNNGenome:
    lo = MIN_LAYERS if min_layers is None else min_layers
    hi = MAX_LAYERS if max_layers is None else max_layers
    k = rng.randint(lo, hi)
    layers = [_rand_layer(rng) for _ in range(k)]
    # Sample the pooling budget explicitly. Assigning pooling per layer instead
    # would make many-layer genomes saturate to TARGET_QUBITS almost always, so
    # the "free" regime would collapse back onto the forced one.
    budget = rng.randint(0, REQUIRED_HALVINGS) if FREE_POOLING else REQUIRED_HALVINGS
    for _ in range(budget):
        cand = [l for l in layers if l["pool"] < REQUIRED_HALVINGS]
        if not cand:
            break
        rng.choice(cand)["pool"] += 1
    return repair(QCNNGenome(N_QUBITS, layers), rng)


def mutate(g: QCNNGenome, rng: random.Random) -> QCNNGenome:
    g = g.clone()
    r = rng.random()
    if r < 0.22 and len(g.layers) < MAX_LAYERS:              # grow
        g.layers.insert(rng.randrange(len(g.layers) + 1), _rand_layer(rng))
    elif r < 0.40 and len(g.layers) > MIN_LAYERS:            # shrink
        g.layers.pop(rng.randrange(len(g.layers)))
    elif r < 0.62:                                            # retype conv
        rng.choice(g.layers)["conv"] = rng.choice(CONV_TYPES)
    elif r < 0.78:                                            # change pattern
        rng.choice(g.layers)["pattern"] = rng.choice(PATTERNS)
    elif r < 0.90:                                            # change pool type
        rng.choice(g.layers)["pool_type"] = rng.choice(POOL_TYPES)
    elif FREE_POOLING:                                        # add/remove pooling
        l = rng.choice(g.layers)
        l["pool"] = max(0, min(REQUIRED_HALVINGS, l["pool"] + rng.choice([-1, 1])))
    else:                                                     # move a halving
        src = [l for l in g.layers if l["pool"] > 0]
        if src and len(g.layers) > 1:
            a = rng.choice(src)
            b = rng.choice([l for l in g.layers if l is not a])
            a["pool"] -= 1
            b["pool"] = min(REQUIRED_HALVINGS, b["pool"] + 1)
    return repair(g, rng)


def crossover(a: QCNNGenome, b: QCNNGenome, rng: random.Random) -> QCNNGenome:
    ca = rng.randint(1, len(a.layers))
    cb = rng.randint(0, len(b.layers) - 1)
    child = QCNNGenome(a.n_qubits,
                       copy.deepcopy(a.layers[:ca]) + copy.deepcopy(b.layers[cb:]))
    return repair(child, rng)