"""Encoding-circuit genome.

Layers:
  {"type": "rot"}                                  -> RY on every qubit  (n angles)
  {"type": "ent", "pairs": [(c, t, gate), ...]}    -> two-qubit gates

Gates: 'cnot' and 'cz' are structural (no parameters); 'cry' is a controlled-RY
and contributes ONE trainable angle each. All three count toward the two-qubit
gate cost, which is the hardware objective.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from config import ENTANGLER_POOL


@dataclass
class Genome:
    n_qubits: int
    layers: list = field(default_factory=list)

    # ---- structure --------------------------------------------------------
    def n_rot_layers(self) -> int:
        return sum(1 for l in self.layers if l["type"] == "rot")

    def n_cry(self) -> int:
        return sum(1 for l in self.layers if l["type"] == "ent"
                   for p in l["pairs"] if p[2] == "cry")

    def n_params(self) -> int:
        """RY layers contribute n angles each; every CRY contributes one."""
        return self.n_qubits * self.n_rot_layers() + self.n_cry()

    def two_qubit_gates(self) -> int:
        return sum(len(l["pairs"]) for l in self.layers if l["type"] == "ent")

    def depth(self) -> int:
        return len(self.layers)

    def summary(self) -> str:
        return (f"layers={self.depth()} params={self.n_params()} "
                f"2q={self.two_qubit_gates()} (cry={self.n_cry()})")

    def signature(self) -> str:
        out = []
        for l in self.layers:
            if l["type"] == "rot":
                out.append("R")
            else:
                out.append("E" + ",".join(f"{c}>{t}:{g}" for c, t, g in l["pairs"]))
        return "|".join(out)

    def clone(self) -> "Genome":
        return Genome(self.n_qubits, copy.deepcopy(self.layers))

    def to_dict(self) -> dict:
        return {"n_qubits": self.n_qubits, "layers": self.layers}

    @staticmethod
    def from_dict(d: dict) -> "Genome":
        layers = []
        for l in d["layers"]:
            if l["type"] == "rot":
                layers.append({"type": "rot"})
            else:
                layers.append({"type": "ent", "pairs": [tuple(p) for p in l["pairs"]]})
        return Genome(d["n_qubits"], layers)


# ---------------------------------------------------------------------------
def _rand_pair(n: int, rng: random.Random) -> tuple:
    c = rng.randrange(n)
    t = rng.randrange(n)
    while t == c:
        t = rng.randrange(n)
    return (c, t, rng.choice(ENTANGLER_POOL))


def _rand_ent_layer(n: int, rng: random.Random) -> dict:
    k = rng.randint(1, max(1, n // 2))
    return {"type": "ent", "pairs": [_rand_pair(n, rng) for _ in range(k)]}


def enforce_min(g: Genome, min_gates: int, rng: random.Random) -> Genome:
    """Guarantee >=1 rot layer and >=min_gates two-qubit gates."""
    g = g.clone()
    if g.n_rot_layers() == 0:
        g.layers.insert(0, {"type": "rot"})
    while g.two_qubit_gates() < min_gates:
        ent = [l for l in g.layers if l["type"] == "ent"]
        if ent:
            rng.choice(ent)["pairs"].append(_rand_pair(g.n_qubits, rng))
        else:
            g.layers.append(_rand_ent_layer(g.n_qubits, rng))
    return g


def random_genome(n: int, rng: random.Random, min_layers=2, max_layers=6,
                  min_gates=1) -> Genome:
    layers = []
    for i in range(rng.randint(min_layers, max_layers)):
        layers.append({"type": "rot"} if i % 2 == 0 else _rand_ent_layer(n, rng))
    return enforce_min(Genome(n, layers), min_gates, rng)


def mutate(g: Genome, rng: random.Random, max_layers=10, min_gates=1,
           p_add=0.3, p_del=0.25) -> Genome:
    g = g.clone()
    n = g.n_qubits
    r = rng.random()

    if r < p_add and len(g.layers) < max_layers:
        layer = {"type": "rot"} if rng.random() < 0.5 else _rand_ent_layer(n, rng)
        g.layers.insert(rng.randint(0, len(g.layers)), layer)

    elif r < p_add + p_del and len(g.layers) > 1:
        i = rng.randrange(len(g.layers))
        if not (g.layers[i]["type"] == "rot" and g.n_rot_layers() == 1):
            g.layers.pop(i)

    else:
        ent_idx = [i for i, l in enumerate(g.layers) if l["type"] == "ent"]
        if ent_idx:
            pairs = g.layers[rng.choice(ent_idx)]["pairs"]
            op = rng.random()
            if op < 0.35 and len(pairs) < n:
                pairs.append(_rand_pair(n, rng))
            elif op < 0.6 and len(pairs) > 1:
                pairs.pop(rng.randrange(len(pairs)))
            elif op < 0.8:                      # retype the gate (cnot/cz/cry)
                j = rng.randrange(len(pairs))
                c, t, _ = pairs[j]
                pairs[j] = (c, t, rng.choice(ENTANGLER_POOL))
            else:                                # rewire
                pairs[rng.randrange(len(pairs))] = _rand_pair(n, rng)
        else:
            g.layers.append(_rand_ent_layer(n, rng))

    return enforce_min(g, min_gates, rng)


def crossover(a: Genome, b: Genome, rng: random.Random, min_gates=1) -> Genome:
    if not a.layers or not b.layers:
        return a.clone()
    ca = rng.randint(1, len(a.layers))
    cb = rng.randint(0, len(b.layers) - 1)
    child = Genome(a.n_qubits, copy.deepcopy(a.layers[:ca]) + copy.deepcopy(b.layers[cb:]))
    return enforce_min(child, min_gates, rng)
