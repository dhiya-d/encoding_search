"""The two selection strategies behind the `pareto_front` switch.

pareto_front=True   -> NSGA-II on (frobenius [min], gates [min]); knee chosen after.
pareto_front=False  -> single-objective GA on  frobenius + gate_penalty * gates.

Both minimise, so internally we work with raw values and compare with '<'.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

from genome import Genome, crossover, mutate, random_genome


@dataclass
class Individual:
    genome: Genome
    frob: float = float("inf")     # mean Frobenius distance (minimise)
    gates: int = 0                 # two-qubit gate count (minimise)
    score: float = float("inf")    # single-objective composite (minimise)
    rank: int = 0
    crowd: float = 0.0
    info: dict = field(default_factory=dict)

    def objectives(self):
        return (self.frob, float(self.gates))


# ---------------------------------------------------------------------------
# NSGA-II (both objectives minimised)
# ---------------------------------------------------------------------------
def dominates(a, b) -> bool:
    return all(x <= y for x, y in zip(a, b)) and any(x < y for x, y in zip(a, b))


def fast_non_dominated_sort(pop):
    fronts, S, ndom = [[]], [[] for _ in pop], [0] * len(pop)
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p].objectives(), pop[q].objectives()):
                S[p].append(q)
            elif dominates(pop[q].objectives(), pop[p].objectives()):
                ndom[p] += 1
        if ndom[p] == 0:
            pop[p].rank = 0
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S[p]:
                ndom[q] -= 1
                if ndom[q] == 0:
                    pop[q].rank = i + 1
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    fronts.pop()
    return fronts


def crowding_distance(pop, front):
    if not front:
        return
    for i in front:
        pop[i].crowd = 0.0
    for obj in range(2):
        order = sorted(front, key=lambda i: pop[i].objectives()[obj])
        pop[order[0]].crowd = float("inf")
        pop[order[-1]].crowd = float("inf")
        lo, hi = pop[order[0]].objectives()[obj], pop[order[-1]].objectives()[obj]
        span = (hi - lo) or 1.0
        for k in range(1, len(order) - 1):
            pop[order[k]].crowd += (pop[order[k + 1]].objectives()[obj]
                                    - pop[order[k - 1]].objectives()[obj]) / span


def pareto_front_of(pop):
    f = fast_non_dominated_sort(pop)
    return [pop[i] for i in f[0]] if f else []


def select_next_pareto(combined, pop_size):
    fronts = fast_non_dominated_sort(combined)
    nxt, i = [], 0
    while i < len(fronts) and len(nxt) + len(fronts[i]) <= pop_size:
        crowding_distance(combined, fronts[i])
        nxt.extend(combined[j] for j in fronts[i])
        i += 1
    if len(nxt) < pop_size and i < len(fronts):
        crowding_distance(combined, fronts[i])
        rest = sorted(fronts[i], key=lambda j: combined[j].crowd, reverse=True)
        nxt.extend(combined[j] for j in rest[:pop_size - len(nxt)])
    return nxt


def knee_of(front, tolerance: float):
    """Cheapest genome whose Frobenius is within `tolerance` of the front best."""
    best = min(i.frob for i in front)
    ok = [i for i in front if i.frob <= best + tolerance]
    return min(ok, key=lambda i: i.gates)


# ---------------------------------------------------------------------------
# single-objective
# ---------------------------------------------------------------------------
def composite_score(frob: float, gates: int, gate_penalty: float) -> float:
    return frob + gate_penalty * gates


def select_next_single(combined, pop_size):
    return sorted(combined, key=lambda i: i.score)[:pop_size]


# ---------------------------------------------------------------------------
# shared variation
# ---------------------------------------------------------------------------
def _tournament(pop, rng, use_pareto: bool):
    a, b = rng.sample(pop, 2)
    if use_pareto:
        if a.rank != b.rank:
            return a if a.rank < b.rank else b
        return a if a.crowd > b.crowd else b
    return a if a.score < b.score else b


def make_offspring(pop, rng, cfg, use_pareto: bool):
    kids = []
    while len(kids) < len(pop):
        p1 = _tournament(pop, rng, use_pareto)
        if rng.random() < cfg.p_crossover:
            p2 = _tournament(pop, rng, use_pareto)
            child = crossover(p1.genome, p2.genome, rng, min_gates=cfg.min_gates)
        else:
            child = p1.genome.clone()
        child = mutate(child, rng, max_layers=cfg.max_layers, min_gates=cfg.min_gates)
        kids.append(Individual(genome=child))
    return kids


def init_population(n_qubits, cfg, rng):
    return [Individual(genome=random_genome(n_qubits, rng, min_gates=cfg.min_gates,
                                            max_layers=min(6, cfg.max_layers)))
            for _ in range(cfg.pop)]


def dedup(inds):
    seen, out = set(), []
    for i in inds:
        s = i.genome.signature()
        if s not in seen:
            seen.add(s)
            out.append(i)
    return out
