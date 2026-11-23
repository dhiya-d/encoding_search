# ----------------------------------------------------------------
# encoding_genome.py  —  the searchable data-encoding circuits.
#
# A genome is {"n_features": d, "layers": [layer, ...]}; layers act on
# all qubits in order:
#   {"kind": "enc", "axis": "X|Y|Z", "scale": s, "start": k}
#       qubit q gets exp(-i s x_{(k+q) mod d} P/2)
#   {"kind": "zz", "scale": s, "stride": ., "step": ., "offset": ., "start": k}
#       exp(-i s x_f x_g Z Z/2) on each hierarqcal Qcycle pair (a, b), with
#       f = (k+a) mod d, g = (k+b) mod d
#   {"kind": "ent", "gate": "CNOT|CZ", "stride": ., "step": ., "offset": .}
#   {"kind": "H"}
#
# Following Schuld, Sweke & Meyer (2021), a model built on this state is a
# Fourier series in the features whose frequencies come only from the
# encoding gates: a feature encoded by Pauli rotations with scales
# s_1..s_m has spectrum {sum(+-s_i/2) - sum(+-s_i/2)}, degree sum(s_i).
# Repeating a feature serially or in parallel raises its degree; the
# fixed gates (H, CNOT, CZ) and any trainable blocks only shape the
# coefficients. ZZ terms add products x_f x_g, which are not covered by
# this per-feature spectrum.
# ----------------------------------------------------------------
import json
from functools import lru_cache

import numpy as np
from hierarqcal import Qcycle, Qunitary

import config as C
from qsim import Circuit


def _noop(bits, symbols=None):
    """Placeholder gate for hierarqcal motifs; only the connectivity is used."""
    pass


@lru_cache(maxsize=None)
def edges(stride, step, offset, arity, n):
    """Qubit tuples of a hierarqcal Qcycle motif (stride/step/offset) on n qubits."""
    m = Qcycle(stride=stride, step=step, offset=offset,
               mapping=Qunitary(_noop, n_symbols=0, arity=arity))
    m(list(range(n)))
    return tuple(tuple(int(q) for q in e) for e in m.E)


def key(g):
    """Canonical string of a genome (used for caching and duplicate detection)."""
    return json.dumps(g, sort_keys=True)


def copy(g):
    """Deep copy of a genome."""
    return json.loads(json.dumps(g))


# ----------------------------------------------------------------
# Compilation
# ----------------------------------------------------------------

def encoding_ops(g, n=C.N_QUBITS):
    """Translate a Fourier-encoding genome into qsim operations. An 'enc' layer puts
    feature (start + q) mod d on qubit q with angle scale * x; 'zz' layers encode
    feature products on the edges of a ring; 'ent' layers add fixed CNOT/CZ rings.
    """
    d = g["n_features"]
    ops = []
    for L in g["layers"]:
        k = L["kind"]
        if k == "H":
            ops += [("H", q) for q in range(n)]
        elif k == "enc":
            ops += [("E", L["axis"], q, (L["start"] + q) % d, float(L["scale"])) for q in range(n)]
        elif k == "zz":
            for a, b in edges(L["stride"], L["step"], L["offset"], 2, n):
                f, h = (L["start"] + a) % d, (L["start"] + b) % d
                ops.append(("ZZ", a, b, f, h, float(L["scale"])))
        elif k == "ent":
            ops += [(L["gate"], a, b) for a, b in edges(L["stride"], L["step"], L["offset"], 2, n)]
        else:
            raise ValueError(f"unknown layer kind {k!r}")
    return ops


def encoding_circuit(g):
    """Simulator for the encoding part of a genome (no trainable gates)."""
    return Circuit(encoding_ops(g))


def two_qubit_cost(g, n=C.N_QUBITS):
    """Number of two-qubit gates of a genome (a ZZ rotation counts as two CNOTs)."""
    cost = 0
    for L in g["layers"]:
        if L["kind"] == "ent":
            cost += len(edges(L["stride"], L["step"], L["offset"], 2, n))
        elif L["kind"] == "zz":
            cost += 2 * len(edges(L["stride"], L["step"], L["offset"], 2, n))
    return cost


# ----------------------------------------------------------------
# Fourier spectrum (Schuld, Sweke & Meyer 2021, Eq. 10)
# ----------------------------------------------------------------

def feature_scales(g, n=C.N_QUBITS):
    """For every feature, the list of scales with which it is encoded by single-qubit rotations."""
    scales = {f: [] for f in range(g["n_features"])}
    for op in encoding_ops(g, n):
        if op[0] == "E":
            scales[op[3]].append(op[4])
    return scales


def spectrum_1d(scales):
    """Frequencies accessible to one feature encoded with the given scales: all differences
    of sums of +-s/2 (Schuld, Sweke & Meyer 2021).
    """
    sums = {0.0}
    for s in scales:
        sums = {round(a + e, 6) for a in sums for e in (-s / 2, s / 2)}
    return sorted({round(a - b, 6) for a in sums for b in sums})


def spectrum_summary(g):
    """Per-feature degree and spectrum size, encoded features and whether the genome has
    feature-product (ZZ) terms.
    """
    sc = feature_scales(g)
    degrees = {f: float(sum(v)) for f, v in sc.items()}
    sizes = {f: (len(spectrum_1d(v)) - 1) // 2 for f, v in sc.items()}
    encoded = [f for f, v in sc.items() if v]
    has_zz = any(L["kind"] == "zz" for L in g["layers"])
    return {
        "n_features": g["n_features"],
        "features_encoded": len(encoded),
        "mean_degree": float(np.mean([degrees[f] for f in encoded])) if encoded else 0.0,
        "max_degree": max(degrees.values()) if encoded else 0.0,
        "mean_spectrum_size": float(np.mean([sizes[f] for f in encoded])) if encoded else 0.0,
        "per_feature_degree": degrees,
        "has_zz_products": has_zz,
    }


def frequency_sets(g):
    """Per-feature frequency sets (used by the classical RFF baseline)."""
    return {f: spectrum_1d(v) for f, v in feature_scales(g).items()}


# ----------------------------------------------------------------
# Reference encodings from the literature (seeded into the search and
# always trained downstream for comparison)
# ----------------------------------------------------------------

def references():
    """Literature reference encodings used as baselines (angle encoding, its entangled
    variant, re-uploading, IQP-style ZZ maps; see the comments).
    """
    ring = {"kind": "ent", "gate": "CNOT", "stride": 1, "step": 1, "offset": 0}
    return {
        # one Pauli rotation per feature: a degree-1 Fourier series (Schuld et al. Sec. III A)
        "ref_angle_RY_d8": {"n_features": 8, "layers": [
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}]},
        # same with entanglement: coefficients change, spectrum does not
        "ref_angle_RY_ring_d8": {"n_features": 8, "layers": [
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}, ring]},
        # serial repetition x3 on 8 features: degree 3 per feature (Sec. III B)
        "ref_reupload3_d8": {"n_features": 8, "layers": [
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}, ring,
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}, ring,
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}]},
        # parallel repetition: 4 features, each on two qubits, degree 2
        "ref_parallel_d4": {"n_features": 4, "layers": [
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}, ring]},
        # 16 features over two layers, degree 1 each
        "ref_two_layer_d16": {"n_features": 16, "layers": [
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}, ring,
            {"kind": "enc", "axis": "Y", "scale": 1.0, "start": 8}, ring]},
        # IQP / Havlicek-type ZZ feature map, two repetitions
        "ref_iqp_zz_d8": {"n_features": 8, "layers": [
            {"kind": "H"}, {"kind": "enc", "axis": "Z", "scale": 1.0, "start": 0},
            {"kind": "zz", "scale": 1.0, "stride": 1, "step": 1, "offset": 0, "start": 0},
            {"kind": "H"}, {"kind": "enc", "axis": "Z", "scale": 1.0, "start": 0},
            {"kind": "zz", "scale": 1.0, "stride": 1, "step": 1, "offset": 0, "start": 0}]},
    }


# ----------------------------------------------------------------
# Variation
# ----------------------------------------------------------------

def _layer(kind, rng, d):
    """Random layer of the given kind (all parameters drawn from the search space in config.py)."""
    if kind == "H":
        return {"kind": "H"}
    if kind == "enc":
        return {"kind": "enc", "axis": str(rng.choice(C.ENC_AXES)),
                "scale": float(rng.choice(C.ENC_SCALES)), "start": int(rng.integers(0, d))}
    if kind == "zz":
        return {"kind": "zz", "scale": float(rng.choice(C.ZZ_SCALES)),
                "stride": int(rng.choice(C.STRIDES)), "step": int(rng.choice(C.STEPS)),
                "offset": int(rng.choice(C.OFFSETS)), "start": int(rng.integers(0, d))}
    return {"kind": "ent", "gate": str(rng.choice(C.ENTANGLERS)),
            "stride": int(rng.choice(C.STRIDES)), "step": int(rng.choice(C.STEPS)),
            "offset": int(rng.choice(C.OFFSETS))}


def _random_kind(rng):
    """Random layer kind, drawn with config.LAYER_WEIGHTS."""
    kinds = list(C.LAYER_WEIGHTS)
    w = np.array([C.LAYER_WEIGHTS[k] for k in kinds], float)
    return str(rng.choice(kinds, p=w / w.sum()))


def random_genome(rng):
    """Random valid genome: one encoding layer plus random further layers, in random order."""
    d = int(rng.choice(C.FEATURE_COUNTS))
    n_layers = int(rng.integers(max(C.MIN_LAYERS, 1), C.INIT_MAX_LAYERS + 1))
    layers = [_layer("enc", rng, d)]
    while len(layers) < n_layers:
        layers.append(_layer(_random_kind(rng), rng, d))
    rng.shuffle(layers)
    return {"n_features": d, "layers": layers}


def valid(g):
    """True if the genome respects the search-space limits (layers, features, two-qubit
    cost) and encodes data at least once.
    """
    L = g["layers"]
    return (C.MIN_LAYERS <= len(L) <= C.MAX_LAYERS
            and g["n_features"] in C.FEATURE_COUNTS
            and any(l["kind"] in ("enc", "zz") for l in L)
            and all(l.get("start", 0) < g["n_features"] for l in L)
            and two_qubit_cost(g) <= C.MAX_TWO_QUBIT_COST)


def _refit_starts(g):
    """Wrap every feature start index into range after n_features changed."""
    d = g["n_features"]
    for l in g["layers"]:
        if "start" in l:
            l["start"] %= d
    return g


def mutate(g, rng):
    """Random mutation: add, remove, modify or swap a layer, or change the number of features."""
    g = copy(g)
    L = g["layers"]
    ops = ["add", "remove", "modify", "modify", "modify", "swap", "features"]
    op = str(rng.choice(ops))
    if not L:
        op = "add"
    if op == "add" and len(L) < C.MAX_LAYERS:
        L.insert(int(rng.integers(0, len(L) + 1)), _layer(_random_kind(rng), rng, g["n_features"]))
    elif op == "remove" and len(L) > C.MIN_LAYERS:
        L.pop(int(rng.integers(0, len(L))))
    elif op == "swap" and len(L) >= 2:
        i, j = rng.choice(len(L), 2, replace=False)
        L[i], L[j] = L[j], L[i]
    elif op == "features":
        g["n_features"] = int(rng.choice(C.FEATURE_COUNTS))
        _refit_starts(g)
    else:
        i = int(rng.integers(0, len(L)))
        old = L[i]
        fresh = _layer(old["kind"], rng, g["n_features"])
        fields = [k for k in fresh if k != "kind"]
        if not fields or rng.random() < 0.2:
            L[i] = _layer(_random_kind(rng), rng, g["n_features"])
        else:
            f = str(rng.choice(fields))
            old[f] = fresh[f]
    return g


def crossover(a, b, rng):
    """One-point crossover: prefix of parent a + suffix of parent b."""
    i = int(rng.integers(0, len(a["layers"]) + 1))
    j = int(rng.integers(0, len(b["layers"]) + 1))
    child = {"n_features": a["n_features"] if rng.random() < 0.5 else b["n_features"],
             "layers": copy(a["layers"][:i] + b["layers"][j:])[:C.MAX_LAYERS]}
    return _refit_starts(child)


def describe(g):
    """Compact human-readable description of a genome (one token per layer)."""
    parts = []
    for l in g["layers"]:
        k = l["kind"]
        if k == "enc":
            parts.append(f"R{l['axis']}({l['scale']:g}x[{l['start']}+q])")
        elif k == "zz":
            parts.append(f"ZZ({l['scale']:g},s{l['stride']}/t{l['step']}/o{l['offset']})")
        elif k == "ent":
            parts.append(f"{l['gate']}(s{l['stride']}/t{l['step']}/o{l['offset']})")
        else:
            parts.append("H")
    return f"d={g['n_features']}: " + " > ".join(parts)
