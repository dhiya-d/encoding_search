# ----------------------------------------------------------------
# search_core.py  —  Shared machinery for encoding_search.py and
# select_encoder.py: data batches, retention fitness, a process pool,
# genome variation operators and NSGA-II.
# ----------------------------------------------------------------

import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import diag_common as D
import search_config as S
from aae_sim import AAECircuit, DEFAULT_GENOME, fit_states

VALID_MODES = ("uncentered", "centered")


# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------

def prepare_batches(dataset, mode, seed=None):
    """Search batch from TRAIN; selection batch from VAL plus train images
    not in the search batch (disjoint from the search batch). The test
    split is never touched here — it is reserved for diag_a."""
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")
    seed = S.SEED if seed is None else seed          # read at call time (--seed)
    tr_x, tr_y = D.load_split(S.DATA_ROOT, dataset, "train")
    va_x, va_y = D.load_split(S.DATA_ROOT, dataset, "val")
    tr_v = D.images_to_vectors(tr_x)
    mean, std = D.compute_stats(tr_v, mode)

    rng = np.random.default_rng(seed)
    s_idx = np.sort(D.stratified_sample(tr_y, S.SEARCH_N_IMAGES, rng))
    search = {"T": D.apply_stats(tr_v[s_idx], mean, std), "y": tr_y[s_idx],
              "train_idx": s_idx}

    rest = np.setdiff1d(np.arange(len(tr_y)), s_idx)
    pool_T = [D.apply_stats(tr_v[rest], mean, std)]
    pool_y = [tr_y[rest]]
    pool_src = [np.stack([np.zeros(len(rest), int), rest], 1)]
    if va_x is not None:
        va_v = D.images_to_vectors(va_x)
        pool_T.append(D.apply_stats(va_v, mean, std))
        pool_y.append(va_y)
        pool_src.append(np.stack([np.ones(len(va_y), int), np.arange(len(va_y))], 1))
    pT, py, psrc = np.concatenate(pool_T), np.concatenate(pool_y), np.concatenate(pool_src)
    sel_idx = D.stratified_sample(py, S.SELECT_N_IMAGES, np.random.default_rng(seed + 1))
    select = {"T": pT[sel_idx], "y": py[sel_idx],
              "source": [("train" if a == 0 else "val", int(b)) for a, b in psrc[sel_idx]]}
    return search, select


def ceiling(T, y, folds):
    """Exact-state ceiling: fidelity-kernel SVM on the exact targets, C chosen by CV. The
    chosen C is then fixed for all genomes.
    """
    K = D.fidelity_kernel(T, T)
    C, scores = D.choose_C(K, y, S.SVM_C_GRID, folds)
    return {"C": float(C), **scores[C],
            "all_C": {str(k): v for k, v in scores.items()},
            "kernel_stats": D.kernel_stats(K)}


# ----------------------------------------------------------------
# Fitness (runs inside worker processes)
# ----------------------------------------------------------------

_W = {}


def _init_worker(T, y, folds, C, ceiling_auc, steps, restarts, seed):
    """Worker-process setup: one BLAS thread per worker and the shared batch data."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _W.update(T=T, y=y, folds=folds, C=C, ceiling_auc=ceiling_auc,
              steps=steps, restarts=restarts, seed=seed)


def _evaluate(genome):
    """Fitness of one genome: fit every image of the batch, build the fidelity kernel of
    the fitted states and score it by CV; retention = CV AUC / ceiling AUC.
    """
    circ = AAECircuit(genome)
    fid, _, states = fit_states(circ, _W["T"], steps=_W["steps"], lr=S.FIT_LR,
                                restarts=_W["restarts"], seed=_W["seed"],
                                chunk=len(_W["T"]), verbose=False)
    K = D.fidelity_kernel(states, states)
    sc = D.cv_kernel_score(K, _W["y"], _W["C"], _W["folds"])
    return {"retention": sc["auc"] / _W["ceiling_auc"], "auc": sc["auc"],
            "auc_std": sc["auc_std"], "bal_acc": sc["bal_acc"],
            "fid_mean": float(fid.mean()), "fid_median": float(np.median(fid)),
            "n2q": circ.n_two_qubit, "n_params": circ.n_params}


class Evaluator:
    """Caches results by genome key and evaluates new genomes in parallel."""

    def __init__(self, T, y, folds, C, ceiling_auc, steps, restarts, seed, cache=None):
        """Start the worker pool (or run serially when N_WORKERS = 1); `cache` holds
        results of earlier evaluations.
        """
        self.cache = dict(cache or {})
        n = S.N_WORKERS or min(8, os.cpu_count() or 1)
        self.n_workers = n
        args = (T, y, folds, C, ceiling_auc, steps, restarts, seed)
        if n > 1:
            self.pool = ProcessPoolExecutor(n, initializer=_init_worker, initargs=args)
        else:
            self.pool = None
            _init_worker(*args)

    def evaluate(self, genomes):
        """Results for a list of genomes, evaluating only those not yet in the cache.
        Returns (results, number of new evaluations).
        """
        todo, seen = [], set()
        for g in genomes:
            k = key(g)
            if k not in self.cache and k not in seen:
                todo.append(g)
                seen.add(k)
        if todo:
            results = (list(self.pool.map(_evaluate, todo)) if self.pool
                       else [_evaluate(g) for g in todo])
            for g, r in zip(todo, results):
                self.cache[key(g)] = r
        return [self.cache[key(g)] for g in genomes], len(todo)

    def close(self):
        """Shut down the worker pool."""
        if self.pool:
            self.pool.shutdown()


# ----------------------------------------------------------------
# Genomes
# ----------------------------------------------------------------

def key(genome):
    """Canonical string of a genome (cache key)."""
    return json.dumps(genome, sort_keys=True)


def n_blocks(g):
    """Number of (entangler, rotation) blocks of a genome."""
    return (len(g) - 1) // 2


def n_two_qubit(g):
    """Number of two-qubit gates of a genome."""
    return AAECircuit(g).n_two_qubit


def valid(g):
    """True if the genome alternates rot/ent layers and respects the block and gate limits."""
    return (len(g) % 2 == 1 and S.MIN_BLOCKS <= n_blocks(g) <= S.MAX_BLOCKS
            and all(l["kind"] == ("rot" if i % 2 == 0 else "ent") for i, l in enumerate(g))
            and n_two_qubit(g) <= S.MAX_TWO_QUBIT_GATES)


def _rot(rng):
    """Random rotation layer."""
    return {"kind": "rot", "gate": str(rng.choice(S.ROT_GATES))}


def _ent(rng):
    """Random entangling layer (gate, stride, step, offset drawn from the search space)."""
    return {"kind": "ent", "gate": str(rng.choice(S.ENTANGLERS)),
            "stride": int(rng.choice(S.STRIDES)), "step": int(rng.choice(S.STEPS)),
            "offset": int(rng.choice(S.OFFSETS))}


def random_genome(rng):
    """Random genome with between MIN_BLOCKS and INIT_MAX_BLOCKS blocks."""
    g = [_rot(rng)]
    for _ in range(int(rng.integers(S.MIN_BLOCKS, S.INIT_MAX_BLOCKS + 1))):
        g += [_ent(rng), _rot(rng)]
    return g


def seed_genomes():
    """Structured starting points so the front covers the whole cost range."""
    out = []
    for b in (1, 2, 4, 6, 8):
        if S.MIN_BLOCKS <= b <= S.MAX_BLOCKS:
            g = [{"kind": "rot", "gate": S.ROT_GATES[0]}]
            for i in range(b):
                g += [{"kind": "ent", "gate": S.ENTANGLERS[0], "stride": 1 + (i % 2),
                       "step": 1, "offset": i % 2},
                      {"kind": "rot", "gate": S.ROT_GATES[0]}]
            out.append(g)
    if valid(DEFAULT_GENOME):
        out.append(json.loads(json.dumps(DEFAULT_GENOME)))
    return [g for g in out if valid(g)]


def mutate(g, rng):
    """Random mutation: add, remove or swap a block, or change an entangler (or rotation)."""
    g = json.loads(json.dumps(g))
    ops = ["add", "remove", "modify_ent", "modify_ent", "swap"]
    if len(S.ROT_GATES) > 1:
        ops.append("modify_rot")
    op = rng.choice(ops)
    b = n_blocks(g)
    if b == 0:
        op = "add"
    if op == "add" and b < S.MAX_BLOCKS:
        pos = 2 * int(rng.integers(0, b + 1)) + 1
        g[pos:pos] = [_ent(rng), _rot(rng)]
    elif op == "remove" and b > S.MIN_BLOCKS:
        k = int(rng.integers(0, b))
        del g[2 * k + 1:2 * k + 3]
    elif op == "swap" and b >= 2:
        i, j = rng.choice(b, 2, replace=False)
        g[2 * i + 1], g[2 * j + 1] = g[2 * j + 1], g[2 * i + 1]
    elif op == "modify_rot":
        g[2 * int(rng.integers(0, b + 1))] = _rot(rng)
    else:
        idx = 2 * int(rng.integers(0, b)) + 1
        field = rng.choice(["gate", "stride", "step", "offset"])
        fresh = _ent(rng)
        g[idx][field] = fresh[field]
    return g


def crossover(a, b, rng):
    """One-point crossover at block boundaries: blocks of a, then blocks of b."""
    ka = int(rng.integers(0, n_blocks(a) + 1))
    kb = int(rng.integers(0, n_blocks(b) + 1))
    child = a[:2 * ka + 1] + b[2 * kb + 1:]
    child = child[:2 * S.MAX_BLOCKS + 1]
    return json.loads(json.dumps(child))


# ----------------------------------------------------------------
# NSGA-II (objective vector: (retention, n2q); maximise, minimise)
# ----------------------------------------------------------------

def dominates(a, b):
    """Pareto dominance for objectives (maximise first, minimise second)."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def nondominated_sort(objs):
    """NSGA-II fast non-dominated sort: list of fronts (lists of indices), best first."""
    n = len(objs)
    S_, counts, fronts = [[] for _ in range(n)], [0] * n, [[]]
    for p in range(n):
        for q in range(n):
            if p != q:
                if dominates(objs[p], objs[q]):
                    S_[p].append(q)
                elif dominates(objs[q], objs[p]):
                    counts[p] += 1
        if counts[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in S_[p]:
                counts[q] -= 1
                if counts[q] == 0:
                    nxt.append(q)
        i += 1
        fronts.append(nxt)
    return fronts[:-1]


def crowding(objs, front):
    """NSGA-II crowding distance of every index in one front (boundary points get infinity)."""
    d = {i: 0.0 for i in front}
    if len(front) <= 2:
        return {i: float("inf") for i in front}
    for m in range(2):
        srt = sorted(front, key=lambda i: objs[i][m])
        lo, hi = objs[srt[0]][m], objs[srt[-1]][m]
        d[srt[0]] = d[srt[-1]] = float("inf")
        if hi == lo:
            continue
        for j in range(1, len(srt) - 1):
            d[srt[j]] += (objs[srt[j + 1]][m] - objs[srt[j - 1]][m]) / (hi - lo)
    return d


def rank_and_crowd(objs):
    """Front rank and crowding distance of every individual (for tournament selection)."""
    rank, crowd = {}, {}
    for r, f in enumerate(nondominated_sort(objs)):
        for i in f:
            rank[i] = r
        crowd.update(crowding(objs, f))
    return rank, crowd


def select_survivors(objs, size):
    """NSGA-II survivor selection: fill `size` slots front by front, breaking the last
    front by crowding distance.
    """
    chosen = []
    for f in nondominated_sort(objs):
        if len(chosen) + len(f) <= size:
            chosen += f
        else:
            cd = crowding(objs, f)
            chosen += sorted(f, key=lambda i: -cd[i])[:size - len(chosen)]
            break
    return chosen


def objective_key():
    """Result field that is maximised: 'retention' (default) or 'fid_mean' (fidelity search)."""
    if S.OBJECTIVE not in ("retention", "fidelity"):
        raise ValueError(f"search_config.OBJECTIVE must be 'retention' or 'fidelity', got {S.OBJECTIVE!r}")
    return "fid_mean" if S.OBJECTIVE == "fidelity" else "retention"


def objectives(result):
    """Objective tuple of a result: (maximised quantity, two-qubit gates)."""
    return (result[objective_key()], result["n2q"])
