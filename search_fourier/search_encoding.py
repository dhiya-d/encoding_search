#!/usr/bin/env python3
# ----------------------------------------------------------------
# search_encoding.py  —  evolve data-encoding circuits for classification.
#
# Per dataset in config.RUNS:
#   1. Features: low-frequency DCT coefficients (RGB where available),
#      ranked by Fisher score on TRAIN, mapped to angles.
#   2. Search (NSGA-II): genomes are data-encoding circuits S(x) built from
#      Pauli-rotation encodings, ZZ encodings, Hadamards and hierarqcal
#      entanglers. Objectives: CV AUC of an SVM on the encoding's quantum
#      kernel (maximise) and two-qubit cost (minimise). No per-image or
#      per-genome circuit training is needed, so thousands of candidates
#      are cheap to score. Hand-designed reference encodings from the
#      literature are seeded into the first generation.
#   3. Selection: the final front and the references are re-scored on a
#      disjoint batch (val split + unused train images). Up to N_PICKS
#      encodings are chosen for downstream training:
#        best       highest held-out AUC
#        knee       cheapest within SELECT_TOLERANCE of best
#        efficient  cheapest that beats every reference
#
# Usage (from the repository root):
#   python search_fourier/search_encoding.py                     # all datasets in config.RUNS
#   python search_fourier/search_encoding.py --dataset bloodmnist
#   python search_fourier/search_encoding.py --force             # ignore checkpoints
#   python search_fourier/search_encoding.py --quick             # tiny settings, pipeline test
#
# Output: experiments_fourier/<dataset>/
#   features/   {train,val,test}_features.npy + labels, feature_meta.json
#   search/     evaluations.csv  history.csv  state.json  front.json
#               selection.csv  picks.json
#               search_front.png  selection.png  degree_vs_auc.png
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as U
import config as C
import encoding_genome as G
import nsga2

# ----------------------------------------------------------------
# Fitness
# ----------------------------------------------------------------
_W = {}


def _init_worker(X, y, folds, kernel):
    """Worker-process setup: one BLAS thread per worker and the shared search batch."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _W.update(X=X, y=y, folds=folds, kernel=kernel)


def kernel_matrix(circ, S, kernel):
    """Fitness kernel of the encoded states: fidelity kernel, or RBF kernel on the single-
    qubit Pauli expectations (projected kernel).
    """
    if kernel == "fidelity":
        return U.fidelity_kernel(S, S)
    P = circ.pauli_features(S)
    return U.rbf_kernel(P, P, U.median_gamma(P))


def _evaluate(g):
    """Fitness of one genome: CV AUC of the kernel SVM on the search batch, plus kernel
    concentration and spectrum statistics.
    """
    circ = G.encoding_circuit(g)
    S = circ.states(_W["X"])
    K = kernel_matrix(circ, S, _W["kernel"])
    off = K[~np.eye(len(K), dtype=bool)]
    sc = U.best_cv_kernel_score(K, _W["y"], _W["folds"])
    spec = G.spectrum_summary(g)
    return {"auc": sc["auc"], "auc_std": sc["auc_std"], "bal_acc": sc["bal_acc"], "C": sc["C"],
            "cost": G.two_qubit_cost(g), "n_layers": len(g["layers"]),
            "n_features": g["n_features"], "features_encoded": spec["features_encoded"],
            "mean_degree": spec["mean_degree"], "max_degree": spec["max_degree"],
            "has_zz": spec["has_zz_products"],
            "kernel_offdiag_mean": float(off.mean()), "kernel_offdiag_std": float(off.std())}


class Evaluator:
    """Evaluates genomes in parallel and caches the results by genome key."""
    def __init__(self, X, y, folds, cache=None):
        """Start the worker pool (or run serially when N_WORKERS = 1)."""
        self.cache = dict(cache or {})
        self.n_workers = C.N_WORKERS or min(8, os.cpu_count() or 1)
        args = (X, y, folds, C.FITNESS_KERNEL)
        if self.n_workers > 1:
            self.pool = ProcessPoolExecutor(self.n_workers, initializer=_init_worker, initargs=args)
        else:
            self.pool = None
            _init_worker(*args)

    def __call__(self, genomes):
        """Results for a list of genomes, evaluating only those not yet cached. Returns
        (results, number of new evaluations).
        """
        todo, seen = [], set()
        for g in genomes:
            k = G.key(g)
            if k not in self.cache and k not in seen:
                todo.append(g)
                seen.add(k)
        if todo:
            res = list(self.pool.map(_evaluate, todo, chunksize=2)) if self.pool else [_evaluate(g) for g in todo]
            for g, r in zip(todo, res):
                self.cache[G.key(g)] = r
        return [self.cache[G.key(g)] for g in genomes], len(todo)

    def close(self):
        """Shut down the worker pool."""
        if self.pool:
            self.pool.shutdown()


def objectives(r):
    """Objective tuple: (CV AUC to maximise, two-qubit cost to minimise)."""
    return (r["auc"], r["cost"])


# ----------------------------------------------------------------
# Batches
# ----------------------------------------------------------------

def batches(feats):
    """Search batch (stratified train sample) and a disjoint selection batch (remaining
    train images plus the validation split).
    """
    rng = np.random.default_rng(C.SEED)
    Xtr, ytr = feats["train"]
    s_idx = U.stratified_sample(ytr, C.SEARCH_N, rng)
    rest = np.setdiff1d(np.arange(len(ytr)), s_idx)
    pool_X, pool_y = [Xtr[rest]], [ytr[rest]]
    if "val" in feats:
        pool_X.append(feats["val"][0])
        pool_y.append(feats["val"][1])
    pX, py = np.concatenate(pool_X), np.concatenate(pool_y)
    sel = U.stratified_sample(py, C.SELECT_N, np.random.default_rng(C.SEED + 1))
    return (Xtr[s_idx], ytr[s_idx]), (pX[sel], py[sel])


# ----------------------------------------------------------------
# Logging and figures
# ----------------------------------------------------------------

def snapshot():
    """Settings that affect the search; a checkpoint is only resumed if they are unchanged."""
    keys = ["FEATURE_METHOD", "DCT_MAX_ORDER", "USE_RGB", "MAX_FEATURES", "ANGLE_CLIP",
            "ANGLE_RANGE", "SEED", "FITNESS_KERNEL", "SEARCH_N", "CV_FOLDS", "SVM_C_GRID",
            "FEATURE_COUNTS", "ENC_AXES", "ENC_SCALES", "ZZ_SCALES", "ENTANGLERS", "STRIDES",
            "STEPS", "OFFSETS", "LAYER_WEIGHTS", "MIN_LAYERS", "MAX_LAYERS", "INIT_MAX_LAYERS",
            "MAX_TWO_QUBIT_COST", "POP_SIZE", "TOURNAMENT_K", "CROSSOVER_RATE", "MUTATION_RATE"]
    return json.loads(json.dumps({k: getattr(C, k) for k in keys}))


def write_logs(sdir, cache, history, ref_keys):
    """Write every evaluated genome (evaluations.csv) and the per-generation history."""
    rows = []
    for k, r in cache.items():
        g = json.loads(k)
        rows.append({**{f: v for f, v in r.items() if f != "gen"},
                     "first_generation": r.get("gen", ""),
                     "reference": ref_keys.get(k, ""), "description": G.describe(g), "genome": k})
    rows.sort(key=lambda r: (-r["auc"], r["cost"]))
    U.write_csv(rows, os.path.join(sdir, "evaluations.csv"))
    U.write_csv(history, os.path.join(sdir, "history.csv"))


def plot_search(sdir, cache, front, ref_keys, tag):
    """All evaluated genomes (AUC vs. two-qubit cost, coloured by generation), the final
    front and the reference encodings.
    """
    plt = U.get_plt()
    items = list(cache.items())
    auc = np.array([r["auc"] for _, r in items])
    cost = np.array([r["cost"] for _, r in items])
    gen = np.array([r.get("gen", 0) for _, r in items])
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    sc = ax.scatter(cost, auc, c=gen, cmap="viridis", s=14, alpha=0.7)
    f = sorted(front, key=lambda r: r["cost"])
    ax.step([r["cost"] for r in f], [r["auc"] for r in f], "r-", where="post", label="final front")
    for k, r in items:
        if k in ref_keys:
            ax.scatter(r["cost"], r["auc"], marker="*", s=160, color="orange", edgecolor="k", zorder=3)
            ax.annotate(ref_keys[k].replace("ref_", ""), (r["cost"], r["auc"]), fontsize=7,
                        xytext=(4, -8), textcoords="offset points")
    ax.set_xlabel("two-qubit cost"); ax.set_ylabel(f"CV AUC ({C.FITNESS_KERNEL} kernel SVM)")
    fig.colorbar(sc, ax=ax, label="generation first evaluated")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title(f"{tag}: encoding search (stars = references)")
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "search_front.png"), dpi=150); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    deg = np.array([r["mean_degree"] for _, r in items])
    nf = np.array([r["n_features"] for _, r in items])
    s0 = axes[0].scatter(deg, auc, c=nf, cmap="plasma", s=14)
    axes[0].set_xlabel("mean Fourier degree per encoded feature"); axes[0].set_ylabel("CV AUC")
    fig.colorbar(s0, ax=axes[0], label="features d")
    for d in sorted(set(nf)):
        m = nf == d
        axes[1].scatter(np.full(m.sum(), d) + np.random.default_rng(0).uniform(-.8, .8, m.sum()),
                        auc[m], s=10, alpha=0.6)
    axes[1].set_xlabel("number of features d"); axes[1].set_ylabel("CV AUC")
    fig.suptitle(f"{tag}: what the search learned about the encoding spectrum")
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "degree_vs_auc.png"), dpi=150); plt.close(fig)


def plot_selection(sdir, rows, picks, tag):
    """Held-out AUC of the final front and the references, with the picks marked."""
    plt = U.get_plt()
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for r in rows:
        ref = r["name"].startswith("ref_")
        ax.errorbar(r["cost"], r["sel_auc"], yerr=r["sel_auc_std"], fmt="*" if ref else "o",
                    color="orange" if ref else "#3b6ea5", ms=12 if ref else 6, alpha=0.9)
        if ref:
            ax.annotate(r["name"].replace("ref_", ""), (r["cost"], r["sel_auc"]), fontsize=7,
                        xytext=(4, -8), textcoords="offset points")
    for p in picks:
        ax.scatter(p["cost"], p["sel_auc"], s=260, facecolors="none", edgecolors="r", linewidths=2)
        ax.annotate(p["pick"], (p["cost"], p["sel_auc"]), color="r", fontsize=8,
                    xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel("two-qubit cost"); ax.set_ylabel("held-out CV AUC")
    ax.set_title(f"{tag}: selection on a held-out batch (red = picks)")
    fig.tight_layout(); fig.savefig(os.path.join(sdir, "selection.png"), dpi=150); plt.close(fig)


# ----------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------

def run(dataset, force):
    """Search for one dataset: features, batches, NSGA-II (resumable), then re-evaluation
    of the final front and the references on the selection batch and the final picks.
    """
    t0 = time.time()
    print(f"\n{'=' * 70}\n[{dataset}] features...")
    feats, meta = U.load_or_build_features(dataset, force=force)
    print(f"  {'RGB' if meta['rgb'] else 'grayscale'} {meta['method']} features, "
          f"top Fisher: {', '.join(meta['feature_names'][:5])}")
    (Xs, ys), (Xsel, ysel) = batches(feats)
    folds = U.make_folds(ys, C.CV_FOLDS, C.SEED)
    sdir = U.out_dir(dataset, "search")
    state_path = os.path.join(sdir, "state.json")
    rng = np.random.default_rng(C.SEED)
    refs = G.references()
    ref_keys = {G.key(g): name for name, g in refs.items()}

    gen0, pop, cache, history = 0, None, {}, []
    if os.path.exists(state_path) and not force:
        st = U.load_json(state_path)
        if st.get("config") == snapshot():
            gen0, pop, cache, history = st["next_gen"], st["population"], st["cache"], st["history"]
            print(f"  resuming at generation {gen0} ({len(cache)} genomes cached)")
        else:
            print("  settings changed since the checkpoint: starting fresh")

    ev = Evaluator(Xs, ys, folds, cache)
    print(f"  search batch {len(ys)} images {U.class_counts(ys)}, {ev.n_workers} workers, "
          f"fitness kernel: {C.FITNESS_KERNEL}")

    def evaluate(genomes, gen):
        """Evaluate genomes (cached ones are reused) and record the generation they first
        appeared in.
        """
        res, n = ev(genomes)
        for g in genomes:
            ev.cache[G.key(g)].setdefault("gen", gen)
        return res, n

    def save(next_gen):
        """Checkpoint the search state and the logs."""
        U.save_json({"config": snapshot(), "next_gen": next_gen, "population": pop,
                     "cache": ev.cache, "history": history}, state_path)
        write_logs(sdir, ev.cache, history, ref_keys)

    try:
        if pop is None:
            pop, keys = [], set()
            for g in list(refs.values()) + [G.random_genome(rng) for _ in range(50 * C.POP_SIZE)]:
                if len(pop) >= C.POP_SIZE:
                    break
                if G.valid(g) and G.key(g) not in keys:
                    pop.append(g)
                    keys.add(G.key(g))
            t = time.time()
            res, n = evaluate(pop, 0)
            best = max(res, key=lambda r: r["auc"])
            print(f"  gen  0: {n} evaluated | best AUC {best['auc']:.3f} (cost {best['cost']}) | "
                  f"{time.time() - t:.0f}s")
            gen0 = 1
            save(1)

        for gen in range(gen0, C.N_GENERATIONS + 1):
            t = time.time()
            res, _ = ev(pop)
            rank, crowd = nsga2.rank_and_crowd([objectives(r) for r in res])

            def tournament():
                idx = rng.choice(len(pop), C.TOURNAMENT_K, replace=False)
                return pop[min(idx, key=lambda i: (rank[i], -crowd[i]))]

            children, keys, tries = [], {G.key(g) for g in pop}, 0
            while len(children) < C.POP_SIZE and tries < 100 * C.POP_SIZE:
                tries += 1
                a, b = tournament(), tournament()
                child = G.crossover(a, b, rng) if rng.random() < C.CROSSOVER_RATE else G.copy(a)
                if rng.random() < C.MUTATION_RATE or G.key(child) in keys:
                    child = G.mutate(child, rng)
                k = G.key(child)
                if G.valid(child) and k not in keys and (k not in ev.cache or tries > 50 * C.POP_SIZE):
                    children.append(child)
                    keys.add(k)
            _, n_new = evaluate(children, gen)
            union = pop + children
            ures, _ = ev(union)
            keep = nsga2.select_survivors([objectives(r) for r in ures], C.POP_SIZE)
            pop = [union[i] for i in keep]
            pres = [ures[i] for i in keep]
            front = [pres[i] for i in nsga2.nondominated_sort([objectives(r) for r in pres])[0]]
            best = max(pres, key=lambda r: r["auc"])
            fs = " ".join(f"{r['cost']}:{r['auc']:.3f}" for r in sorted(front, key=lambda r: r["cost"]))
            history.append({"generation": gen, "new": n_new, "total_evaluated": len(ev.cache),
                            "best_auc": best["auc"], "best_cost": best["cost"],
                            "median_auc": float(np.median([r["auc"] for r in pres])),
                            "front": fs, "seconds": round(time.time() - t, 1)})
            print(f"  gen {gen:2d}: {n_new:2d} new | best {best['auc']:.3f} (cost {best['cost']}) "
                  f"| front {fs} | {time.time() - t:.0f}s")
            save(gen + 1)
    finally:
        ev.close()

    # ---- final front + selection -----------------------------------
    res = [ev.cache[G.key(g)] for g in pop]
    front_idx = nsga2.nondominated_sort([objectives(r) for r in res])[0]
    front = sorted([{"genome": pop[i], **res[i]} for i in front_idx], key=lambda r: r["cost"])
    U.save_json({"dataset": dataset, "front": front}, os.path.join(sdir, "front.json"))
    plot_search(sdir, ev.cache, front, ref_keys, dataset)

    print(f"  selection: re-scoring {len(front)} front genomes + {len(refs)} references "
          f"on {len(ysel)} held-out images...")
    cands = [(f"front_{i}_cost{r['cost']}", r["genome"]) for i, r in enumerate(front)]
    cands += [(n, g) for n, g in refs.items() if G.valid(g)]
    sel_folds = U.make_folds(ysel, C.CV_FOLDS, C.SEED + 1)
    sev = Evaluator(Xsel, ysel, sel_folds)
    try:
        sres, _ = sev([g for _, g in cands])
    finally:
        sev.close()
    rows = []
    for (name, g), r in zip(cands, sres):
        rows.append({"name": name, "cost": r["cost"], "sel_auc": r["auc"], "sel_auc_std": r["auc_std"],
                     "sel_bal_acc": r["bal_acc"], "search_auc": ev.cache.get(G.key(g), {}).get("auc"),
                     "n_features": g["n_features"], "mean_degree": r["mean_degree"],
                     "description": G.describe(g), "genome": g})
    frows = [r for r in rows if not r["name"].startswith("ref_")]
    best = max(frows, key=lambda r: r["sel_auc"])
    picks = [{"pick": "best", **best}]
    knee = min((r for r in frows if r["sel_auc"] >= best["sel_auc"] - C.SELECT_TOLERANCE),
               key=lambda r: (r["cost"], -r["sel_auc"]))
    picks.append({"pick": "knee", **knee})
    best_ref = max((r for r in rows if r["name"].startswith("ref_")), key=lambda r: r["sel_auc"])
    beating = [r for r in frows if r["sel_auc"] > best_ref["sel_auc"]]
    if beating:
        picks.append({"pick": "efficient", **min(beating, key=lambda r: (r["cost"], -r["sel_auc"]))})
    uniq, seen = [], set()
    for p in picks:
        k = G.key(p["genome"])
        if k in seen:
            for q in uniq:
                if G.key(q["genome"]) == k:
                    q["pick"] += "+" + p["pick"]
            continue
        seen.add(k)
        uniq.append(p)
    picks = uniq[:C.N_PICKS]
    for p in picks:
        p["name"] = f"search_{p['pick'].replace('+', '_')}"

    U.write_csv([{k: v for k, v in r.items() if k != "genome"} for r in rows],
                os.path.join(sdir, "selection.csv"))
    U.save_json({"dataset": dataset, "best_reference": best_ref["name"],
                 "best_reference_auc": best_ref["sel_auc"], "picks": picks, "all": rows},
                os.path.join(sdir, "picks.json"))
    plot_selection(sdir, rows, picks, dataset)

    print(f"  best reference: {best_ref['name']} AUC {best_ref['sel_auc']:.3f} (cost {best_ref['cost']})")
    for p in picks:
        print(f"  pick {p['name']:28s} AUC {p['sel_auc']:.3f} (cost {p['cost']:2d}) {p['description']}")
    print(f"  done in {time.time() - t0:.0f}s -> {sdir}")


def apply_quick():
    """Tiny settings for a pipeline test (--quick); the numbers are meaningless."""
    C.SEARCH_N, C.SELECT_N, C.POP_SIZE, C.N_GENERATIONS, C.CV_FOLDS = 160, 160, 12, 3, 3
    C.OUT_ROOT = "experiments_fourier_quick"
    C.VQC_N_TRAIN, C.VQC_N_VAL, C.VQC_EPOCHS, C.KERNEL_N_TRAIN = 200, 100, 3, 300
    C.TEST_MAX = 300


def main():
    """Command line: datasets, seed, output folder, quick mode."""
    p = argparse.ArgumentParser(description="evolutionary search over Fourier data encodings")
    p.add_argument("--dataset", nargs="*", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--seed", type=int, default=None, help="search seed (default config.SEED)")
    p.add_argument("--out-root", default=None, help="results root (default config.OUT_ROOT)")
    a = p.parse_args()
    if a.quick:
        apply_quick()
    if a.seed is not None:
        C.SEED = a.seed
    if a.out_root:
        C.OUT_ROOT = a.out_root
    names = a.dataset or C.RUNS
    U.check_datasets(names)
    for d in names:
        run(d, a.force)


if __name__ == "__main__":
    main()
