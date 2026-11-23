#!/usr/bin/env python3
# ----------------------------------------------------------------
# encoding_search.py  —  Evolutionary search for amplitude-encoding
# circuits that KEEP CLASS INFORMATION, not just fidelity.
#
# Objectives (NSGA-II):
#   1. retention  = CV AUC of a sign-blind fidelity-kernel SVM on the
#                   encoded states / the same on the exact states (maximise)
#   2. two-qubit gate count                                      (minimise)
# Mean fidelity is recorded for every genome but not optimised: the
# diagnostics showed that at typical fidelities the circuit error is
# larger than the entire class signal, so fidelity cannot rank encoders.
#
# Genomes use the benchmark repo's format; connectivity comes from
# hierarqcal Qcycle; circuits are simulated in NumPy (aae_sim.py).
# Settings live in search_config.py.
#
# Usage (from the repository root):
#   python search_amplitude/encoding_search.py                         # every run in search_config.RUNS
#   python search_amplitude/encoding_search.py --dataset bloodmnist --mode centered
#   python search_amplitude/encoding_search.py --force                 # restart instead of resuming
#   python search_amplitude/encoding_search.py --quick                 # tiny settings, pipeline test only
#
# Output: experiments/<dataset>/search/<mode>/
#   config_used.json   ceiling.json       batch_indices.npz
#   state.json         (checkpoint: resumes automatically)
#   evaluations.csv    every unique genome with retention, AUC, fidelity, gates
#   history.csv        per-generation summary
#   front_final.json   final Pareto front (input to select_encoder.py)
#   search_front.png   retention vs two-qubit gates, all evaluations
#   fidelity_vs_retention.png
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

import numpy as np

import diag_common as D
import search_config as S
import search_core as SC


def apply_quick():
    """Tiny settings for a pipeline test (--quick); the numbers are meaningless."""
    S.SEARCH_N_IMAGES, S.FIT_STEPS, S.POP_SIZE, S.N_GENERATIONS = 120, 30, 8, 2
    S.SELECT_N_IMAGES, S.SELECT_FIT_STEPS, S.SELECT_FIT_RESTARTS = 120, 40, 1
    S.CV_FOLDS, S.MAX_BLOCKS, S.INIT_MAX_BLOCKS = 3, 4, 3


# settings that do not change what a genome's fitness means, so changing
# them must not invalidate a checkpoint (e.g. extending N_GENERATIONS)
NON_FITNESS_KEYS = {"RUNS", "N_WORKERS", "N_GENERATIONS", "OUT_ROOT"}


def config_snapshot(fitness_only=False):
    """Current search settings as a JSON-safe dict. With fitness_only=True, only the
    settings that change the fitness values (used to decide whether a checkpoint can be
    resumed).
    """
    snap = {k: getattr(S, k) for k in dir(S) if k.isupper() and not k.startswith("SELECT_")}
    if fitness_only:
        snap = {k: v for k, v in snap.items() if k not in NON_FITNESS_KEYS}
    return json.loads(json.dumps(snap))


def write_logs(odir, ev_cache, history):
    """Write every evaluated genome (evaluations.csv) and the per-generation history."""
    rows = []
    for k, r in ev_cache.items():
        g = json.loads(k)
        rows.append({**{f: r[f] for f in ("retention", "auc", "auc_std", "bal_acc",
                                          "fid_mean", "fid_median", "n2q", "n_params")},
                     "n_blocks": SC.n_blocks(g), "first_generation": r.get("gen", ""),
                     "genome": k})
    rows.sort(key=lambda r: (-r["retention"], r["n2q"]))
    D.write_csv(rows, os.path.join(odir, "evaluations.csv"))
    D.write_csv(history, os.path.join(odir, "history.csv"))


def plot(odir, ev_cache, front, ceil, tag):
    """Search overview: all evaluated genomes (retention vs. two-qubit gates, coloured by
    generation), the final front and the fidelity of each genome.
    """
    plt = D.get_plt()
    res = list(ev_cache.values())
    gens = np.array([r.get("gen", 0) for r in res])
    ret = np.array([r["retention"] for r in res])
    n2q = np.array([r["n2q"] for r in res])
    fid = np.array([r["fid_mean"] for r in res])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    sc = ax.scatter(n2q, ret, c=gens, cmap="viridis", s=22, alpha=0.8)
    f = sorted(front, key=lambda r: r["n2q"])
    ax.step([r["n2q"] for r in f], [r["retention"] for r in f], "r-", where="post",
            label="final Pareto front")
    ax.axhline(1.0, color="k", ls=":", label=f"exact-state ceiling (AUC {ceil['auc']:.3f})")
    ax.set_xlabel("two-qubit gates"); ax.set_ylabel("retention (CV AUC / ceiling AUC)")
    fig.colorbar(sc, ax=ax, label="generation first evaluated")
    ax.legend(fontsize=8); ax.set_title(f"{tag}: {S.OBJECTIVE}-driven search (front in {S.OBJECTIVE})")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "search_front.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(fid, ret, c=n2q, cmap="plasma", s=22)
    if len(fid) > 2:
        rho = np.corrcoef(np.argsort(np.argsort(fid)), np.argsort(np.argsort(ret)))[0, 1]
        ax.set_title(f"{tag}: fidelity vs retention (Spearman {rho:.2f})")
    ax.set_xlabel("mean fidelity"); ax.set_ylabel("retention")
    fig.colorbar(sc, ax=ax, label="two-qubit gates")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "fidelity_vs_retention.png"), dpi=150); plt.close(fig)


def run(dataset, mode, force):
    """NSGA-II search for one (dataset, mode): prepare the batch and the exact-state
    ceiling, evolve (resuming from state.json if possible) and save the final front.
    """
    tag = f"{dataset}/{mode}"
    odir = D.out_dir(S.OUT_ROOT, dataset, "search", mode)
    state_path = os.path.join(odir, "state.json")
    rng = np.random.default_rng(S.SEED)
    t0 = time.time()

    print(f"\n{'=' * 64}\n[{tag}] preparing search batch...")
    search, select = SC.prepare_batches(dataset, mode)
    np.savez(os.path.join(odir, "batch_indices.npz"), search_train_idx=search["train_idx"],
             select_source=np.array([f"{a}:{b}" for a, b in select["source"]]))
    folds = D.make_folds(search["y"], S.CV_FOLDS, S.SEED)
    ceil = SC.ceiling(search["T"], search["y"], folds)
    D.save_json(ceil, os.path.join(odir, "ceiling.json"))
    D.save_json(config_snapshot(), os.path.join(odir, "config_used.json"))
    print(f"  {len(search['T'])} images, classes {D.class_counts(search['y'])}")
    print(f"  ceiling (exact states): CV AUC {ceil['auc']:.3f} ± {ceil['auc_std']:.3f}, "
          f"bal_acc {ceil['bal_acc']:.3f}, C = {ceil['C']}")

    start_gen, pop, cache, history = 0, None, {}, []
    if os.path.exists(state_path) and not force:
        st = D.load_json(state_path)
        if st.get("config") == config_snapshot(fitness_only=True):
            start_gen, pop, cache, history = st["next_gen"], st["population"], st["cache"], st["history"]
            print(f"  resuming at generation {start_gen} ({len(cache)} genomes cached)")
        else:
            print("  config changed since the checkpoint, starting fresh (old state overwritten)")

    ev = SC.Evaluator(search["T"], search["y"], folds, ceil["C"], ceil["auc"],
                      S.FIT_STEPS, S.FIT_RESTARTS, S.SEED + 17, cache)
    print(f"  workers: {ev.n_workers}")

    def evaluate(genomes, gen):
        """Evaluate genomes (cached ones are reused) and record the generation they first
        appeared in.
        """
        res, n_new = ev.evaluate(genomes)
        for g in genomes:
            ev.cache[SC.key(g)].setdefault("gen", gen)
        return res, n_new

    try:
        if pop is None:
            pop, keys = [], set()
            for g in SC.seed_genomes() + [SC.random_genome(rng) for _ in range(20 * S.POP_SIZE)]:
                if len(pop) >= S.POP_SIZE:
                    break
                if SC.valid(g) and SC.key(g) not in keys:
                    pop.append(g)
                    keys.add(SC.key(g))
            t = time.time()
            res, n_new = evaluate(pop, 0)
            print(f"  gen  0: evaluated {n_new} genomes in {time.time() - t:.0f}s")

        for gen in range(max(start_gen, 1), S.N_GENERATIONS + 1):
            res, _ = ev.evaluate(pop)
            objs = [SC.objectives(r) for r in res]
            rank, crowd = SC.rank_and_crowd(objs)

            def tournament():
                idx = rng.choice(len(pop), S.TOURNAMENT_K, replace=False)
                return pop[min(idx, key=lambda i: (rank[i], -crowd[i]))]

            offspring, keys = [], {SC.key(g) for g in pop}
            tries = 0
            while len(offspring) < S.POP_SIZE and tries < 50 * S.POP_SIZE:
                tries += 1
                p1, p2 = tournament(), tournament()
                child = SC.crossover(p1, p2, rng) if rng.random() < S.CROSSOVER_RATE else p1
                if rng.random() < S.MUTATION_RATE or SC.key(child) in keys:
                    child = SC.mutate(child, rng)
                k = SC.key(child)
                if SC.valid(child) and k not in keys and (k not in ev.cache or tries > 20 * S.POP_SIZE):
                    offspring.append(child)
                    keys.add(k)

            t = time.time()
            _, n_new = evaluate(offspring, gen)
            union = pop + offspring
            ures, _ = ev.evaluate(union)
            uobjs = [SC.objectives(r) for r in ures]
            keep = SC.select_survivors(uobjs, S.POP_SIZE)
            pop = [union[i] for i in keep]

            pres = [ures[i] for i in keep]
            front_idx = SC.nondominated_sort([SC.objectives(r) for r in pres])[0]
            front = [pres[i] for i in front_idx]
            key = SC.objective_key()
            best = max(pres, key=lambda r: r[key])
            history.append({
                "generation": gen, "new_evaluations": n_new, "total_evaluated": len(ev.cache),
                "objective": S.OBJECTIVE, "best_score": best[key], "best_score_n2q": best["n2q"],
                "median_score": float(np.median([r[key] for r in pres])),
                "best_retention": best["retention"], "best_fidelity": best["fid_mean"],
                "median_retention": float(np.median([r["retention"] for r in pres])),
                "front_size": len(front),
                "front": " ".join(f"{r['n2q']}:{r[key]:.3f}"
                                  for r in sorted(front, key=lambda r: r["n2q"])),
                "seconds": round(time.time() - t, 1)})
            print(f"  gen {gen:2d}: {n_new:2d} new | best {S.OBJECTIVE} {best[key]:.3f} "
                  f"({best['n2q']} 2q) | front {history[-1]['front']} | {time.time() - t:.0f}s")

            D.save_json({"config": config_snapshot(fitness_only=True), "next_gen": gen + 1, "population": pop,
                         "cache": ev.cache, "history": history}, state_path)
            write_logs(odir, ev.cache, history)
    finally:
        ev.close()

    res = [ev.cache[SC.key(g)] for g in pop]
    front_idx = SC.nondominated_sort([SC.objectives(r) for r in res])[0]
    front = [{"genome": pop[i], **res[i]} for i in front_idx]
    front.sort(key=lambda r: r["n2q"])
    fpath = os.path.join(odir, "front_final.json")
    new_front = {"dataset": dataset, "mode": mode, "ceiling": ceil, "front": front}
    old_front = D.load_json(fpath) if os.path.exists(fpath) else None
    if old_front != json.loads(json.dumps(new_front, default=D._json_default)):
        D.save_json(new_front, fpath)          # unchanged fronts keep their timestamp
    D.save_json({"config": config_snapshot(fitness_only=True), "next_gen": max(S.N_GENERATIONS + 1, start_gen), "population": pop,
                 "cache": ev.cache, "history": history}, state_path)
    write_logs(odir, ev.cache, history)
    plot(odir, ev.cache, front, ceil, tag)
    print(f"  final front: " + ", ".join(f"{r['n2q']} gates -> {r['retention']:.3f} "
                                         f"(F={r['fid_mean']:.2f})" for r in front))
    print(f"  saved to {odir}  ({time.time() - t0:.0f}s)")


def main():
    """Command line: select runs, seed, objective and output folder, then search each run."""
    p = argparse.ArgumentParser(description="retention-driven encoding search")
    p.add_argument("--dataset", default=None)
    p.add_argument("--mode", default=None, choices=SC.VALID_MODES)
    p.add_argument("--force", action="store_true", help="ignore checkpoint and restart")
    p.add_argument("--seed", type=int, default=None, help="search seed (default search_config.SEED)")
    p.add_argument("--objective", choices=["retention", "fidelity"], default=None,
                   help="what to maximise (default search_config.OBJECTIVE)")
    p.add_argument("--out-root", default=None, help="output root (default search_config.OUT_ROOT)")
    p.add_argument("--quick", action="store_true", help="tiny settings to test the pipeline")
    args = p.parse_args()
    if args.quick:
        apply_quick()
        S.OUT_ROOT = "experiments_quick"
    if args.seed is not None:
        S.SEED = args.seed
    if args.objective:
        S.OBJECTIVE = args.objective
    if args.out_root:
        S.OUT_ROOT = args.out_root
    runs = [(args.dataset, args.mode)] if args.dataset else S.RUNS
    if args.dataset and not args.mode:
        raise SystemExit("--dataset needs --mode")
    for ds, mode in runs:
        run(ds, mode, args.force)


if __name__ == "__main__":
    main()
