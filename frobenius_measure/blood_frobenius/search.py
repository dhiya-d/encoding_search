"""Frobenius-measure encoding-architecture search for BloodMNIST.

Fitness base: mean_x || t_x - U(theta_x)|0> ||_F  (minimised), where the angles
theta_x are fitted per image (approximate amplitude encoding).

Selection strategy is chosen with the pareto_front switch:
    --pareto-front      NSGA-II on (frobenius, gate count); knee chosen after.
    --no-pareto-front   single objective: frobenius + gate_penalty * gates.

Examples:
    python search.py --data-dir bloodmnist --merge none --pareto-front \
        --pop 24 --gen 12 --fit-steps 150 --outdir experiments/blo_seed0 --seed 0

    python search.py --data-dir bloodmnist --merge none --no-pareto-front \
        --gate-penalty 0.004 --pop 24 --gen 12 --outdir experiments/frob_blood_single
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                    # noqa: E402
import torch                                          # noqa: E402

import selection as sel                               # noqa: E402
from config import DataConfig, SearchConfig           # noqa: E402
from data_local import load                           # noqa: E402
from encoder import fit_batch                         # noqa: E402
from plot_safety import safe_render                   # noqa: E402


def pick_device(arg):
    if arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return arg


def score_individual(ind, targets, cfg, device, final=False):
    r = fit_batch(ind.genome, targets, steps=cfg.fit_steps, lr=cfg.fit_lr,
                  restarts=cfg.restarts, seed=cfg.seed, device=device)
    ind.frob = r["mean_frobenius"]
    ind.gates = ind.genome.two_qubit_gates()
    ind.score = sel.composite_score(ind.frob, ind.gates, cfg.gate_penalty)
    ind.info = {"mean_frobenius": r["mean_frobenius"], "max_frobenius": r["max_frobenius"],
                "mean_overlap": r["mean_overlap"], "gates": ind.gates,
                "params": ind.genome.n_params(), "cry": ind.genome.n_cry(),
                "depth": ind.genome.depth(), "final": final}
    return ind


def main():
    ap = argparse.ArgumentParser(description="Frobenius encoding search (BloodMNIST).")
    ap.add_argument("--data-dir", default="bloodmnist")
    ap.add_argument("--merge", default="none", choices=("none",))
    ap.add_argument("--channel", default="gray", choices=("gray", "green", "rgb"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--no-center", dest="center", action="store_false", default=True)
    # the switch
    ap.add_argument("--pareto-front", dest="pareto_front", action="store_true", default=True)
    ap.add_argument("--no-pareto-front", dest="pareto_front", action="store_false")
    ap.add_argument("--gate-penalty", type=float, default=0.004)
    ap.add_argument("--knee-tolerance", type=float, default=0.015)
    # GA
    ap.add_argument("--pop", type=int, default=24)
    ap.add_argument("--gen", type=int, default=12)
    ap.add_argument("--min-gates", type=int, default=1)
    ap.add_argument("--max-layers", type=int, default=10)
    ap.add_argument("--elite", type=int, default=2)
    # AAE fitting
    ap.add_argument("--fit-steps", type=int, default=150)
    ap.add_argument("--fit-lr", type=float, default=0.08)
    ap.add_argument("--restarts", type=int, default=1)
    ap.add_argument("--batch-train", type=int, default=256)
    ap.add_argument("--batch-final", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="experiments/frob_search")
    a = ap.parse_args()

    dcfg = DataConfig(data_dir=a.data_dir, channel=a.channel, size=a.size,
                      center=a.center, merge=a.merge)
    cfg = SearchConfig(pareto_front=a.pareto_front, gate_penalty=a.gate_penalty,
                       knee_tolerance=a.knee_tolerance, pop=a.pop, gen=a.gen,
                       min_gates=a.min_gates, max_layers=a.max_layers, elite=a.elite,
                       fit_steps=a.fit_steps, fit_lr=a.fit_lr, restarts=a.restarts,
                       batch_train=a.batch_train, batch_final=a.batch_final,
                       seed=a.seed, device=a.device, outdir=a.outdir)

    os.makedirs(cfg.outdir, exist_ok=True)
    rng = random.Random(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = pick_device(cfg.device)

    data = load(dcfg.data_dir, channel=dcfg.channel, size=dcfg.size,
                center=dcfg.center, merge=dcfg.merge)
    n = data["n_qubits"]
    Xtr = torch.as_tensor(data["Xtr"])
    idx = np.random.default_rng(cfg.seed).permutation(len(Xtr))
    t_search = Xtr[idx[:cfg.batch_train]]
    t_final = Xtr[idx[:cfg.batch_final]]

    mode = "PARETO (NSGA-II)" if cfg.pareto_front else f"SINGLE (penalty={cfg.gate_penalty})"
    print(f"[data] {dcfg.data_dir} {dcfg.channel}/{dcfg.size}px merge={dcfg.merge} -> "
          f"{data['side']}x{data['side']} = {data['dim']} amplitudes, {n} qubits")
    print(f"[data] {data['n_classes']} classes | chance={data['chance']:.3f}")
    print(f"[run ] mode={mode} | device={device} | pop={cfg.pop} gen={cfg.gen} "
          f"| fit_steps={cfg.fit_steps} | batch={cfg.batch_train}")

    t0 = time.time()
    pop = sel.dedup(sel.init_population(n, cfg, rng))
    for ind in pop:
        score_individual(ind, t_search, cfg, device)

    log = []
    for gen in range(cfg.gen):
        kids = sel.make_offspring(pop, rng, cfg, cfg.pareto_front)
        seen = {i.genome.signature() for i in pop}
        kids = [k for k in kids if k.genome.signature() not in seen]
        for k in kids:
            score_individual(k, t_search, cfg, device)

        combined = pop + kids
        if cfg.pareto_front:
            pop = sel.select_next_pareto(combined, cfg.pop)
            front = sel.pareto_front_of(pop)
            best = min(front, key=lambda i: i.frob)
            g = [i.gates for i in front]
            line = (f"gen {gen:2d} | best_frob={best.frob:.4f} | front={len(front)} "
                    f"| gates {min(g)}-{max(g)}")
            row = {"gen": gen, "best_frob": best.frob, "front": len(front),
                   "min_gates": min(g), "max_gates": max(g)}
        else:
            pop = sel.select_next_single(combined, cfg.pop)
            best = pop[0]
            line = (f"gen {gen:2d} | best_score={best.score:.4f} "
                    f"(frob={best.frob:.4f}, gates={best.gates})")
            row = {"gen": gen, "best_score": best.score, "best_frob": best.frob,
                   "best_gates": best.gates}
        row["elapsed_s"] = round(time.time() - t0, 1)
        log.append(row)
        print(f"{line} | {row['elapsed_s']}s")

    # ---- final re-benchmark on a larger batch ------------------------------
    if cfg.pareto_front:
        finalists = sel.dedup(sel.pareto_front_of(pop))
    else:
        finalists = sel.dedup(pop)[:min(8, len(pop))]
    print(f"\n[re-benchmarking {len(finalists)} finalists on {len(t_final)} images]")
    for ind in finalists:
        score_individual(ind, t_final, cfg, device, final=True)

    if cfg.pareto_front:
        finalists.sort(key=lambda i: i.gates)
        winner = sel.knee_of(finalists, cfg.knee_tolerance)
        header = "PARETO FRONT (re-benchmarked)"
    else:
        for i in finalists:
            i.score = sel.composite_score(i.frob, i.gates, cfg.gate_penalty)
        finalists.sort(key=lambda i: i.score)
        winner = finalists[0]
        header = "TOP CANDIDATES (re-benchmarked)"

    print(f"\n{header:^64}")
    print(f"  {'frob':>7} {'overlap':>8} {'gates':>6} {'cry':>4} {'params':>7}   architecture")
    for i in finalists:
        mark = "  <-- selected" if i is winner else ""
        print(f"  {i.frob:>7.4f} {i.info['mean_overlap']:>8.4f} {i.gates:>6} "
              f"{i.info['cry']:>4} {i.info['params']:>7}   {i.genome.summary()}{mark}")

    out = {
        "data": dcfg.__dict__, "search": cfg.to_dict(), "n_qubits": n, "log": log,
        "candidates": [{"frob": i.frob, "gates": i.gates, "score": i.score,
                        "info": i.info, "genome": i.genome.to_dict()} for i in finalists],
        "winner": {"frob": winner.frob, "gates": winner.gates, "score": winner.score,
                   "genome": winner.genome.to_dict()},
    }
    with open(os.path.join(cfg.outdir, "search_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(cfg.outdir, "best_genome.json"), "w") as fh:
        json.dump(winner.genome.to_dict(), fh, indent=2)

    safe_render("plot", _plot, finalists, cfg, os.path.join(cfg.outdir, "front.png"))

    try:
        from draw_circuit import build_columns, draw as draw_circuit
        gd = winner.genome.to_dict()
        cols = build_columns(gd)   # pure Python, no rendering -- safe in-process
        circuit_path = os.path.join(cfg.outdir, "best_circuit.png")
        safe_render("circuit image", draw_circuit, gd, cols, circuit_path)
    except Exception as e:
        print("circuit image skipped:", e)

    print(f"\nselected: frob={winner.frob:.4f} gates={winner.gates} "
          f"-> {os.path.join(cfg.outdir, 'best_genome.json')}")


def _plot(finalists, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [i.gates for i in finalists]
    f = [i.frob for i in finalists]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(g, f, "o" if not cfg.pareto_front else "-o", color="#5A4BD6")
    ax.set_xlabel("two-qubit gate count")
    ax.set_ylabel(r"mean $\|t-s\|_F$  (lower is better)")
    ax.set_title("Pareto front" if cfg.pareto_front else "Candidates (single-objective)")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
