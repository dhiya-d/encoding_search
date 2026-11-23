"""Stage 2: evolutionary search over QCNN architectures.

Runs on the CACHED encoded states produced by encode_states.py, so the input
representation is fixed and any difference in accuracy is attributable to the
QCNN architecture.

Search space: 1..6 layers, each a convolution (one of ten Qcycle ansatze, three
wire patterns) plus 0/1/2 pooling halvings, constrained so the register reduces
to --target-qubits. DermaMNIST has 7 classes, which don't divide 8 as a clean
power of two, so pass --target-qubits 4: the 4 surviving qubits give 16
outcomes, of which 7 are ever assigned a label (9 sit unused -- the model
just learns near-zero probability on them). No linear head needed.

Two selection modes, mirroring the encoding search:
    --no-pareto-front (default)  fitness = bal_acc - l1*gates - l2*params
    --pareto-front               NSGA-II on (bal_acc up, 2q gates down)

Example:
    python qcnn_search.py --data encoded/derma_seed0 --target-qubits 4 \
        --pop 12 --gen 10 --epochs 60 --lr 0.01 \
        --device auto --outdir experiments/qcnn_derma
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass, field

import numpy as np
import torch

import qcnn_genome as G
from qcnn_genome import QCNNGenome, configure, crossover, mutate, random_genome
from qcnn_model import TrainConfig, train_and_score


# ---------------------------------------------------------------------------
@dataclass
class Ind:
    genome: QCNNGenome
    bal_acc: float = 0.0
    gates: int = 0
    params: int = 0
    fitness: float = -1e9
    rank: int = 0
    crowd: float = 0.0
    info: dict = field(default_factory=dict)

    def objectives(self):
        return (self.bal_acc, -float(self.gates))     # both maximised


def dominates(a, b):
    return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))


def fronts_of(pop):
    F, S, nd = [[]], [[] for _ in pop], [0] * len(pop)
    for p in range(len(pop)):
        for q in range(len(pop)):
            if p == q:
                continue
            if dominates(pop[p].objectives(), pop[q].objectives()):
                S[p].append(q)
            elif dominates(pop[q].objectives(), pop[p].objectives()):
                nd[p] += 1
        if nd[p] == 0:
            pop[p].rank = 0
            F[0].append(p)
    i = 0
    while F[i]:
        nxt = []
        for p in F[i]:
            for q in S[p]:
                nd[q] -= 1
                if nd[q] == 0:
                    pop[q].rank = i + 1
                    nxt.append(q)
        i += 1
        F.append(nxt)
    F.pop()
    return F


def crowding(pop, front):
    for i in front:
        pop[i].crowd = 0.0
    for o in range(2):
        order = sorted(front, key=lambda i: pop[i].objectives()[o])
        pop[order[0]].crowd = pop[order[-1]].crowd = float("inf")
        lo, hi = pop[order[0]].objectives()[o], pop[order[-1]].objectives()[o]
        span = (hi - lo) or 1.0
        for k in range(1, len(order) - 1):
            pop[order[k]].crowd += (pop[order[k + 1]].objectives()[o]
                                    - pop[order[k - 1]].objectives()[o]) / span


def dedup(inds):
    seen, out = set(), []
    for i in inds:
        s = i.genome.signature()
        if s not in seen:
            seen.add(s)
            out.append(i)
    return out


# ---------------------------------------------------------------------------
def load_cache(path):
    d = {}
    for split, xk, yk in (("train", "Xtr", "ytr"), ("val", "Xval", "yval"),
                          ("test", "Xte", "yte")):
        d[xk] = np.load(os.path.join(path, f"{split}_states.npy"))
        d[yk] = np.load(os.path.join(path, f"{split}_labels.npy"))
    meta = json.load(open(os.path.join(path, "meta.json")))
    return d, meta


def main():
    ap = argparse.ArgumentParser(description="QCNN architecture search on cached states.")
    ap.add_argument("--data", required=True, help="dir produced by encode_states.py")
    ap.add_argument("--pop", type=int, default=12)
    ap.add_argument("--gen", type=int, default=10)
    ap.add_argument("--min-layers", type=int, default=1)
    ap.add_argument("--max-layers", type=int, default=6)
    ap.add_argument("--target-qubits", type=int, default=2,
                    help="how many qubits are measured (2 -> 4 outcomes)")
    ap.add_argument("--free-pooling", action="store_true",
                    help="do not force the reduction; measure the first "
                         "--target-qubits of whatever survives")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.003)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--l1-gate", type=float, default=5e-4)
    ap.add_argument("--l2-param", type=float, default=1e-3)
    ap.add_argument("--use-head", action="store_true",
                    help="add a 4->K linear head instead of direct outcome mapping")
    ap.add_argument("--pareto-front", dest="pareto", action="store_true", default=False)
    ap.add_argument("--p-crossover", type=float, default=0.6)
    ap.add_argument("--final-epochs", type=int, default=100,
                    help="epochs for re-benchmarking the survivors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default="experiments/qcnn_search")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    rng = random.Random(a.seed)
    np.random.seed(a.seed)
    os.makedirs(a.outdir, exist_ok=True)

    data, meta = load_cache(a.data)
    K = int(meta["n_classes"])
    configure(8, a.target_qubits, a.min_layers, a.max_layers, a.free_pooling)
    chance = float(meta["chance"])
    print(f"[data] {a.data}")
    print(f"       encoder: {meta['encoder_summary']}")
    print(f"       merge={meta['merge']} | {K} classes | chance={chance:.3f} | "
          f"train {len(data['ytr'])} / val {len(data['yval'])} / test {len(data['yte'])}")
    print(f"       mean |chi| train={meta.get('train_mean_frobenius', float('nan')):.4f}")
    mode = "PARETO (bal_acc, gates)" if a.pareto else \
           f"SINGLE (bal_acc - {a.l1_gate}*gates - {a.l2_param}*params)"
    print(f"[run ] mode={mode} | device={device} | pop={a.pop} gen={a.gen} "
          f"| layers {a.min_layers}-{a.max_layers} | epochs={a.epochs}")

    print(f"       readout {a.target_qubits} qubit(s) = {G.n_outcomes()} outcomes | "
          f"{'FREE pooling (trace out the rest)' if a.free_pooling else 'forced reduction'}")
    if K != G.n_outcomes():
        print(f"[warn] readout gives {G.n_outcomes()} outcomes but data has {K} classes; "
              f"pass --use-head or adjust --target-qubits.")

    cfg = TrainConfig(epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                      seed=a.seed, use_head=a.use_head,
                      l1_gate=a.l1_gate, l2_param=a.l2_param)

    def score(ind):
        r = train_and_score(ind.genome, data, cfg, device=device, n_classes=K)
        ind.bal_acc, ind.gates, ind.params = r["bal_acc"], r["gates"], r["params"]
        ind.fitness = r["fitness"]
        ind.info = r
        return ind

    t0 = time.time()
    pop = dedup([Ind(genome=random_genome(rng, a.min_layers, a.max_layers))
                 for _ in range(a.pop)])
    for i in pop:
        score(i)

    log = []
    for gen in range(a.gen):
        kids = []
        while len(kids) < a.pop:
            if rng.random() < a.p_crossover and len(pop) >= 2:
                p1, p2 = rng.sample(pop, 2)
                c = crossover(p1.genome, p2.genome, rng)
            else:
                c = mutate(rng.choice(pop).genome, rng)
            kids.append(Ind(genome=c))
        seen = {i.genome.signature() for i in pop}
        kids = [k for k in kids if k.genome.signature() not in seen]
        for k in kids:
            score(k)

        comb = pop + kids
        if a.pareto:
            F = fronts_of(comb)
            nxt, i = [], 0
            while i < len(F) and len(nxt) + len(F[i]) <= a.pop:
                crowding(comb, F[i])
                nxt.extend(comb[j] for j in F[i])
                i += 1
            if len(nxt) < a.pop and i < len(F):
                crowding(comb, F[i])
                rest = sorted(F[i], key=lambda j: comb[j].crowd, reverse=True)
                nxt.extend(comb[j] for j in rest[:a.pop - len(nxt)])
            pop = nxt
            best = max(pop, key=lambda i: i.bal_acc)
        else:
            pop = sorted(comb, key=lambda i: -i.fitness)[:a.pop]
            best = pop[0]

        row = {"gen": gen, "best_bal_acc": best.bal_acc, "best_fitness": best.fitness,
               "best_gates": best.gates, "best_params": best.params,
               "best_layers": len(best.genome.layers),
               "elapsed_s": round(time.time() - t0, 1)}
        log.append(row)
        print(f"gen {gen:2d} | best_acc={best.bal_acc:.4f} "
              f"(fit={best.fitness:.4f}, {len(best.genome.layers)}L, "
              f"{best.gates} 2q, {best.params}p) | {row['elapsed_s']}s")

    # ---- re-benchmark survivors with a longer training run -----------------
    finalists = dedup(pop)[:min(6, len(pop))]
    print(f"\n[re-benchmarking {len(finalists)} finalists at {a.final_epochs} epochs]")
    fcfg = TrainConfig(epochs=a.final_epochs, lr=a.lr, batch_size=a.batch_size,
                       seed=a.seed, use_head=a.use_head,
                       l1_gate=a.l1_gate, l2_param=a.l2_param)
    for ind in finalists:
        r = train_and_score(ind.genome, data, fcfg, device=device, n_classes=K)
        ind.bal_acc, ind.fitness, ind.info = r["bal_acc"], r["fitness"], r
    finalists.sort(key=lambda i: -(i.bal_acc if a.pareto else i.fitness))
    winner = finalists[0]

    print(f"\n{'FINALISTS (re-benchmarked, validation)':^72}")
    print(f"  chance = {chance:.3f}")
    print(f"  {'bal_acc':>8} {'fitness':>8} {'2q':>5} {'params':>7}  architecture")
    for i in finalists:
        mark = "  <-- selected" if i is winner else ""
        print(f"  {i.bal_acc:>8.4f} {i.fitness:>8.4f} {i.gates:>5} {i.params:>7}  "
              f"{i.genome.summary()}{mark}")

    out = {"args": vars(a), "meta": meta, "log": log,
           "finalists": [{"bal_acc": i.bal_acc, "fitness": i.fitness,
                          "gates": i.gates, "params": i.params,
                          "summary": i.genome.summary(),
                          "genome": i.genome.to_dict()} for i in finalists],
           "winner": winner.genome.to_dict(), "winner_bal_acc": winner.bal_acc}
    with open(os.path.join(a.outdir, "qcnn_search_result.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=float)
    with open(os.path.join(a.outdir, "best_qcnn.json"), "w") as fh:
        json.dump(winner.genome.to_dict(), fh, indent=2)

    try:
        from draw_qcnn_circuit import build_qcnn_columns, draw as draw_qcnn_circuit
        from plot_safety import safe_render
        gd = winner.genome.to_dict()
        cols, discarded_at, final_active, nq = build_qcnn_columns(gd)  # pure Python, safe in-process
        circuit_path = os.path.join(a.outdir, "best_qcnn_circuit.png")
        safe_render("circuit image", draw_qcnn_circuit, gd, cols, discarded_at,
                   final_active, nq, circuit_path)
    except Exception as e:
        print("circuit image skipped:", e)

    print(f"\nselected: val bal_acc={winner.bal_acc:.4f} -> "
          f"{os.path.join(a.outdir, 'best_qcnn.json')}")
    print("run qcnn_evaluate.py for the held-out test number.")


if __name__ == "__main__":
    main()