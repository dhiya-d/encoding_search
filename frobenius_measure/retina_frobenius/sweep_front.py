"""Cost vs class-signal curve across a saved front.

Takes a search_result.json, walks its candidates (cheapest -> most expensive),
and for each one measures BOTH:

    reconstruction : mean ||t - s||_F on the test split
    class signal   : linear-probe balanced accuracy on the ENCODED STATES

against the probe on the raw amplitude targets. The output answers the question
the search itself cannot: at what gate count does the encoding stop preserving
the class information?

Example:
    python sweep_front.py --result experiments/r_seed0/search_result.json \
        --data-dir retinamnist --fit-steps 300 --restarts 2 \
        --outdir experiments/r_seed0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.metrics import balanced_accuracy_score      # noqa: E402

from data_local import load                              # noqa: E402
from encoder import fit_batch                            # noqa: E402
from genome import Genome                                # noqa: E402
from plot_safety import safe_render                      # noqa: E402


def probe(Xtr, ytr, Xte, yte, seed=0):
    m = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    m.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, m.predict(Xte)))


def main():
    ap = argparse.ArgumentParser(description="Cost vs class-signal sweep over a front.")
    ap.add_argument("--result", required=True, help="search_result.json")
    ap.add_argument("--data-dir", default="retinamnist")
    ap.add_argument("--merge", default="retina4", choices=("retina4",))
    ap.add_argument("--channel", default="gray", choices=("gray", "green", "rgb"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--no-center", dest="center", action="store_false", default=True)
    ap.add_argument("--fit-steps", type=int, default=300)
    ap.add_argument("--fit-lr", type=float, default=0.08)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--every", type=int, default=1,
                    help="evaluate every Nth candidate (use 2 or 3 to save time)")
    ap.add_argument("--max-images", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    res = json.load(open(a.result))
    cands = res["candidates"]
    cands = sorted(cands, key=lambda c: c["gates"])[::a.every]

    d = load(a.data_dir, channel=a.channel, size=a.size, center=a.center, merge=a.merge)
    Xtr_t = torch.as_tensor(d["Xtr"][:a.max_images])
    Xte_t = torch.as_tensor(d["Xte"][:a.max_images])
    ytr, yte = d["ytr"][:a.max_images], d["yte"][:a.max_images]

    base = probe(d["Xtr"][:a.max_images], ytr, d["Xte"][:a.max_images], yte, a.seed)
    print(f"[data] {a.data_dir} merge={a.merge} | {d['n_classes']} classes | "
          f"chance={d['chance']:.3f} | linear probe on RAW TARGETS = {base:.4f}")
    print(f"[sweep] {len(cands)} candidates, fit_steps={a.fit_steps}, "
          f"restarts={a.restarts}, device={device}\n")
    print(f"  {'gates':>6} {'cry':>4} {'frob':>7} {'overlap':>8} {'probe':>7} {'vs raw':>8}   verdict")

    rows = []
    for c in cands:
        g = Genome.from_dict(c["genome"])
        rtr = fit_batch(g, Xtr_t, steps=a.fit_steps, lr=a.fit_lr, restarts=a.restarts,
                        seed=a.seed, device=device, return_states=True)
        rte = fit_batch(g, Xte_t, steps=a.fit_steps, lr=a.fit_lr, restarts=a.restarts,
                        seed=a.seed, device=device, return_states=True)
        p = probe(rtr["states"].numpy(), ytr, rte["states"].numpy(), yte, a.seed)
        delta = p - base
        verdict = "preserved" if delta > -0.05 else ("degraded" if delta > -0.12 else "LOST")
        row = {"gates": g.two_qubit_gates(), "cry": g.n_cry(),
               "params": g.n_params(), "depth": g.depth(),
               "frob_test": rte["mean_frobenius"], "overlap_test": rte["mean_overlap"],
               "probe_encoded": p, "probe_raw": base, "delta": delta}
        rows.append(row)
        print(f"  {row['gates']:>6} {row['cry']:>4} {row['frob_test']:>7.4f} "
              f"{row['overlap_test']:>8.4f} {p:>7.4f} {delta:>+8.4f}   {verdict}")

    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        cp = os.path.join(a.outdir, "cost_signal_curve.csv")
        with open(cp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nsaved -> {cp}")
        plot_path = os.path.join(a.outdir, "cost_signal_curve.png")
        if safe_render("plot", _plot, rows, base, d["chance"], plot_path):
            print(f"plot  -> {plot_path}")


def _plot(rows, base, chance, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = [r["gates"] for r in rows]
    f = [r["frob_test"] for r in rows]
    p = [r["probe_encoded"] for r in rows]

    fig, ax1 = plt.subplots(figsize=(7.2, 4.6))
    ax1.plot(g, f, "-o", color="#5A4BD6", label=r"$\|t-s\|_F$ (test)")
    ax1.set_xlabel("two-qubit gate count")
    ax1.set_ylabel(r"mean $\|t-s\|_F$   (lower = better reconstruction)", color="#5A4BD6")
    ax1.tick_params(axis="y", labelcolor="#5A4BD6")

    ax2 = ax1.twinx()
    ax2.plot(g, p, "-s", color="#0E9384", label="probe on encoded states")
    ax2.axhline(base, ls="--", color="#B45309", lw=1.2, label=f"probe on raw targets ({base:.3f})")
    ax2.axhline(chance, ls=":", color="#C0392B", lw=1.2, label=f"chance ({chance:.2f})")
    ax2.set_ylabel("balanced accuracy   (higher = signal preserved)", color="#0E9384")
    ax2.tick_params(axis="y", labelcolor="#0E9384")
    ax2.set_ylim(0.4, 1.0)

    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], fontsize=8, loc="center right")
    ax1.set_title("Encoding cost vs reconstruction vs class signal")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
