#!/usr/bin/env python3
# ----------------------------------------------------------------
# frobenius_summary.py  —  summary table and figure for the Frobenius
# (reconstruction) search, all four datasets (paper Sec. "Reconstruction
# search", Table "tab:frobenius", Fig. "fig:frobenius").
#
# The Frobenius search itself runs from frobenius_measure/<ds>_frobenius/
# (PyTorch, run_pipeline.sh: search -> sweep_front -> aggregate_seeds ->
# evaluate -> encode_states -> qcnn_search -> qcnn_evaluate). This script
# only READS its result folders:
#
#   frobenius_measure/<ds>_frobenius_results/<p>_seed<s>/
#       search_result.json     final front (re-benchmarked), winner = knee
#       search.log             per-generation progress, run time
#       cost_signal_curve.csv  every front circuit refitted on train and test:
#                              Frobenius distance and linear-probe accuracy
#   frobenius_measure/<ds>_frobenius_results/qcnn_*/
#       qcnn_search_result.json  QCNN search (VALIDATION accuracy, selection-biased)
#       qcnn_evaluate.log        QCNN test accuracy (written when you pipe
#                                qcnn_evaluate.py into it; optional)
#
# For every dataset it reports: the probe on the exact (raw) targets, the
# saturation point g* (same rule as aggregate_seeds.py: the smallest gate
# count whose suffix-mean probe is within --tolerance of the raw-target
# probe), the probe and Frobenius distance for cheap (1-5 gates) and
# expensive (>= 26 gates) circuits, the knee circuits, run times, and the
# QCNN results if present, and the between-class signal B of the exact
# targets (Proposition "class-signal stability"), computed with each suite's
# own preprocessing when its data folder is available.
#
# Usage (from the repository root):
#   python reporting/frobenius_summary.py
#   python reporting/frobenius_summary.py --root frobenius_measure --out-fig paper_figures
# Output: reports/frobenius_summary.md/.json, reports/paper/frobenius_table.tex,
#         <out-fig>/fig7_frobenius.pdf
# ----------------------------------------------------------------
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

import numpy as np

# dataset -> (folder name, seed-folder prefix, QCNN folder name or None)
DATASETS = {"bloodmnist": ("blood", "blo", None),
            "breastmnist": ("breast", "b", "qcnn_breast"),
            "dermamnist": ("derma", "der", "qcnn_derma"),
            "retinamnist": ("retina", "r", "qcnn_retina4")}
CHEAP, EXPENSIVE = (1, 5), 26          # gate ranges for the "cheap" and "expensive" columns
COLOR = "#0e9384"                      # Frobenius search (validated with the paper palette)


def pretty(ds):
    """'bloodmnist' -> 'BloodMNIST'."""
    return ds[:-5].capitalize() + "MNIST"


def read_curve(path):
    """cost_signal_curve.csv as a list of dicts with numeric values, sorted by gate count."""
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (int(v) if k in ("gates", "cry", "params", "depth") else float(v))
                         for k, v in r.items()})
    return sorted(rows, key=lambda r: r["gates"])


def saturation_point(rows, tol):
    """Smallest gate count whose suffix-mean probe is within tol of the raw-target probe
    (identical to aggregate_seeds.py); None if never reached."""
    base = rows[0]["probe_raw"]
    for i, r in enumerate(rows):
        if st.mean(x["probe_encoded"] for x in rows[i:]) >= base - tol:
            return r["gates"]
    return None


def mean_sd(v):
    """(mean, sd) of a list; sd = 0 for a single value; (nan, nan) if empty."""
    v = [x for x in v if x is not None]
    if not v:
        return float("nan"), float("nan")
    return st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0)


def log_info(path):
    """Number of classes, chance level and total run time from a search.log."""
    txt = open(path).read()
    m = re.search(r"(\d+) classes \| chance=([\d.]+)", txt)
    t = re.findall(r"\|\s*([\d.]+)s\s*$", txt, re.M)
    return int(m.group(1)), float(m.group(2)), (float(t[-1]) if t else float("nan"))


def qcnn_info(qdir):
    """QCNN validation accuracy (search) and test accuracy (qcnn_evaluate.log), if present."""
    out = {}
    p = os.path.join(qdir, "qcnn_search_result.json")
    if os.path.exists(p):
        r = json.load(open(p))
        out.update(val_bal_acc=r["winner_bal_acc"], n_finalists=len(r["finalists"]),
                   winner=r["finalists"][0]["summary"] if r.get("finalists") else "")
    p = os.path.join(qdir, "qcnn_evaluate.log")
    if os.path.exists(p):
        m = re.search(r"QCNN TEST balanced accuracy : ([\d.]+) \+/- ([\d.]+) \(n=(\d+)\)", open(p).read())
        if m:
            out.update(test_mean=float(m.group(1)), test_sd=float(m.group(2)), test_n=int(m.group(3)))
    return out


def between_class_signal(root, folder, merge):
    """B = sum_c pi_c |mean_c - mean|^2 of the exact targets on the test split, computed
    with the suite's own loader (same preprocessing as its search); None if its data
    folder is not available (e.g. bloodmnist train/test are not in the repository)."""
    import importlib
    import sys
    suite = os.path.abspath(os.path.join(root, f"{folder}_frobenius"))
    data_dir = os.path.join(suite, f"{folder}mnist")
    if not os.path.exists(os.path.join(data_dir, "test_images.npy")):
        return None
    sys.path.insert(0, suite)
    for m in ("config", "data_local"):           # every suite has its own copies
        sys.modules.pop(m, None)
    try:
        d = importlib.import_module("data_local").load(data_dir, channel="gray", size=16,
                                                        center=True, merge=merge)
    finally:
        sys.path.remove(suite)
    X, y = d["Xte"].astype(float), d["yte"]
    mu = X.mean(0)
    return float(sum((y == c).mean() * np.sum((X[y == c].mean(0) - mu) ** 2) for c in np.unique(y)))


def summarise(root, tol):
    """Per-dataset summary (dict) plus the raw curves for plotting."""
    summary, curves = {}, {}
    for ds, (folder, pre, qname) in DATASETS.items():
        rdir = os.path.join(root, f"{folder}_frobenius_results")
        seeds = sorted(glob.glob(os.path.join(rdir, f"{pre}_seed*")),
                       key=lambda d: int(d.rsplit("seed", 1)[1]))
        if not seeds:
            print(f"  skipped {ds}: no results in {rdir}")
            continue
        per_seed, cs = [], []
        for d in seeds:
            res = json.load(open(os.path.join(d, "search_result.json")))
            rows = read_curve(os.path.join(d, "cost_signal_curve.csv"))
            K, chance, secs = log_info(os.path.join(d, "search.log"))
            cs.append(rows)
            per_seed.append({"seed": res["search"]["seed"], "front": len(res["candidates"]),
                             "gates_max": max(c["gates"] for c in res["candidates"]),
                             "knee_gates": res["winner"]["gates"], "knee_frob": res["winner"]["frob"],
                             "saturation": saturation_point(rows, tol), "seconds": secs,
                             "settings": res["search"], "data": res["data"]})
        allr = [r for rows in cs for r in rows]
        cheap = [r for r in allr if CHEAP[0] <= r["gates"] <= CHEAP[1]]
        dear = [r for r in allr if r["gates"] >= EXPENSIVE]
        sats = [s["saturation"] for s in per_seed]
        s = {"classes": K, "chance": chance, "merge": per_seed[0]["data"]["merge"],
             "probe_raw": cs[0][0]["probe_raw"], "n_seeds": len(seeds),
             "front_sizes": [p["front"] for p in per_seed],
             "saturation_per_seed": sats, "saturation": mean_sd(sats),
             "n_saturated": sum(x is not None for x in sats),
             "probe_cheap": mean_sd([r["probe_encoded"] for r in cheap]), "n_cheap": len(cheap),
             "probe_expensive": mean_sd([r["probe_encoded"] for r in dear]), "n_expensive": len(dear),
             "frob_cheap": mean_sd([r["frob_test"] for r in cheap]),
             "frob_expensive": mean_sd([r["frob_test"] for r in dear]),
             "knee_gates": [p["knee_gates"] for p in per_seed],
             "hours": mean_sd([p["seconds"] / 3600 for p in per_seed]),
             "per_seed": per_seed}
        s["B_test"] = between_class_signal(root, folder, s["merge"])
        if qname:
            s["qcnn"] = qcnn_info(os.path.join(rdir, qname))
        summary[ds], curves[ds] = s, cs
    return summary, curves


def fmt(ms, d=3):
    """'m ± s' with d decimals (or '--')."""
    m, s = ms
    return "--" if np.isnan(m) else f"{m:.{d}f} $\\pm$ {s:.{d}f}"


def latex_table(summary, tol):
    """LaTeX table (tab:frobenius)."""
    head = (r"\begin{table*}[t]" "\n"
            r"\caption{Frobenius (reconstruction) search, five search seeds per dataset. Probe: "
            r"balanced accuracy of a class-balanced logistic regression on the encoded test states "
            r"(raw: on the exact targets). $g^\ast$: smallest two-qubit gate count from which the "
            r"mean probe of all costlier front circuits is within " + f"{tol:g}" +
            r" of the raw-target probe (mean $\pm$ s.d. over seeds that reach it). Cheap: front "
            r"circuits with 1--5 gates; expensive: $\ge" + str(EXPENSIVE) + r"$ gates, pooled over "
            r"seeds. Knee: circuit selected by the search (cheapest within 0.015 of the best "
            r"Frobenius distance). $B$: between-class signal of the exact test targets "
            r"(Proposition~\ref{prop:budget}); not computed where the data are not in the repository.}" "\n"
            r"\label{tab:frobenius}" "\n" r"\begin{ruledtabular}" "\n"
            r"\begin{tabular}{lccccccccc}" "\n"
            r"Dataset & $K$ & $B$ & Probe, raw & $g^\ast$ & Probe, cheap & Probe, expensive & "
            r"$\lVert\chi\rVert_F$, cheap & $\lVert\chi\rVert_F$, expensive & Knee gates\\" "\n"
            r"\hline" "\n")
    rows = []
    for ds, s in summary.items():
        g = s["saturation"]
        gs = "--" if np.isnan(g[0]) else f"{g[0]:.1f} $\\pm$ {g[1]:.1f}"
        if s["n_saturated"] < s["n_seeds"]:
            gs += f" ($n={s['n_saturated']}$)"
        name = pretty(ds)
        if s["merge"] != "none":
            name += f" ({s['merge']})"
        B = "--" if s["B_test"] is None else f"{s['B_test']:.3f}"
        rows.append(f"{name} & {s['classes']} & {B} & {s['probe_raw']:.3f} & {gs} & {fmt(s['probe_cheap'])} & "
                    f"{fmt(s['probe_expensive'])} & {fmt(s['frob_cheap'], 2)} & {fmt(s['frob_expensive'], 2)} & "
                    f"{min(s['knee_gates'])}--{max(s['knee_gates'])}\\\\")
    return head + "\n".join(rows) + "\n" + r"\end{tabular}" "\n" r"\end{ruledtabular}" "\n" r"\end{table*}" "\n"


def figure(summary, curves, path):
    """fig7_frobenius.pdf: Frobenius distance (top) and probe accuracy (bottom) against
    two-qubit gates, one column per dataset. Faint lines: individual seeds; markers:
    means (+- s.d.) in gate bins; dashed: probe on the exact targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    bins = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40), (41, 100)]
    n = len(summary)
    fig, axes = plt.subplots(2, n, figsize=(7.0, 3.3), sharex=True)
    axes = np.atleast_2d(axes).reshape(2, n)
    for j, (ds, s) in enumerate(summary.items()):
        cs = curves[ds]
        for i, key in enumerate(("frob_test", "probe_encoded")):
            ax = axes[i, j]
            for rows in cs:
                ax.plot([r["gates"] for r in rows], [r[key] for r in rows], "-", color=COLOR,
                        alpha=0.22, lw=0.8)
            xs, ms, es = [], [], []
            for lo, hi in bins:
                v = [r[key] for rows in cs for r in rows if lo <= r["gates"] <= hi]
                g = [r["gates"] for rows in cs for r in rows if lo <= r["gates"] <= hi]
                if v:
                    xs.append(np.exp(np.mean(np.log(g)))); ms.append(np.mean(v)); es.append(np.std(v))
            ax.errorbar(xs, ms, yerr=es, fmt="o-", color=COLOR, ms=3.5, lw=1.4, capsize=2)
            ax.set_xscale("log")
            ax.set_xticks([1, 3, 10, 30, 100]); ax.set_xticklabels(["1", "3", "10", "30", "100"])
            if i == 0:
                ax.set_title(pretty(ds))
                ax.set_ylim(0, 1.0)
            else:
                # chance (1/K) is given in the table; drawing it would flatten the curves
                ax.axhline(s["probe_raw"], ls="--", color="#333333", lw=0.9)
                ax.text(1.0, s["probe_raw"], " exact targets", fontsize=6, va="bottom", ha="left",
                        color="#333333")
                lo = min(r[key] for rows in cs for r in rows)
                ax.set_ylim(lo - 0.02, s["probe_raw"] + 0.04)
                ax.set_xlabel("two-qubit gates")
        axes[0, 0].set_ylabel(r"Frobenius dist. $\|\chi\|_F$")
        axes[1, 0].set_ylabel("probe bal. acc.")
    fig.tight_layout(h_pad=0.6, w_pad=0.6)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def markdown(summary, tol):
    """Human-readable report."""
    out = ["# Frobenius (reconstruction) search: summary\n",
           f"Saturation rule: suffix-mean probe within {tol:g} of the probe on exact targets.\n",
           "| dataset | K | B (test) | probe raw | g* per seed | probe 1-5 gates | probe >= 26 gates | "
           "Frobenius >= 26 gates | knee gates | hours/seed | QCNN val | QCNN test |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for ds, s in summary.items():
        q = s.get("qcnn", {})
        qv = f"{q['val_bal_acc']:.3f}" if "val_bal_acc" in q else "--"
        qt = f"{q['test_mean']:.3f} ± {q['test_sd']:.3f}" if "test_mean" in q else "pending"
        B = "n/a" if s["B_test"] is None else f"{s['B_test']:.4f}"
        out.append(f"| {ds} ({s['merge']}) | {s['classes']} | {B} | {s['probe_raw']:.4f} | {s['saturation_per_seed']} | "
                   f"{s['probe_cheap'][0]:.3f} ± {s['probe_cheap'][1]:.3f} | "
                   f"{s['probe_expensive'][0]:.3f} ± {s['probe_expensive'][1]:.3f} | "
                   f"{s['frob_expensive'][0]:.3f} ± {s['frob_expensive'][1]:.3f} | {s['knee_gates']} | "
                   f"{s['hours'][0]:.2f} | {qv} | {qt if 'qcnn' in s else 'not run'} |")
    return "\n".join(out) + "\n"


def main():
    """Command line: result root, tolerance and output folders."""
    p = argparse.ArgumentParser(description="summary of the Frobenius search results")
    p.add_argument("--root", default="frobenius_measure")
    p.add_argument("--tolerance", type=float, default=0.03, help="as in aggregate_seeds.py")
    p.add_argument("--reports", default="reports")
    p.add_argument("--out-fig", default=os.path.join("paper", "figures"))
    a = p.parse_args()
    summary, curves = summarise(a.root, a.tolerance)
    if not summary:
        raise SystemExit(f"no Frobenius results under {a.root}")
    os.makedirs(os.path.join(a.reports, "paper"), exist_ok=True)
    with open(os.path.join(a.reports, "frobenius_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=float)
    with open(os.path.join(a.reports, "frobenius_summary.md"), "w") as fh:
        fh.write(markdown(summary, a.tolerance))
    with open(os.path.join(a.reports, "paper", "frobenius_table.tex"), "w") as fh:
        fh.write(latex_table(summary, a.tolerance))
    figure(summary, curves, os.path.join(a.out_fig, "fig7_frobenius.pdf"))
    print(markdown(summary, a.tolerance))
    print(f"wrote {a.reports}/frobenius_summary.{{md,json}}, {a.reports}/paper/frobenius_table.tex, "
          f"{a.out_fig}/fig7_frobenius.pdf")


if __name__ == "__main__":
    main()
