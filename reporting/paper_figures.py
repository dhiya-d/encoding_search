#!/usr/bin/env python3
# ----------------------------------------------------------------
# paper_figures.py  —  vector figures for the paper (REVTeX column widths).
#
#   fig1_pipeline.pdf        schematic of the two encoding families
#   fig2_diagnostics.pdf     phase artefact + class-signal budget      (values below)
#   fig3_objective.pdf       objective study                (reports/objective_<ds>.json)
#   fig4_robustness.pdf      per-seed effects                (reports/multiseed_<ds>.json)
#   fig5_fourier.pdf         amplitude vs Fourier encodings           (values below)
#   fig6_noise.pdf           noise study                     (reports/noise_<ds>.json)
#
# Usage (from the repository root):  python reporting/paper_figures.py --dataset bloodmnist
# Output: paper/figures/*.pdf. Missing report files are skipped with a note.
# The hard-coded values for figs 2 and 5 are copied from the diagnostic and
# downstream reports (bloodmnist); check them against your reports before submission.
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

COL = 3.4      # REVTeX single column width (inches)
WIDE = 7.0     # two-column width
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8,
                     "axes.titlesize": 8, "legend.fontsize": 6.5, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})
C = {"ret": "#c0504d", "fidm": "#3b6ea5", "fidb": "#8fb3dc", "lit": "#4a9a5b", "hand": "#222222",
     "rand": "#9a9a9a", "four": "#7a5aa6", "cls": "#555555"}

# ---- values copied from the reports (bloodmnist) ----------------------------
SIGN = {"labels": ["linear probe,\nstates as produced", "exact images,\nsame random signs",
                   "linear probe,\nsigns aligned", "fidelity kernel,\nencoded states",
                   "fidelity kernel,\nexact states"],
        "auc": [0.490, 0.506, 0.803, 0.811, 0.905]}                         # diag a, uncentered, 31 CNOTs
BUDGET = {"datasets": ["blood", "breast", "derma", "retina"],               # diag c + diag a
          "B_unc": [0.0063, 0.0042, 0.0014, 0.0018], "inf_unc": [0.031, 0.050, 0.016, 0.151],
          "B_cen": [0.254, 0.026, 0.086, 0.071], "inf_cen": [0.422, 0.263, 0.235, 0.256]}
JOURNEY = [("hand-designed amplitude circuit (31 CNOTs) + HEA-8", 0.520, C["hand"]),
           ("searched amplitude circuit (19 CNOTs) + HEA-8", 0.574, C["ret"]),
           ("exact amplitude states + best trained classifier", 0.637, C["fidm"]),
           ("Fourier search pick (16 CNOTs) + HEA-8", 0.710, C["four"]),
           ("Fourier search pick, fidelity kernel", 0.788, C["four"]),
           ("classical RBF SVM, same 32 features", 0.813, C["cls"])]
FOURIER = {"models": ["kernel", "re-upl.", "QCNN", "HEA-8"],
           "pick": [0.788, 0.687, 0.648, 0.710], "ref": [0.778, 0.635, 0.640, 0.694]}


def load(path):
    """JSON file contents, or None (with a note) if the file does not exist."""
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    print(f"  skipped: {path} not found")
    return None


def box(ax, x, y, w, h, text, color, fs=6.5):
    """Rounded, filled box with centred white text (schematic figures)."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.05",
                                fc=color, ec="none"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color="white", fontsize=fs)


def arrow(ax, x0, y0, x1, y1):
    """Arrow between two points (schematic figures)."""
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=7, color="#444", lw=0.8))


def fig1(out):
    """fig1_pipeline.pdf (fig:pipeline): schematic of the two encoding families."""
    fig, ax = plt.subplots(figsize=(WIDE, 1.9))
    ax.set_xlim(0, 10); ax.set_ylim(0, 2.55); ax.axis("off")
    rows = [(1.35, "amplitude", "16×16 grayscale\nimage (256 values)",
             "approximate amplitude\nencoding: circuit fitted\nper image (fidelity)", C["ret"]),
            (0.15, "Fourier", "low-frequency DCT\ncoefficients (RGB,\nd ≤ 32 angles)",
             "angle encoding S(x):\none fixed circuit,\nspectrum from gates", C["four"])]
    for y, lab, pre, enc, col in rows:
        box(ax, 0.0, y, 1.3, 0.8, "MedMNIST\nimage", "#8a8a8a")
        box(ax, 1.6, y, 1.9, 0.8, pre, "#5b7fa8")
        box(ax, 3.8, y, 2.2, 0.8, enc, col)
        arrow(ax, 1.3, y + 0.4, 1.6, y + 0.4); arrow(ax, 3.5, y + 0.4, 3.8, y + 0.4)
        arrow(ax, 6.0, y + 0.4, 6.35, 1.15)
    box(ax, 6.35, 0.55, 1.75, 1.2, "measurement-based\nclassifier\n(kernel SVM, HEA,\nQCNN, re-uploading)", "#4a9a5b")
    arrow(ax, 8.1, 1.15, 8.4, 1.15)
    box(ax, 8.4, 0.75, 1.6, 0.8, "class\nprediction", "#8a8a8a")
    ax.text(4.9, 2.38, "searched (NSGA-II, HierarQcal genomes)", ha="center", fontsize=6.5, style="italic")
    fig.savefig(os.path.join(out, "fig1_pipeline.pdf")); plt.close(fig)


def fig2(out):
    """fig2_diagnostics.pdf (fig:diagnostics): sign artefact and class-signal budget
    (values copied into SIGN and BUDGET above).
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.2), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    cols = [C["ret"], C["ret"], C["lit"], C["lit"], C["rand"]]
    y = np.arange(len(SIGN["auc"]))[::-1]
    ax.barh(y, SIGN["auc"], color=cols)
    for yi, v in zip(y, SIGN["auc"]):
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=6.5)
    ax.axvline(0.5, color="k", ls=":", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(SIGN["labels"], fontsize=6.3)
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("test AUC")
    ax.set_title("(a) global-sign ambiguity (fidelity 0.97)", loc="left")
    ax = axes[1]
    x = np.arange(4); w = 0.2
    ax.bar(x - 1.5 * w, BUDGET["B_unc"], w, color=C["ret"], label="$B$, uncentered")
    ax.bar(x - 0.5 * w, BUDGET["inf_unc"], w, color="#e6a3a1", label=r"$1-\bar F$, uncentered")
    ax.bar(x + 0.5 * w, BUDGET["B_cen"], w, color=C["fidm"], label="$B$, centered")
    ax.bar(x + 1.5 * w, BUDGET["inf_cen"], w, color=C["fidb"], label=r"$1-\bar F$, centered")
    ax.set_yscale("log"); ax.set_xticks(x); ax.set_xticklabels(BUDGET["datasets"])
    ax.set_ylabel("fraction of state norm")
    ax.set_title("(b) class-mean signal vs infidelity", loc="left")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.14), frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig2_diagnostics.pdf")); plt.close(fig)


KIND = {"retention": ("retention search", C["ret"], "o"), "fidelity_matched": ("fidelity search, matched", C["fidm"], "s"),
        "fidelity_best": ("fidelity search, own pick", C["fidb"], "s"), "literature": ("literature ansätze", C["lit"], "^"),
        "handref": ("hand-designed reference", C["hand"], "D"), "random": ("random circuits", C["rand"], ".")}


def fig3(out, ds, reports):
    """fig3_objective.pdf (fig:objective): objective study, from reports/objective_<ds>.json."""
    o = load(os.path.join(reports, f"objective_{ds}.json"))
    if not o:
        fig, ax = plt.subplots(figsize=(WIDE, 2.4))
        ax.text(0.5, 0.5, "objective study figure: run paper_figures.py with reports/objective_<dataset>.json",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(os.path.join(out, "fig3_objective.pdf")); plt.close(fig)
        return
    arms = o["arms"]
    clf = "hea8" if any("hea8" in v.get("clf_bal_acc", {}) for v in arms.values()) else None
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.4))
    have_random = any(v["kind"] == "random" for v in arms.values())
    if not have_random and "kernel AUC" in o.get("random", {}):
        r = o["random"]["kernel AUC"]
        axes[0].axhspan(r["mean"] - r["std"], r["mean"] + r["std"], color=C["rand"], alpha=0.25,
                        label="random circuits (mean ± s.d.)")
    for kind, (lab, col, mk) in KIND.items():
        vs = [v for v in arms.values() if v["kind"] == kind]
        if not vs:
            continue
        axes[0].scatter([v["n2q"] for v in vs], [v["kernel_auc"] for v in vs], c=col, marker=mk,
                        s=22 if kind != "random" else 12, label=lab, edgecolor="k", lw=0.3, zorder=3)
        axes[1].scatter([v["test_fidelity"] for v in vs], [v["kernel_auc"] for v in vs], c=col, marker=mk,
                        s=22 if kind != "random" else 12, edgecolor="k", lw=0.3, zorder=3)
    ceil = np.nanmean([v.get("ceiling_auc", np.nan) for v in arms.values()]) if any(
        "ceiling_auc" in v for v in arms.values()) else o.get("ceiling_auc")
    for ax in axes:
        if ceil:
            ax.axhline(ceil, color="k", ls=":", lw=0.8)
    axes[0].set_xlabel("CNOT gates"); axes[0].set_ylabel("fidelity-kernel SVM AUC (test)")
    axes[0].set_title("(a) class information vs cost", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    axes[1].legend(h, l, frameon=False, loc="upper left", fontsize=6)
    axes[1].set_xlabel(r"mean test fidelity $\bar F$")
    sp = o.get("spearman", {}).get("fidelity_vs_kernel_auc")
    axes[1].set_title("(b) fidelity vs class information" + (f" (Spearman {sp:.2f})" if sp else ""), loc="left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig3_objective.pdf")); plt.close(fig)


def _ci_panel(ax, entries, key, clfs, title, labels=None):
    """Bar panel of paired differences with 95% bootstrap intervals, per classifier and seed."""
    w = 0.8 / max(1, len(entries))
    cols = ["#3b6ea5", "#e08a2e", "#4a9a5b", "#7a5aa6"]
    for i, e in enumerate(entries):
        xs, d, lo, hi = [], [], [], []
        for j, c in enumerate(clfs):
            t = e["classifiers"].get(c, {}).get(key, {}).get("bal_acc")
            if t:
                xs.append(j + (i - (len(entries) - 1) / 2) * w)
                d.append(t["diff"]); lo.append(t["diff"] - t["ci_low"]); hi.append(t["ci_high"] - t["diff"])
        ax.errorbar(xs, d, yerr=[lo, hi], fmt="o", ms=3, capsize=2, lw=0.8, color=cols[i % 4],
                    label=f"seed {e['seed']}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(clfs))); ax.set_xticklabels(labels or clfs)
    ax.set_title(title, loc="left")


def fig4(out, ds, reports):
    """fig4_robustness.pdf (fig:robustness): paired differences per search seed, from
    reports/multiseed_<ds>.json.
    """
    m = load(os.path.join(reports, f"multiseed_{ds}.json"))
    if not m:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.2))
    if "aae" in m:
        clfs = [c for c in ["hea", "hea8", "hea8_probs", "qcnn"]
                if any(c in e["classifiers"] for e in m["aae"]["entries"])]
        _ci_panel(axes[0], m["aae"]["entries"], "vs_default", clfs,
                  "(a) searched amplitude circuit − hand-designed ref.",
                  [c.replace("_probs", "\n(probs)") for c in clfs])
        axes[0].set_ylabel(r"$\Delta$ balanced accuracy (95% CI)")
        axes[0].legend(frameon=False, fontsize=6)
    if "fourier" in m:
        clfs = ["reupload", "qcnn", "hea8"]
        _ci_panel(axes[1], m["fourier"]["entries"], "vs_reference", clfs,
                  "(b) Fourier search pick − best literature encoding", ["re-uploading", "QCNN", "HEA-8"])
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig4_robustness.pdf")); plt.close(fig)


def fig5(out):
    """fig5_fourier.pdf (fig:fourier): amplitude vs. Fourier encodings (values copied into
    JOURNEY and FOURIER above).
    """
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.2), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    y = np.arange(len(JOURNEY))[::-1]
    for yi, (lab, v, col) in zip(y, JOURNEY):
        ax.barh(yi, v, color=col)
        ax.text(v + 0.005, yi, f"{v:.3f}", va="center", fontsize=6.5)
    ax.set_yticks(y); ax.set_yticklabels([j[0] for j in JOURNEY], fontsize=6.2)
    ax.axvline(0.125, color="k", ls=":", lw=0.8)
    ax.set_xlim(0, 0.95); ax.set_xlabel("test balanced accuracy (8 classes)")
    ax.set_title("(a) amplitude vs Fourier encodings", loc="left")
    ax = axes[1]
    x = np.arange(4); w = 0.38
    ax.bar(x - w / 2, FOURIER["pick"], w, color=C["four"], label="search pick (16 CNOTs)")
    ax.bar(x + w / 2, FOURIER["ref"], w, color="#b9a7d6", label="best literature encoding (16 CNOTs)")
    ax.axhline(0.813, color=C["cls"], ls="--", lw=0.8, label="classical RBF SVM")
    ax.set_xticks(x); ax.set_xticklabels(FOURIER["models"], fontsize=6.5)
    ax.set_ylim(0.55, 0.85); ax.set_ylabel("test balanced accuracy")
    ax.set_title("(b) Fourier search vs literature", loc="left")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig5_fourier.pdf")); plt.close(fig)


def fig6(out, ds, reports):
    """fig6_noise.pdf (fig:noise): test metrics vs. gate-error rate, from
    reports/noise_<ds>.json.
    """
    n = load(os.path.join(reports, f"noise_{ds}.json"))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.5))
    if not n:
        for ax in axes:
            ax.text(0.5, 0.5, "noise study pending", ha="center", va="center", transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        fig.savefig(os.path.join(out, "fig6_noise.pdf")); plt.close(fig)
        return
    ps = n["p"]
    style = {"aae_retention": (C["ret"], "o"), "aae_literature": (C["lit"], "^"), "aae_fidelity": (C["fidm"], "s"),
             "aae_fidelity_deep": (C["fidb"], "s"), "aae_literature_deep": ("#9bc79f", "^"),
             "aae_handref": (C["hand"], "D"), "fourier_pick": (C["four"], "o"), "fourier_ref": ("#b9a7d6", "o"),
             "fourier_zero": ("#d8b4e2", "o")}
    for enc, v in n["results"].items():
        col, mk = style.get(enc, ("#777", "x"))
        pts = [(p, v[str(p)]) for p in ps if str(p) in v]
        if not pts:
            continue
        clf = next(iter(pts[0][1]["clf"]))
        n2q = pts[0][1]["n2q"]
        axes[0].errorbar([p for p, _ in pts], [r["clf"][clf]["bal_acc"] for _, r in pts],
                         yerr=[r["clf"][clf]["bal_acc_std"] for _, r in pts], marker=mk, ms=3, lw=0.9,
                         capsize=2, color=col, label=f"{enc.replace('_', ' ')} ({n2q})")
        sh = n.get("shots", {}).get(enc, {})
        smin = min((int(s) for r in sh.values() for s in r), default=None)
        if smin:
            axes[1].plot([p for p, _ in pts], [sh[str(p)][str(smin)]["clf_bal_acc"] for p, _ in pts
                                              if str(smin) in sh.get(str(p), {})],
                         marker=mk, ms=3, lw=0.9, color=col)
    for ax in axes:
        ax.set_xlabel("two-qubit depolarizing probability $p$")
        ax.set_xticks(ps)
        ax.set_xticklabels([f"{v:g}" for v in ps], fontsize=6.5)
    axes[0].set_ylabel("test balanced accuracy")
    axes[0].set_title("(a) exact readout", loc="left")
    axes[1].set_title("(b) finite number of shots (smallest budget)", loc="left")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=6, ncol=5, loc="lower center", bbox_to_anchor=(0.5, -0.09))
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig6_noise.pdf")); plt.close(fig)


def main():
    """Command line: dataset, report folder and output folder; draws all figures."""
    p = argparse.ArgumentParser(description="paper figures")
    p.add_argument("--dataset", default="bloodmnist")
    p.add_argument("--reports", default="reports")
    p.add_argument("--out", default=os.path.join("paper", "figures"))
    a = p.parse_args()
    os.makedirs(a.out, exist_ok=True)
    fig1(a.out); fig2(a.out); fig5(a.out)
    fig3(a.out, a.dataset, a.reports); fig4(a.out, a.dataset, a.reports); fig6(a.out, a.dataset, a.reports)
    print(f"figures written to {a.out}")


if __name__ == "__main__":
    main()
