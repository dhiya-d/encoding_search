#!/usr/bin/env python3
# ----------------------------------------------------------------
# paper_tables.py  —  LaTeX tables and a figure folder for the paper,
# built from the JSON files the experiment scripts write to reports/.
#
#   Table 1  retention-selected amplitude encoder vs default, per seed
#   Table 2  objective study: retention vs fidelity vs random vs default
#   Table 3  Fourier search pick vs best literature reference, per seed
#   Table 4  noise study
#   Table 5  best quantum vs best classical model
#
# Usage (from the repository root, after the experiments):
#   python reporting/paper_tables.py
#   python reporting/paper_tables.py --pairs bloodmnist:centered:bloodmnist pathmnist:centered:pathmnist
#
# Output: reports/paper/tables.tex   (\usepackage{booktabs})
#         reports/paper/figures/*.png and reports/paper/figures.tex
# Missing inputs are skipped with a note, so the script can be run at any time.
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import glob
import json
import os
import shutil

import numpy as np


def load(path):
    """JSON file contents, or None if the file does not exist."""
    if os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return None


def esc(s):
    """Escape LaTeX special characters (_ % &)."""
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def ci(t, key="bal_acc"):
    """Paired difference with its 95% interval; bold when significant."""
    if not t or key not in t:
        return "--"
    d = t[key]
    s = f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
    return rf"\textbf{{{s}}}" if d["significant"] else s


def ms(vals, digits=3):
    """'mean +- std' of the finite values (or the single value)."""
    v = np.array([x for x in vals if x is not None and np.isfinite(x)], float)
    if not len(v):
        return "--"
    f = f"{{:.{digits}f}}"
    return f.format(v.mean()) if len(v) == 1 else f"{f.format(v.mean())} $\\pm$ {f.format(v.std())}"


def table(caption, label, cols, header, rows, note=None, size=r"\small"):
    """LaTeX table environment (booktabs); 'MID' rows become a midrule."""
    out = [r"\begin{table}[t]", r"\centering", size,
           rf"\caption{{{caption}}}", rf"\label{{{label}}}",
           r"\resizebox{\linewidth}{!}{%" if len(header) > 6 else "",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        out.append(r"\midrule" if r == "MID" else " & ".join(str(x) for x in r) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    if len(header) > 6:
        out.append("}")
    if note:
        out.append(rf"\\[2pt]\parbox{{\linewidth}}{{\footnotesize {note}}}")
    out.append(r"\end{table}")
    return "\n".join(out) + "\n\n"


def t1(a, pairs):
    """tab:retention_vs_default: retention picks (all seeds) vs. the hand-designed
    reference circuit.
    """
    rows, clfs = [], None
    for ds, mode, _ in pairs:
        m = load(os.path.join(a.reports, f"multiseed_{ds}.json"))
        if not m or "aae" not in m:
            continue
        ents = m["aae"]["entries"]
        clfs = clfs or [c for c in ["hea", "hea8", "hea8_probs", "qcnn"] if any(c in e["classifiers"] for e in ents)]
        if rows:
            rows.append("MID")
        for e in ents:
            rows.append([esc(ds) if e is ents[0] else "", e["seed"], e["n2q"], f"{e['kernel_auc']:.3f}"]
                        + [ci(e["classifiers"].get(c, {}).get("vs_default")) for c in clfs])
    if not rows:
        return "% Table 1 skipped: no reports/multiseed_<dataset>.json\n\n"
    return table("Retention-selected amplitude encoders against the hand-designed 31-CNOT reference circuit. "
                 "$\\Delta$ = difference in test balanced accuracy under the same trained classifier "
                 "(paired bootstrap, 95\\% CI; bold = CI excludes 0).",
                 "tab:retention_vs_default", "ll" + "r" * (2 + len(clfs)),
                 ["Dataset", "Seed", "CNOTs", "Kernel AUC"] + [rf"$\Delta$ {esc(c)}" for c in clfs], rows)


def t2(a, pairs):
    """tab:objective_<ds> and tab:objective_tests_<ds>: objective study and its paired tests."""
    out = ""
    for ds, mode, _ in pairs:
        o = load(os.path.join(a.reports, f"objective_{ds}.json"))
        if not o:
            out += f"% Table 2 ({ds}) skipped: no reports/objective_{ds}.json\n\n"
            continue
        arms = o["arms"]
        clfs = sorted({c for v in arms.values() for c in v.get("clf_bal_acc", {})})
        rows = []
        for kind, lab in (("retention", "Retention search (ours)"), ("fidelity_matched", "Fidelity search, matched CNOTs"),
                          ("fidelity_best", "Fidelity search, own selection"),
                          ("literature", "Literature ansätze (all)"),
                          ("handref", "Hand-designed reference"), ("random", "Random circuits")):
            vs = [v for v in arms.values() if v["kind"] == kind]
            if not vs:
                continue
            rows.append([lab, len(vs), ms([v["n2q"] for v in vs], 1 if len(vs) > 1 else 0),
                         ms([v["test_fidelity"] for v in vs]),
                         ms([v["kernel_auc"] for v in vs])]
                        + [ms([v.get("clf_bal_acc", {}).get(c, np.nan) for v in vs]) for c in clfs])
        sp = o.get("spearman", {})
        note = ("Mean $\\pm$ standard deviation over search seeds (or over random circuits). "
                + (f"Spearman correlation between test fidelity and kernel AUC over all circuits: "
                   f"{sp['fidelity_vs_kernel_auc']:.2f}." if "fidelity_vs_kernel_auc" in sp else ""))
        out += table(f"Objective study on {esc(ds)} ({mode}): amplitude encoders chosen by different criteria, "
                     "evaluated identically on the test split.",
                     f"tab:objective_{ds}", "lr" + "r" * (3 + len(clfs)),
                     ["Circuits", "n", "CNOTs", "Fidelity", "Kernel AUC"] + [f"{esc(c)} bal.\\ acc." for c in clfs],
                     rows, note)
        trows = []
        for key, t in o.get("tests", {}).items():
            s, other, c = key.split("|")
            if t:
                trows.append([s, esc(other.replace(f"_s{s}", "")), esc(c), ci(t)])
        if trows:
            out += table(f"Objective study on {esc(ds)}: paired differences, retention pick minus the other circuit "
                         "(test balanced accuracy, 95\\% CI; bold = CI excludes 0).",
                         f"tab:objective_tests_{ds}", "llll", ["Seed", "Compared with", "Classifier", r"$\Delta$"], trows)
    return out


def t3(a, pairs):
    """tab:fourier_vs_reference: Fourier picks (all seeds) vs. the best literature reference."""
    rows, clfs = [], ["reupload", "qcnn", "hea8"]
    for ds, mode, fds in pairs:
        m = load(os.path.join(a.reports, f"multiseed_{ds}.json"))
        if not m or "fourier" not in m:
            continue
        ents = m["fourier"]["entries"]
        if rows:
            rows.append("MID")
        for e in ents:
            rows.append([esc(fds) if e is ents[0] else "", e["seed"], e["cost"], e["n_features"],
                         ", ".join(f"{x:g}" for x in e["scales"]), f"{e['kernel_bal_acc']:.3f}"]
                        + [ci(e["classifiers"].get(c, {}).get("vs_reference")) for c in clfs])
    if not rows:
        return "% Table 3 skipped: no Fourier entries in reports/multiseed_<dataset>.json\n\n"
    return table("Fourier (angle) encoding search: selected encoding against the best literature reference "
                 "(chosen on held-out data). $\\Delta$ = difference in test balanced accuracy under the same "
                 "classifier (paired bootstrap, 95\\% CI; bold = CI excludes 0).",
                 "tab:fourier_vs_reference", "llrrlr" + "r" * len(clfs),
                 ["Dataset", "Seed", "CNOTs", "$d$", "Scales", "Kernel bal.\\ acc."]
                 + [rf"$\Delta$ {esc(c)}" for c in clfs], rows)


def t4(a, pairs):
    """tab:noise_<ds>: test balanced accuracy vs. gate-error rate for every encoder."""
    out = ""
    for ds, mode, fds in pairs:
        n = load(os.path.join(a.reports, f"noise_{ds}.json"))
        if not n:
            out += f"% Table 4 ({ds}) skipped: no reports/noise_{ds}.json\n\n"
            continue
        ps = n["p"]
        rows = []
        for enc, v in n["results"].items():
            first = next(iter(v.values()))
            clf = next(iter(first["clf"]))
            row = [esc(enc), first["n2q"]]
            for p in ps:
                r = v.get(str(p)) or v.get(f"{p}")
                row.append(f"{r['clf'][clf]['bal_acc']:.3f}" if r else "--")
            rows.append(row)
        trows = []
        for key, t in n.get("tests", {}).items():
            # keys: "<A>|<B>|<p>" (exact readout) or "<A>|<B>|<p>|S<shots>" (finite shots)
            parts = key.split("|")
            x, y, p = parts[:3]
            where = f"$p={p}$" + (f", {parts[3][1:]} shots" if len(parts) > 3 else "")
            if t:
                trows.append((x, y, where, ci(t)))
        note = "Paired differences: " + "; ".join(f"{esc(x)} $-$ {esc(y)} at {w}: {c}" for x, y, w, c in trows) + "." if trows else None
        out += table(f"Noise study on {esc(ds)}: test balanced accuracy of a trained classifier ({esc(clf)}) on "
                     "encodings with two-qubit depolarizing noise $p$ after every entangling gate "
                     "(classifier noise-free).",
                     f"tab:noise_{ds}", "lr" + "r" * len(ps), ["Encoder", "CNOTs"] + [f"$p={p:g}$" for p in ps],
                     rows, note)
    return out


def t5(a, pairs):
    """tab:classical: best classical and quantum models per dataset (reports/claims.json)."""
    c = load(os.path.join(a.reports, "claims.json"))
    if not c:
        return "% Table 5 skipped: no reports/claims.json\n\n"
    rows = []
    for ds, mode, fds in pairs:
        e = c.get(f"{ds}:{mode}") or {}
        cb = e.get("competitive_with_classical") or {}
        dq = (e.get("fourier") or {}).get("dequantisation") or {}
        if not (cb or dq):
            continue
        f = lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else "--"
        rows.append([esc(ds), f(cb.get("best_classical_bal_acc")), f(dq.get("best_classical_same_features")),
                     f(dq.get("best_quantum")), f(cb.get("best_trained_quantum_bal_acc"))])
    if not rows:
        return "% Table 5 skipped: no comparable entries in claims.json\n\n"
    return table("Best classical and quantum models (test balanced accuracy). Classical same features = best "
                 "of logistic regression, RBF SVM and a Fourier model restricted to the encoding's spectrum, on "
                 "the selected Fourier encoding's features.",
                 "tab:classical", "lrrrr",
                 ["Dataset", "Best classical", "Classical, same features", "Best quantum (any)", "Best trained quantum"],
                 rows)


def figures(a):
    """Copy the report figures next to the tables and write the LaTeX include lines."""
    fdir = os.path.join(a.out, "figures")
    os.makedirs(fdir, exist_ok=True)
    lines = []
    for f in sorted(glob.glob(os.path.join(a.reports, "fig_*.png"))):
        name = os.path.basename(f)
        shutil.copy(f, os.path.join(fdir, name))
        lines += [r"\begin{figure}[t]", r"\centering",
                  rf"\includegraphics[width=\linewidth]{{figures/{name}}}",
                  rf"\caption{{TODO: caption for {esc(name)}.}}",
                  rf"\label{{fig:{os.path.splitext(name)[0]}}}", r"\end{figure}", ""]
    with open(os.path.join(a.out, "figures.tex"), "w") as fh:
        fh.write("\n".join(lines))
    return len(lines) // 7


def main():
    """Command line: (dataset, mode, Fourier dataset) triples, report and output folders."""
    p = argparse.ArgumentParser(description="LaTeX tables for the paper")
    p.add_argument("--pairs", nargs="*", default=["bloodmnist:centered:bloodmnist", "pathmnist:centered:pathmnist"])
    p.add_argument("--reports", default="reports")
    p.add_argument("--out", default=os.path.join("reports", "paper"))
    a = p.parse_args()
    pairs = [tuple(x.split(":")) for x in a.pairs]
    os.makedirs(a.out, exist_ok=True)
    tex = ("% Generated by paper_tables.py. Requires \\usepackage{booktabs} and \\usepackage{graphicx}.\n"
           "% Bold intervals exclude zero.\n\n")
    tex += t1(a, pairs) + t2(a, pairs) + t3(a, pairs) + t4(a, pairs) + t5(a, pairs)
    with open(os.path.join(a.out, "tables.tex"), "w") as fh:
        fh.write(tex)
    n = figures(a)
    skipped = [l for l in tex.splitlines() if "skipped" in l]
    print(f"wrote {a.out}/tables.tex and {n} figures to {a.out}/figures/")
    for s in skipped:
        print("  " + s.lstrip("% "))


if __name__ == "__main__":
    main()
