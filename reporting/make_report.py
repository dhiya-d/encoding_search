#!/usr/bin/env python3
# ----------------------------------------------------------------
# make_report.py  —  turn all results into claims that are either
# supported or not, with paired bootstrap confidence intervals.
#
# Reads (all optional; sections without data are skipped):
#   experiments/<ds>/search/<mode>/selection.csv          AAE encoder search
#   experiments/<ds>/b_ceiling/...                        classical references, ceiling
#   experiments_searched/<ds>/a_phase_check/<mode>/       searched encoder, test split
#   experiments_handref/<ds>/a_phase_check/    hand-designed 31-CNOT reference, test split
#   experiments_searched/<ds>/downstream_classifiers/     trained classifiers (with predictions)
#   experiments_fourier/<ds>/downstream/                  Fourier encodings (with predictions)
#
# Usage (from the repository root):
#   python reporting/make_report.py
#   python reporting/make_report.py --runs bloodmnist:centered --bootstrap 5000
#
# Output: reports/report.md, reports/claims.json, reports/*.png
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import csv
import glob
import json
import os

import numpy as np
from scipy.stats import rankdata

import common as U

# "default" = hand-designed reference circuit (name kept for compatibility with result files)
SOURCES = ("exact", "searched", "default")


# ----------------------------------------------------------------
# loading
# ----------------------------------------------------------------

def jload(path):
    """JSON file contents, or None if the file does not exist."""
    return U.load_json(path) if os.path.exists(path) else None


def csv_rows(path):
    """Rows of a CSV file as dicts ([] if the file does not exist)."""
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    """float(x), or NaN if x is empty or not a number."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_classifier_results(root, ds, mode):
    """Only results whose classifier definition is current (e.g. old,
    unscaled *_probs runs are skipped)."""
    import classifiers as CL
    out = {}
    for p in glob.glob(os.path.join(root, ds, "downstream_classifiers", mode, "*_*.json")):
        r = U.load_json(p)
        if "source" in r and "classifier" in r:
            if r.get("spec_version", 1) != CL.spec_version(r["classifier"]):
                continue
            out[(r["source"], r["classifier"])] = r
    return out


def load_fourier(root, ds):
    """Downstream results of every Fourier encoding of a dataset, and the classical all-
    feature baseline.
    """
    out = {}
    for p in glob.glob(os.path.join(root, ds, "downstream", "*.json")):
        if os.path.basename(p) == "classical_all_features.json":
            continue
        r = U.load_json(p)
        if "genome" in r:
            out[r["name"]] = r
    return out, jload(os.path.join(root, ds, "downstream", "classical_all_features.json"))


# ----------------------------------------------------------------
# statistics
# ----------------------------------------------------------------

def bal_acc(y, pred, K):
    """Balanced accuracy (mean per-class recall) without sklearn, for the bootstrap loop."""
    tot = np.bincount(y, minlength=K)
    hit = np.bincount(y[pred == y], minlength=K)
    m = tot > 0
    return float((hit[m] / tot[m]).mean())


def macro_auc(y, P, K):
    """Binary AUC (rank formula) or macro one-vs-rest AUC, without sklearn."""
    aucs = []
    if K == 2:
        pos = y == 1
        n1, n0 = pos.sum(), (~pos).sum()
        if n1 and n0:
            r = rankdata(P[:, 1])
            aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    else:
        for c in range(K):
            pos = y == c
            n1, n0 = pos.sum(), (~pos).sum()
            if n1 and n0:
                r = rankdata(P[:, c])
                aucs.append((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(aucs)) if aucs else float("nan")


def seed_probas(res, positions=None):
    """Test class probabilities of every training seed of a classifier result."""
    Ps = [np.asarray(r["test_proba"]) for r in res["runs"] if "test_proba" in r]
    if positions is not None:
        Ps = [P[positions] for P in Ps]
    return Ps


def paired_bootstrap(y, PA, PB, K, B, seed=0, with_auc=True):
    """Difference A - B of seed-averaged metrics on the same test images."""
    rng = np.random.default_rng(seed)
    n = len(y)
    predA = [P.argmax(1) for P in PA]
    predB = [P.argmax(1) for P in PB]

    def stat(idx):
        """Seed-averaged difference A - B in balanced accuracy (and AUC) on a bootstrap
        resample of the test images.
        """
        yy = y[idx]
        d_ba = (np.mean([bal_acc(yy, p[idx], K) for p in predA])
                - np.mean([bal_acc(yy, p[idx], K) for p in predB]))
        if not with_auc:
            return d_ba, np.nan
        d_auc = (np.mean([macro_auc(yy, P[idx], K) for P in PA])
                 - np.mean([macro_auc(yy, P[idx], K) for P in PB]))
        return d_ba, d_auc

    obs = stat(np.arange(n))
    boots = np.array([stat(rng.integers(0, n, n)) for _ in range(B)])
    out = {}
    for j, name in enumerate(("bal_acc", "auc")):
        col = boots[:, j]
        if np.all(np.isnan(col)):
            continue
        lo, hi = np.nanpercentile(col, [2.5, 97.5])
        p = 2 * min(np.mean(col <= 0), np.mean(col >= 0))
        out[name] = {"diff": float(obs[j]), "ci_low": float(lo), "ci_high": float(hi),
                     "p_value": float(min(1.0, p)), "significant": bool(lo > 0 or hi < 0)}
    return out


def fmt_ci(d):
    """'diff [low, high]' with a star when the 95% interval excludes zero."""
    if not d:
        return "n/a"
    star = " *" if d["significant"] else ""
    return f"{d['diff']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]{star}"


def table(headers, rows):
    """Markdown table."""
    s = "| " + " | ".join(headers) + " |\n|" + "---|" * len(headers) + "\n"
    for r in rows:
        s += "| " + " | ".join(str(x) for x in r) + " |\n"
    return s + "\n"


# ----------------------------------------------------------------
# sections
# ----------------------------------------------------------------

def section_search(args, ds, mode, figs):
    """Report section on the search: held-out re-evaluation of the final front and the pick."""
    sel = csv_rows(os.path.join(args.search_root, ds, "search", mode, "selection.csv"))
    if not sel:
        return "", None
    n2q = np.array([fnum(r["n2q"]) for r in sel])
    ret = np.array([fnum(r["retention"]) for r in sel])
    auc = np.array([fnum(r["auc"]) for r in sel])
    fid = np.array([fnum(r["fid_mean"]) for r in sel])
    std = np.array([fnum(r["auc_std"]) for r in sel])
    order = np.argsort(n2q)
    n2q, ret, auc, fid, std = n2q[order], ret[order], auc[order], fid[order], std[order]
    best = np.nanmax(ret)
    plateau_i = int(np.argmax(ret >= best - args.plateau_tol))
    after = slice(plateau_i, None)
    fid_gain = float(np.nanmax(fid[after]) - fid[plateau_i])
    ret_spread = float(np.nanmax(ret[after]) - np.nanmin(ret[after]))
    rho = float(np.corrcoef(rankdata(fid), rankdata(ret))[0, 1]) if len(fid) > 2 else float("nan")
    claim = {"plateau_n2q": int(n2q[plateau_i]), "max_n2q": int(n2q[-1]),
             "fidelity_gain_after_plateau": fid_gain, "retention_spread_after_plateau": ret_spread,
             "spearman_fidelity_retention": rho,
             "supported": bool(fid_gain >= 0.05 and ret_spread <= 2 * args.plateau_tol and len(n2q) > plateau_i + 1)}

    plt = U.get_plt()
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(n2q, auc, yerr=std, fmt="o-", label="held-out CV AUC")
    ax.set_xlabel("two-qubit gates"); ax.set_ylabel("held-out CV AUC")
    ax2 = ax.twinx()
    ax2.plot(n2q, fid, "s--", color="g", label="mean fidelity")
    ax2.set_ylabel("mean fidelity", color="g")
    ax.axvline(n2q[plateau_i], color="grey", ls=":")
    ax.set_title(f"{ds}/{mode}: held-out AUC and fidelity along the front")
    fig.tight_layout()
    name = f"fig_search_front_{ds}_{mode}.png"
    fig.savefig(os.path.join(args.out, name), dpi=150); plt.close(fig)
    figs.append(name)

    s = f"### Encoder search front ({ds}, {mode})\n\n"
    s += table(["2q gates", "held-out AUC", "retention", "fidelity"],
               [[int(a), f"{b:.3f} ± {e:.3f}", f"{c:.3f}", f"{d:.3f}"]
                for a, b, e, c, d in zip(n2q, auc, std, ret, fid)])
    s += (f"Retention reaches its plateau (within {args.plateau_tol} of the best) at "
          f"**{claim['plateau_n2q']} gates**. Beyond that, fidelity rises by {fid_gain:+.3f} while "
          f"retention varies by only {ret_spread:.3f}. "
          f"Spearman correlation of fidelity and retention along the front: {rho:.2f}.\n\n"
          f"![]({name})\n\n")
    return s, claim


def section_encoders(args, ds, mode):
    """Report section comparing exact states, the searched circuit and the reference
    (fidelity, gate count, kernel SVM).
    """
    a = jload(os.path.join(args.searched, ds, "a_phase_check", mode, "results.json"))
    d = jload(os.path.join(args.default, ds, "a_phase_check", mode, "results.json"))
    if not a:
        return "", {}
    rows = [["exact states (ceiling)", "—", "—",
             f"{a['readouts']['target_fidkernel_svm']['bal_acc']:.3f}",
             f"{a['readouts']['target_fidkernel_svm']['auc']:.3f}"]]
    info = {"searched_n2q": a["circuit"]["n_two_qubit"]}
    for key, lab, r in (("searched", "searched", a), ("default", "hand-designed reference", d)):
        if r:
            ro = r["readouts"]["encoded_fidkernel_svm"]
            rows.append([lab, r["circuit"]["n_two_qubit"], f"{r['fidelity']['mean']:.3f}",
                         f"{ro['bal_acc']:.3f}", f"{ro['auc']:.3f}"])
            info[f"{key}_n2q"] = r["circuit"]["n_two_qubit"]
    s = f"### Encoders on the test split, sign-blind kernel SVM ({ds}, {mode})\n\n"
    s += table(["encoder", "2q gates", "fidelity", "bal. acc.", "AUC"], rows)
    return s, info


def section_classifiers(args, ds, mode, figs):
    """Report section on the trained classifiers, with paired bootstrap tests searched vs.
    reference circuit.
    """
    res = load_classifier_results(args.searched, ds, mode)
    if not res:
        return "", None, None
    clfs = sorted({c for _, c in res}, key=lambda c: (c.startswith("qcnn"), len(c), c))
    rows = []
    for c in clfs:
        row = [c]
        for s in SOURCES:
            r = res.get((s, c))
            row.append(f"{r['test_bal_acc_mean']:.3f} ± {r['test_bal_acc_std']:.3f} / "
                       f"{r['test_auc_mean']:.3f}" if r else "—")
        rows.append(row)
    s = f"### Trained classifiers on amplitude-encoded states ({ds}, {mode})\n\n"
    s += "Balanced accuracy (mean ± std over seeds) / AUC, test split.\n\n"
    s += table(["classifier", "exact", "searched", "hand-designed ref."], rows)

    tests, K = {}, None
    for c in clfs:
        ra, rb = res.get(("searched", c)), res.get(("default", c))
        if not (ra and rb) or "test_labels" not in ra or "test_labels" not in rb:
            continue
        ya, yb = np.asarray(ra["test_labels"]), np.asarray(rb["test_labels"])
        if not np.array_equal(ya, yb):
            continue
        K = int(max(ya.max(), yb.max())) + 1
        PA, PB = seed_probas(ra), seed_probas(rb)
        if PA and PB:
            tests[c] = paired_bootstrap(ya, PA, PB, K, args.bootstrap, args.seed)
    missing = [c for c in clfs if c not in tests]
    if tests:
        s += "**Searched − hand-designed reference** (paired bootstrap on the same test images, 95% CI; * = CI excludes 0):\n\n"
        s += table(["classifier", "Δ bal. acc.", "Δ AUC"],
                   [[c, fmt_ci(t.get("bal_acc")), fmt_ci(t.get("auc"))] for c, t in tests.items()])
        plt = U.get_plt()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
        for ax, m in zip(axes, ("bal_acc", "auc")):
            names = [c for c in tests if m in tests[c]]
            d = [tests[c][m]["diff"] for c in names]
            lo = [tests[c][m]["diff"] - tests[c][m]["ci_low"] for c in names]
            hi = [tests[c][m]["ci_high"] - tests[c][m]["diff"] for c in names]
            ax.errorbar(range(len(names)), d, yerr=[lo, hi], fmt="o", capsize=4)
            ax.axhline(0, color="k", lw=0.8)
            ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=20)
            ax.set_title(f"searched − hand-designed ref.: {m}")
        fig.suptitle(f"{ds}/{mode}: encoder effect under each classifier (95% CI)")
        fig.tight_layout()
        name = f"fig_searched_vs_default_{ds}_{mode}.png"
        fig.savefig(os.path.join(args.out, name), dpi=150); plt.close(fig)
        figs.append(name)
        s += f"![]({name})\n\n"
    if missing:
        s += (f"_No paired test for: {', '.join(missing)} (predictions not saved; "
              f"rerun train_on_encoded_states.py)._\n\n")
    n_sig = sum(1 for t in tests.values() if t.get("bal_acc", {}).get("significant")
                and t["bal_acc"]["diff"] > 0)
    n_neg = sum(1 for t in tests.values() if t.get("bal_acc", {}).get("significant")
                and t["bal_acc"]["diff"] < 0)
    claim = {"n_classifiers_tested": len(tests), "n_significantly_better": n_sig,
             "n_significantly_worse": n_neg,
             "supported": bool(tests and n_sig >= (len(tests) + 1) // 2 and n_neg == 0),
             "tests": tests}
    best = None
    for c in clfs:
        r = res.get(("exact", c))
        if r and (best is None or r["test_bal_acc_mean"] > best[1]):
            best = (c, r["test_bal_acc_mean"])
    return s, claim, (res, best)


def section_budget(args, ds, mode, clf_info, figs):
    """Report section decomposing the gap from the classical reference to the best quantum
    classifier into preprocessing, encoding and classifier losses.
    """
    ref = jload(os.path.join(args.diag_b, ds, "b_ceiling", "classical_reference", "results.json"))
    ceil = jload(os.path.join(args.diag_b, ds, "b_ceiling", mode, "results.json"))
    if not (ref and ceil and clf_info):
        return "", None
    res, best = clf_info
    steps = []
    if "bal_acc" in ref.get("rgb16_rbf_svm", {}):
        steps.append(("classical RBF, RGB 16×16", ref["rgb16_rbf_svm"]["bal_acc"]))
    steps.append(("classical RBF, grayscale 16×16", ref["pixels16_rbf_svm"]["bal_acc"]))
    steps.append(("kernel SVM, exact amplitude states", ceil["readouts"]["fidkernel_svm"]["bal_acc"]))
    c = best[0]
    for src, lab in (("exact", f"{c}, exact states"), ("searched", f"{c}, searched encoder"),
                     ("default", f"{c}, hand-designed 31-CNOT reference")):
        if (src, c) in res:
            steps.append((lab, res[(src, c)]["test_bal_acc_mean"]))
    rows, prev = [], None
    for lab, v in steps:
        rows.append([lab, f"{v:.3f}", "" if prev is None else f"{v - prev:+.3f}"])
        prev = v
    s = f"### Where the class information goes ({ds}, {mode})\n\n"
    s += ("Balanced accuracy at each stage, same test images; classifier = the trained model "
          f"that is best on exact states ({c}).\n\n")
    s += table(["stage", "bal. acc.", "change"], rows)

    plt = U.get_plt()
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(steps) + 1.2))
    ax.barh(range(len(steps))[::-1], [v for _, v in steps], color="#4f81bd")
    ax.set_yticks(range(len(steps))[::-1]); ax.set_yticklabels([l for l, _ in steps], fontsize=8)
    for i, (_, v) in enumerate(steps):
        ax.text(v, len(steps) - 1 - i, f" {v:.3f}", va="center", fontsize=8)
    ax.axvline(ref["pixels16_rbf_svm"]["chance"], color="r", ls=":", label="chance")
    ax.set_xlim(0, 1); ax.set_xlabel("test balanced accuracy"); ax.legend(fontsize=8)
    ax.set_title(f"{ds}/{mode}: loss budget")
    fig.tight_layout()
    name = f"fig_loss_budget_{ds}_{mode}.png"
    fig.savefig(os.path.join(args.out, name), dpi=150); plt.close(fig)
    figs.append(name)
    s += f"![]({name})\n\n"
    best_q = max(v for l, v in steps if "classical" not in l and "kernel" not in l)
    classical = steps[0][1]
    return s, {"best_trained_quantum_bal_acc": best_q, "best_classical_bal_acc": classical,
               "gap": classical - best_q, "supported": bool(best_q >= classical - 0.02)}


def fourier_test_idx(r, kind):
    """Test-split indices of a Fourier result. Older results lack the field;
    they were computed on the full test split (TEST_MAX = None)."""
    if "test_split_idx" in r:
        return [int(t) for t in r["test_split_idx"]]
    return list(range(len(r[kind]["test_labels"])))


def has_preds(r, kind):
    """True if a result stores per-seed test probabilities for a model (needed for paired
    tests).
    """
    return kind in r and r[kind].get("runs") and all("test_proba" in x for x in r[kind]["runs"]) \
        and "test_labels" in r[kind]


def pick_fourier_encoding(four):
    """Name of the main Fourier pick (search_best if present)."""
    names = [n for n in four if n.startswith("search_")]
    for n in names:
        if "best" in n:
            return n
    return names[0] if names else None


def section_fourier(args, ds, clf_info):
    """Report section on the Fourier encodings: all models, references and paired tests
    pick vs. references and vs. the amplitude encoder.
    """
    four, call = load_fourier(args.fourier, ds)
    if not four:
        return "", None
    kinds = [k for k in ("reupload", "qcnn", "hea8") if any(k in r for r in four.values())]
    rows = []
    for n, r in sorted(four.items(), key=lambda kv: (not kv[0].startswith("search"), kv[1]["cost"])):
        row = [n, r["cost"], r["genome"]["n_features"],
               f"{r['qk_fidelity']['bal_acc']:.3f}", f"{r['qk_projected']['bal_acc']:.3f}"]
        row += [f"{r[k]['test_bal_acc_mean']:.3f}" if k in r else "—" for k in kinds]
        cs, cf = r["classical_same_features"], r.get("classical_fourier_same_spectrum")
        row += [f"{cs['rbf_svm']['bal_acc']:.3f}", f"{cf['bal_acc']:.3f}" if cf else "—"]
        rows.append(row)
    s = f"### Fourier (angle) encodings, full test split ({ds})\n\n"
    if call:
        s += (f"Classical models on all features: logistic regression "
              f"{call['logreg']['bal_acc']:.3f}, RBF SVM {call['rbf_svm']['bal_acc']:.3f} "
              f"(balanced accuracy).\n\n")
    s += table(["encoding", "2q cost", "d", "kernel fid.", "kernel proj."] + kinds
               + ["classical RBF (same d)", "classical Fourier (same spectrum)"], rows)

    claim = {"dequantisation": {}, "vs_amplitude": {}, "vs_reference": {}}
    pick = pick_fourier_encoding(four)
    if pick:
        r = four[pick]
        cs, cf = r["classical_same_features"], r.get("classical_fourier_same_spectrum")
        classical_best = max(cs["rbf_svm"]["bal_acc"], cs["logreg"]["bal_acc"],
                             cf["bal_acc"] if cf else 0)
        quantum_best = max([r["qk_fidelity"]["bal_acc"], r["qk_projected"]["bal_acc"]]
                           + [r[k]["test_bal_acc_mean"] for k in kinds if k in r])
        trained = [r[k]["test_bal_acc_mean"] for k in kinds if k in r]
        claim["dequantisation"] = {"encoding": pick, "best_quantum": quantum_best,
                                   "best_trained_quantum": max(trained) if trained else None,
                                   "best_classical_same_features": classical_best,
                                   "classical_all_features": (max(call["rbf_svm"]["bal_acc"],
                                                                  call["logreg"]["bal_acc"])
                                                              if call else None),
                                   "quantum_ahead": bool(quantum_best > classical_best + 0.02)}
        s += (f"For the search pick **{pick}** (chosen on held-out data, not on test), the best "
              f"quantum model reaches {quantum_best:.3f} and the best classical model on the same "
              f"features {classical_best:.3f} (not significance-tested: classical predictions are "
              f"not saved).\n\n")

        # search pick vs best reference (reference chosen on held-out data)
        picks = jload(os.path.join(args.fourier, ds, "search", "picks.json"))
        ref_name = picks.get("best_reference") if picks else None
        if ref_name in four:
            rr = four[ref_name]
            tests = {}
            for k in kinds:
                if not (has_preds(r, k) and has_preds(rr, k)):
                    continue
                y1, y2 = np.asarray(r[k]["test_labels"]), np.asarray(rr[k]["test_labels"])
                if not np.array_equal(y1, y2):
                    continue
                K = int(y1.max()) + 1
                tests[k] = paired_bootstrap(y1, seed_probas(r[k]), seed_probas(rr[k]), K,
                                            args.bootstrap, args.seed)
            if tests:
                n_pos = sum(1 for t in tests.values() if t["bal_acc"]["significant"] and t["bal_acc"]["diff"] > 0)
                n_neg = sum(1 for t in tests.values() if t["bal_acc"]["significant"] and t["bal_acc"]["diff"] < 0)
                claim["vs_reference"] = {
                    "pick": pick, "reference": ref_name, "tests": tests,
                    "n_significantly_better": n_pos, "n_significantly_worse": n_neg,
                    "n_classifiers_tested": len(tests),
                    "supported": bool(n_pos >= (len(tests) + 1) // 2 and n_neg == 0)}
                s += (f"**Search pick ({pick}, cost {r['cost']}) − best reference ({ref_name}, "
                      f"cost {rr['cost']})**, reference chosen on held-out data; same classifier, "
                      f"same {len(y1)} test images (paired bootstrap, 95% CI):\n\n")
                s += table(["classifier", "Δ bal. acc.", "Δ AUC"],
                           [[k, fmt_ci(t.get("bal_acc")), fmt_ci(t.get("auc"))] for k, t in tests.items()])

        # Fourier pick vs searched amplitude encoder, same test images
        if clf_info and clf_info[0]:
            res = clf_info[0]
            tests, n_img = {}, 0
            for k in kinds:
                aae = res.get(("searched", k))
                if not (aae and has_preds(r, k) and "test_split_idx" in aae):
                    continue
                pos_map = {t: i for i, t in enumerate(fourier_test_idx(r, k))}
                try:
                    pos = np.array([pos_map[int(t)] for t in aae["test_split_idx"]])
                except KeyError:
                    continue
                yF = np.asarray(r[k]["test_labels"])[pos]
                yA = np.asarray(aae["test_labels"])
                if not np.array_equal(yF, yA):
                    continue
                K = int(max(yA.max(), yF.max())) + 1
                tests[k] = paired_bootstrap(yA, seed_probas(r[k], pos), seed_probas(aae), K,
                                            args.bootstrap, args.seed)
                n_img = len(yA)
            if tests:
                s += (f"**Fourier encoding ({pick}) − searched amplitude encoding**, same classifier, "
                      f"same {n_img} test images (paired bootstrap, 95% CI):\n\n")
                s += table(["classifier", "Δ bal. acc.", "Δ AUC"],
                           [[k, fmt_ci(t.get("bal_acc")), fmt_ci(t.get("auc"))] for k, t in tests.items()])
                n_pos = sum(1 for t in tests.values() if t["bal_acc"]["significant"] and t["bal_acc"]["diff"] > 0)
                claim["vs_amplitude"] = {"tests": tests, "n_significantly_better": n_pos,
                                         "n_classifiers_tested": len(tests),
                                         "supported": bool(n_pos >= (len(tests) + 1) // 2)}
    return s, claim


# ----------------------------------------------------------------

LIMITATIONS = """## Limitations to state explicitly

- Noise-free statevector simulation on 8 qubits; no hardware or noise model.
- Amplitude encoding needs a separate optimisation for every image; its cost grows with the data.
  The Fourier encodings do not have this problem.
- Each encoder search was run with a single random seed. Rerun with different `SEED` values
  before claiming that the search reliably finds these circuits.
- The classifier results average only a few training seeds. Confidence intervals account for the
  test images, not for variation between training runs.
- breastmnist has 156 test images; differences below about 0.05 AUC cannot be resolved there.
- Classical models on the same inputs remain clearly ahead. The contribution is automated encoder
  design and an analysis of where information is lost, not an advantage over classical methods.
"""


def main():
    """Command line: runs and result folders; writes reports/report.md and claims.json."""
    p = argparse.ArgumentParser(description="claims report with paired bootstrap statistics")
    p.add_argument("--runs", nargs="*", default=["bloodmnist:centered", "breastmnist:uncentered"])
    p.add_argument("--search-root", default="experiments")
    p.add_argument("--diag-b", default="experiments")
    p.add_argument("--searched", default="experiments_searched")
    p.add_argument("--default", default="experiments_handref")
    p.add_argument("--fourier", default="experiments_fourier")
    p.add_argument("--out", default="reports")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--plateau-tol", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.makedirs(args.out, exist_ok=True)

    md = "# Automated encoding design: results report\n\n"
    claims, figs, body = {}, [], ""
    for spec in args.runs:
        ds, mode = spec.split(":")
        body += f"## {ds} ({mode})\n\n"
        s1, c_search = section_search(args, ds, mode, figs)
        s2, enc = section_encoders(args, ds, mode)
        s3, c_clf, clf_info = section_classifiers(args, ds, mode, figs)
        s4, c_budget = section_budget(args, ds, mode, clf_info, figs)
        s5, c_four = section_fourier(args, ds, clf_info)
        body += s1 + s2 + s3 + s4 + s5
        dq = (c_four or {}).get("dequantisation") or {}
        if c_budget or dq:
            q = [v for v in ((c_budget or {}).get("best_trained_quantum_bal_acc"),
                             dq.get("best_trained_quantum")) if v is not None]
            k = [v for v in ((c_budget or {}).get("best_classical_bal_acc"),
                             dq.get("classical_all_features")) if v is not None]
            if q and k:
                c_budget = {**(c_budget or {}), "best_trained_quantum_bal_acc": max(q),
                            "best_classical_bal_acc": max(k), "gap": max(k) - max(q),
                            "supported": bool(max(q) >= max(k) - 0.02),
                            "note": "best trained quantum model over amplitude and Fourier encodings "
                                    "vs best classical reference; Fourier numbers use the full test split"}
        claims[spec] = {"fidelity_is_a_poor_fitness": c_search,
                        "searched_beats_default_under_trained_classifiers": c_clf,
                        "encoders": enc, "competitive_with_classical": c_budget,
                        "fourier": c_four}

    covered = {spec.split(":")[0] for spec in args.runs}
    for ds in sorted(os.listdir(args.fourier)) if os.path.isdir(args.fourier) else []:
        if ds in covered or not os.path.isdir(os.path.join(args.fourier, ds, "downstream")):
            continue
        s5, c_four = section_fourier(args, ds, None)
        if s5:
            body += f"## {ds} (Fourier encodings only)\n\n" + s5
            claims[ds] = {"fourier": c_four}

    def status(c):
        """Status label of a claim for the summary table."""
        if not c:
            return "no data"
        return "SUPPORTED" if c.get("supported") else "not supported"

    rows = []
    for spec, c in claims.items():
        if "encoders" in c:
            ce = c["encoders"] or {}
            cheaper = (ce.get("searched_n2q") is not None and ce.get("default_n2q") is not None
                       and ce["searched_n2q"] < ce["default_n2q"])
            cc = c["searched_beats_default_under_trained_classifiers"]
            rows.append([spec, "Amplitude: fidelity is a poor fitness for encoder search",
                         status(c["fidelity_is_a_poor_fitness"])])
            rows.append([spec, "Amplitude: searched encoder is cheaper than the hand-designed reference",
                         "SUPPORTED" if cheaper else ("no data" if not ce else "not supported")])
            detail = (f" ({cc['n_significantly_better']}/{cc['n_classifiers_tested']} classifiers significant)"
                      if cc else "")
            rows.append([spec, "Amplitude: searched encoder beats the hand-designed reference under trained classifiers",
                         status(cc) + detail])
        four = c.get("fourier") or {}
        vr = four.get("vs_reference")
        if vr:
            rows.append([spec, f"Fourier: search pick beats the best reference encoding ({vr['reference']})",
                         status(vr) + f" ({vr['n_significantly_better']} better, "
                         f"{vr['n_significantly_worse']} worse of {vr['n_classifiers_tested']})"])
        va = four.get("vs_amplitude")
        if va:
            rows.append([spec, "Fourier encoding beats the searched amplitude encoding (same classifier)",
                         status(va) + f" ({va['n_significantly_better']}/{va['n_classifiers_tested']})"])
        if "competitive_with_classical" in c:
            cb = c["competitive_with_classical"]
            rows.append([spec, "Best trained quantum model is within 0.02 of the best classical model",
                         status(cb) + (f" (gap {cb['gap']:.3f})" if cb else "")])
        cf = four.get("dequantisation")
        if cf:
            rows.append([spec, "Fourier: best quantum model beats classical models on the same features",
                         "SUPPORTED" if cf["quantum_ahead"] else
                         f"not supported ({cf['best_quantum']:.3f} vs {cf['best_classical_same_features']:.3f})"])
    md += "## Claims\n\n" + table(["run", "claim", "status"], rows)
    md += ("\"Supported\" is determined by the thresholds in make_report.py; "
           "read the numbers below before using a claim.\n\n")
    md += body + LIMITATIONS
    with open(os.path.join(args.out, "report.md"), "w") as f:
        f.write(md)
    U.save_json(claims, os.path.join(args.out, "claims.json"))
    print(md[:md.index("## ", md.index("## Claims") + 5)] if "## Claims" in md else md[:2000])
    print(f"report: {args.out}/report.md  ({len(figs)} figures)")


if __name__ == "__main__":
    main()
