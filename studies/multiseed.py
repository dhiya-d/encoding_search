#!/usr/bin/env python3
# ----------------------------------------------------------------
# multiseed.py  —  does the encoder search reliably find good circuits?
#
# Reruns both searches with new seeds, each in its own folders, and
# evaluates the selected circuits exactly like seed 0 (same test images,
# same training data, same classifiers), then compares them against the
# SAME baselines seed 0 was compared against:
#
#   amplitude encoder (AAE):  selected circuit vs the hand-designed 31-CNOT reference
#     experiments_seed<s>/                   search + selection
#     experiments_searched_seed<s>/          test-split encoding + classifiers
#   Fourier encoder:          search pick vs the best literature reference
#     experiments_fourier_seed<s>/           search, selection, downstream (picks only)
#
# Seed 0 is read from the existing folders (experiments/, experiments_searched/,
# experiments_fourier/) and is not recomputed. Every step resumes, so the
# script can be stopped and restarted.
#
# Usage (from the repository root):
#   python studies/multiseed.py                       # seeds 1 and 2, then the report
#   python studies/multiseed.py --seeds 1 2 3
#   python studies/multiseed.py --report-only
#   python studies/multiseed.py --skip-aae            # Fourier search only
#
# Output: reports/multiseed_<dataset>.md/.json, reports/fig_multiseed_*_<dataset>.png
# Runtime on 8 cores: roughly 30-45 min per seed for both searches together.
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import os
import subprocess
import sys
import time

import numpy as np

import common as U
import make_report as R


def sh(cmd):
    """Print and run a command; stop on failure."""
    print("\n$ " + " ".join(cmd), flush=True)
    t = time.time()
    subprocess.run(cmd, check=True)
    print(f"  ({time.time() - t:.0f}s)", flush=True)


def paths(args, s):
    """Output folders of search seed s (seed 0 = the main runs)."""
    if s == 0:
        return {"search": args.base_search, "searched": args.base_searched, "fourier": args.base_fourier}
    return {"search": f"{args.prefix}experiments_seed{s}",
            "searched": f"{args.prefix}experiments_searched_seed{s}",
            "fourier": f"{args.prefix}experiments_fourier_seed{s}"}


# ----------------------------------------------------------------
# running
# ----------------------------------------------------------------

def run_aae(args, s):
    """Amplitude search with seed s: search, selection, test-split encoding, classifiers."""
    ds, mode = args.aae_run.split(":")
    P = paths(args, s)
    py = sys.executable
    q = ["--quick"] if args.quick else []
    sh([py, repo.script("encoding_search.py"), "--dataset", ds, "--mode", mode, "--seed", str(s),
        "--out-root", P["search"]] + q)
    sh([py, repo.script("select_encoder.py"), "--dataset", ds, "--mode", mode, "--seed", str(s),
        "--out-root", P["search"]] + q)
    diag = [py, repo.script("diag_a_phase_check.py"), "--genome-root", P["search"], "--out", P["searched"],
            "--datasets", ds, "--modes", mode] + args.diag_a_args
    sh(diag)
    sh([py, repo.script("train_on_encoded_states.py"), "--runs", f"{ds}:{mode}", "--states-root", P["searched"],
        "--compare-root", "", "--sources", "searched", "--classifiers"] + args.classifiers
       + args.clf_args)


def run_fourier(args, s):
    """Fourier search with seed s, then its downstream evaluation."""
    P = paths(args, s)
    py = sys.executable
    q = ["--quick"] if args.quick else []
    sh([py, repo.script("search_encoding.py"), "--dataset", args.fourier_dataset, "--seed", str(s),
        "--out-root", P["fourier"]] + q)
    sh([py, repo.script("train_downstream.py"), "--dataset", args.fourier_dataset, "--out-root", P["fourier"],
        "--no-references"] + q)


# ----------------------------------------------------------------
# report
# ----------------------------------------------------------------

def aae_seed(args, s, default_res):
    """Summary of the amplitude pick of seed s (gate count, fidelity, classifier results
    and paired comparison with the reference circuit); None if the seed has not run yet.
    """
    ds, mode = args.aae_run.split(":")
    P = paths(args, s)
    sel = R.jload(os.path.join(P["search"], ds, "search", mode, "selected_genome.json"))
    diag = R.jload(os.path.join(P["searched"], ds, "a_phase_check", mode, "results.json"))
    if not (sel and diag):
        return None
    out = {"seed": s, "n2q": sel["gate_count"], "holdout_retention": sel.get("holdout_retention"),
           "holdout_fidelity": sel.get("holdout_fidelity"), "genome": sel["genome"],
           "kernel_bal_acc": diag["readouts"]["encoded_fidkernel_svm"]["bal_acc"],
           "kernel_auc": diag["readouts"]["encoded_fidkernel_svm"]["auc"],
           "test_fidelity": diag["fidelity"]["mean"], "classifiers": {}}
    res = R.load_classifier_results(P["searched"], ds, mode)
    for clf in args.classifiers:
        a, b = res.get(("searched", clf)), default_res.get(("default", clf))
        if not a:
            continue
        entry = {"bal_acc": a["test_bal_acc_mean"], "auc": a["test_auc_mean"]}
        if b and a.get("training") != b.get("training"):
            entry["note"] = "training settings differ from the baseline run: no paired test"
        elif (b and a.get("test_split_idx") == b.get("test_split_idx")
                and a.get("test_labels") == b.get("test_labels")):
            y = np.asarray(a["test_labels"])
            entry["vs_default"] = R.paired_bootstrap(y, R.seed_probas(a), R.seed_probas(b),
                                                     int(y.max()) + 1, args.bootstrap, args.boot_seed)
        out["classifiers"][clf] = entry
    ents = [l for l in sel["genome"] if l["kind"] == "ent"]
    out["structure"] = {"ent_layers": len(ents),
                        "step2_fraction": (sum(l.get("step") == 2 for l in ents) / len(ents)) if ents else 0.0,
                        "gates": sorted({l["gate"] for l in ents})}
    return out


def fourier_seed(args, s):
    """Summary of the Fourier picks of seed s (cost, spectrum, test results); None if the
    seed has not run yet.
    """
    ds = args.fourier_dataset
    P = paths(args, s)
    picks = R.jload(os.path.join(P["fourier"], ds, "search", "picks.json"))
    if not picks:
        return None
    four, _ = R.load_fourier(P["fourier"], ds)
    pick = R.pick_fourier_encoding(four)
    if not pick:
        return None
    r = four[pick]
    ref_name = picks["best_reference"]
    base_four, _ = R.load_fourier(args.base_fourier, ds)
    rr = base_four.get(ref_name)
    g = r["genome"]
    scales = sorted({l["scale"] for l in g["layers"] if l["kind"] in ("enc", "zz")})
    out = {"seed": s, "pick": pick, "cost": r["cost"], "n_features": g["n_features"],
           "description": r["description"], "scales": scales,
           "has_entanglers": any(l["kind"] in ("ent", "zz") for l in g["layers"]),
           "sel_auc": next((p["sel_auc"] for p in picks["picks"] if p["name"] == pick), None),
           "reference": ref_name, "reference_sel_auc": picks.get("best_reference_auc"),
           "kernel_bal_acc": r["qk_fidelity"]["bal_acc"], "kernel_auc": r["qk_fidelity"]["auc"],
           "classifiers": {}}
    for k in ("reupload", "qcnn", "hea8"):
        if not R.has_preds(r, k):
            continue
        entry = {"bal_acc": r[k]["test_bal_acc_mean"], "auc": r[k]["test_auc_mean"]}
        if rr is not None and R.has_preds(rr, k) and r[k].get("training") != rr[k].get("training"):
            entry["note"] = "training settings differ from the baseline run: no paired test"
        elif rr is not None and R.has_preds(rr, k):
            y1, y2 = np.asarray(r[k]["test_labels"]), np.asarray(rr[k]["test_labels"])
            if np.array_equal(y1, y2):
                entry["vs_reference"] = R.paired_bootstrap(
                    y1, R.seed_probas(r[k]), R.seed_probas(rr[k]), int(y1.max()) + 1,
                    args.bootstrap, args.boot_seed)
        out["classifiers"][k] = entry
    return out


def verdict(entries, key):
    """Per seed: supported if most classifiers are significantly better and none worse."""
    per_seed = {}
    for e in entries:
        tests = [c[key]["bal_acc"] for c in e["classifiers"].values() if key in c]
        pos = sum(t["significant"] and t["diff"] > 0 for t in tests)
        neg = sum(t["significant"] and t["diff"] < 0 for t in tests)
        per_seed[e["seed"]] = {"tested": len(tests), "better": pos, "worse": neg,
                               "supported": bool(tests and pos >= (len(tests) + 1) // 2 and neg == 0)}
    n_ok = sum(v["supported"] for v in per_seed.values())
    return per_seed, n_ok


def plot(args, entries, key, title, name):
    """Bar chart of one metric per classifier, one bar per seed."""
    plt = U.get_plt()
    clfs = sorted({c for e in entries for c, v in e["classifiers"].items() if key in v})
    if not clfs:
        return None
    fig, ax = plt.subplots(figsize=(max(6, 1.6 * len(clfs) + 2), 3.8))
    w = 0.8 / len(entries)
    for i, e in enumerate(entries):
        xs, d, lo, hi = [], [], [], []
        for j, c in enumerate(clfs):
            t = e["classifiers"].get(c, {}).get(key, {}).get("bal_acc")
            if t:
                xs.append(j + (i - (len(entries) - 1) / 2) * w)
                d.append(t["diff"]); lo.append(t["diff"] - t["ci_low"]); hi.append(t["ci_high"] - t["diff"])
        ax.errorbar(xs, d, yerr=[lo, hi], fmt="o", capsize=3, label=f"seed {e['seed']}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(clfs))); ax.set_xticklabels(clfs)
    ax.set_ylabel("Δ balanced accuracy (95% CI)"); ax.set_title(title); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(args.out, name), dpi=150); plt.close(fig)
    return name


def report(args):
    """Markdown/JSON report comparing all seeds of both searches (reports/multiseed*.md)."""
    os.makedirs(args.out, exist_ok=True)
    seeds = [0] + [s for s in args.seeds if s != 0]
    md = "# Search robustness across seeds\n\n"
    js = {}

    if not args.skip_aae:
        ds, mode = args.aae_run.split(":")
        default_res = R.load_classifier_results(args.base_searched, ds, mode)
        entries = [e for e in (aae_seed(args, s, default_res) for s in seeds) if e]
        if entries:
            per_seed, n_ok = verdict(entries, "vs_default")
            clfs = args.classifiers
            md += f"## Amplitude encoder search ({ds}, {mode})\n\n"
            md += ("Each seed's selected circuit, evaluated on the same test images; Δ = searched − "
                   "hand-designed 31-CNOT reference circuit under the same classifier (paired bootstrap, 95% CI, "
                   "* = excludes 0).\n\n")
            head = ["seed", "2q gates", "held-out retention", "test fidelity", "kernel bal. acc. / AUC"]
            head += [f"{c}: bal. acc. (Δ vs hand-designed ref.)" for c in clfs]
            rows = []
            for e in entries:
                row = [e["seed"], e["n2q"], f"{e['holdout_retention']:.3f}",
                       f"{e['test_fidelity']:.3f}", f"{e['kernel_bal_acc']:.3f} / {e['kernel_auc']:.3f}"]
                for c in clfs:
                    v = e["classifiers"].get(c)
                    row.append("—" if not v else f"{v['bal_acc']:.3f} (" + (
                        "settings differ" if "note" in v else R.fmt_ci(v.get('vs_default', {}).get('bal_acc'))) + ")")
                rows.append(row)
            md += R.table(head, rows)
            ms = lambda xs: f"{np.mean(xs):.3f} ± {np.std(xs):.3f}"
            md += (f"Across {len(entries)} seeds: 2q gates {np.mean([e['n2q'] for e in entries]):.1f} "
                   f"± {np.std([e['n2q'] for e in entries]):.1f}; kernel balanced accuracy "
                   f"{ms([e['kernel_bal_acc'] for e in entries])}.\n\n")
            md += R.table(["seed", "entangler layers", "share with step 2", "entanglers"],
                          [[e["seed"], e["structure"]["ent_layers"], f"{e['structure']['step2_fraction']:.2f}",
                            ", ".join(e["structure"]["gates"])] for e in entries])
            md += (f"**Searched circuit beats the hand-designed reference under most classifiers (none worse): "
                   f"{n_ok}/{len(entries)} seeds.**\n\n")
            f = plot(args, entries, "vs_default", f"{ds}: searched − hand-designed reference, per seed", f"fig_multiseed_aae_{ds}.png")
            if f:
                md += f"![]({f})\n\n"
            js["aae"] = {"entries": entries, "per_seed": per_seed, "seeds_supported": n_ok}

    if not args.skip_fourier:
        ds = args.fourier_dataset
        entries = [e for e in (fourier_seed(args, s) for s in seeds) if e]
        if entries:
            per_seed, n_ok = verdict(entries, "vs_reference")
            clfs = ["reupload", "qcnn", "hea8"]
            md += f"## Fourier encoder search ({ds})\n\n"
            md += ("Each seed's search pick on the full test split; Δ = pick − the best literature "
                   "reference (chosen on that seed's held-out batch), same classifier and training data "
                   "(paired bootstrap, 95% CI, * = excludes 0).\n\n")
            head = ["seed", "pick", "2q cost", "d", "scales", "held-out AUC (ref)", "kernel bal. acc."]
            head += [f"{c}: bal. acc. (Δ vs ref)" for c in clfs]
            rows = []
            for e in entries:
                row = [e["seed"], e["pick"], e["cost"], e["n_features"], ", ".join(f"{x:g}" for x in e["scales"]),
                       f"{e['sel_auc']:.3f} ({e['reference_sel_auc']:.3f} {e['reference']})",
                       f"{e['kernel_bal_acc']:.3f}"]
                for c in clfs:
                    v = e["classifiers"].get(c)
                    row.append("—" if not v else f"{v['bal_acc']:.3f} (" + (
                        "settings differ" if "note" in v else R.fmt_ci(v.get('vs_reference', {}).get('bal_acc'))) + ")")
                rows.append(row)
            md += R.table(head, rows)
            md += "Pick circuits:\n\n" + "".join(f"- seed {e['seed']}: `{e['description']}`\n" for e in entries) + "\n"
            md += (f"**Search pick beats the best reference under most classifiers (none worse): "
                   f"{n_ok}/{len(entries)} seeds.**\n\n")
            f = plot(args, entries, "vs_reference", f"{ds}: Fourier pick − best reference, per seed",
                     f"fig_multiseed_fourier_{ds}.png")
            if f:
                md += f"![]({f})\n\n"
            js["fourier"] = {"entries": entries, "per_seed": per_seed, "seeds_supported": n_ok}

    md += ("## How to read this\n\n"
           "- A search is reliable if every seed supports the claim, not just seed 0.\n"
           "- If seeds pick different circuits that perform alike, report the common structure "
           "(gate count, entangler pattern, frequency scales) rather than one specific circuit.\n"
           "- Confidence intervals reflect the test images only; seed-to-seed spread is the extra "
           "variation a reader should expect when rerunning the method.\n")
    tag = args.aae_run.split(":")[0] if not args.skip_aae else args.fourier_dataset
    with open(os.path.join(args.out, f"multiseed_{tag}.md"), "w") as fh:
        fh.write(md)
    U.save_json(js, os.path.join(args.out, f"multiseed_{tag}.json"))
    print(md)
    print(f"report: {args.out}/multiseed_{tag}.md")


def main():
    """Command line: seeds, datasets, classifiers and folders; runs missing seeds, then reports."""
    p = argparse.ArgumentParser(description="rerun both searches with more seeds and compare")
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2])
    p.add_argument("--aae-run", default="bloodmnist:centered")
    p.add_argument("--fourier-dataset", default="bloodmnist")
    p.add_argument("--classifiers", nargs="*", default=["hea", "hea8", "hea8_probs", "qcnn"])
    p.add_argument("--base-search", default="experiments")
    p.add_argument("--base-searched", default="experiments_searched")
    p.add_argument("--base-fourier", default="experiments_fourier")
    p.add_argument("--prefix", default="", help="prefix for the per-seed folders")
    p.add_argument("--skip-aae", action="store_true")
    p.add_argument("--skip-fourier", action="store_true")
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--out", default="reports")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="tiny settings (pipeline test only)")
    args = p.parse_args()
    args.diag_a_args, args.clf_args = [], []
    if args.quick:
        args.diag_a_args = ["--n-train", "300", "--n-test", "150", "--steps", "60", "--restarts", "1"]
        args.clf_args = ["--epochs", "4"]

    if not args.report_only:
        for s in args.seeds:
            if s == 0:
                continue
            print(f"\n{'#' * 70}\n# seed {s}\n{'#' * 70}")
            if not args.skip_aae:
                run_aae(args, s)
            if not args.skip_fourier:
                run_fourier(args, s)
    report(args)


if __name__ == "__main__":
    main()
