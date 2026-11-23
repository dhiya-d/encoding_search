#!/usr/bin/env python3
# ----------------------------------------------------------------
# objective_study.py  —  Experiment 1: is the OBJECTIVE what makes the
# difference?
#
# Compares amplitude-encoding circuits chosen in different ways, all
# evaluated identically (same test images, same classifiers, same data):
#   retention_s<k>        retention-driven search, seed k      (existing runs)
#   fidelity_best_s<k>    fidelity-driven search, seed k, its own selection
#   fidelity_matched_s<k> fidelity-driven search, seed k, best-fidelity circuit
#                         with no more CNOTs than retention_s<k>   (matched cost)
#   random_<j>            random genomes from the same search space, with CNOT
#                         counts in the range of the retention picks
#   lit_<family>_<depth>  standard ansaetze from the literature at matched CNOT
#                         counts (literature_ansatze.py): hardware-efficient
#                         RY/CNOT chain (Nakaji et al. 2022), Qiskit
#                         RealAmplitudes, alternating layered ansatz (Cerezo et al.
#                         2021). The best of them is chosen on the SAME held-out
#                         batch the searches use ("lit_best"), never on test data.
#   handref               the hand-designed 31-CNOT reference circuit
#                         ("default" in earlier outputs)                (existing)
# The two searches differ ONLY in the objective (same code, batch, budget,
# seed). Paired bootstrap tests compare retention_s<k> with the other arms
# under each trained classifier; the random arm gives a reference
# distribution; a scatter of test fidelity against class information over
# all circuits shows whether fidelity predicts downstream quality.
#
# Usage (from the repository root):
#   python studies/objective_study.py                          # everything, bloodmnist centered
#   python studies/objective_study.py --run pathmnist:centered
#   python studies/objective_study.py --stage report           # rebuild the report only
#   python studies/objective_study.py --jobs 4                 # evaluate 4 circuits at once
# Stages: search -> arms -> evaluate -> report (all by default); every step resumes.
#
# Output: experiments_fidsearch_seed<k>/     fidelity-driven searches
#         experiments_objective/<dataset>/   arms.json, one folder per new circuit
#         reports/objective_<dataset>.md/.json, reports/fig_objective_*_<dataset>.png
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.stats import spearmanr

import common as U
import make_report as R


# ----------------------------------------------------------------
# helpers
# ----------------------------------------------------------------

def sh(cmd, env=None, log=None):
    """Run a command (optionally logging its output to a file); stop on failure."""
    t = time.time()
    msg = "$ " + " ".join(cmd)
    if log is None:
        print("\n" + msg, flush=True)
        subprocess.run(cmd, check=True, env=env)
    else:
        with open(log, "a") as fh:
            fh.write(msg + "\n")
            fh.flush()
            subprocess.run(cmd, check=True, env=env, stdout=fh, stderr=subprocess.STDOUT)
    return time.time() - t


def thread_env(n):
    """Environment that limits numerical libraries to n threads."""
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[k] = str(n)
    return env


class Paths:
    """Folder layout of the study (retention runs, fidelity runs, study outputs)."""
    def __init__(self, a):
        """Dataset and mode from --run."""
        self.a = a
        self.ds, self.mode = a.run.split(":")

    def retention_root(self, s):
        """Search folder of the retention search with seed s."""
        return self.a.retention_root if s == 0 else f"experiments_seed{s}"

    def searched_root(self, s):
        """Test-split encodings of the retention pick with seed s."""
        return self.a.searched_root if s == 0 else f"experiments_searched_seed{s}"

    def fid_root(self, s):
        """Search folder of the fidelity-driven search with seed s."""
        return f"experiments_fidsearch_seed{s}"

    def selected(self, root):
        """selected_genome.json inside a search folder."""
        return os.path.join(root, self.ds, "search", self.mode, "selected_genome.json")

    def selection(self, root):
        """selection.json (held-out re-evaluation) inside a search folder."""
        return os.path.join(root, self.ds, "search", self.mode, "selection.json")

    def study(self, *parts):
        """Path inside the study's own output folder."""
        return os.path.join(self.a.out_root, self.ds, *parts)


# ----------------------------------------------------------------
# stages
# ----------------------------------------------------------------

def stage_search(a, P):
    """Run the fidelity-driven search (and selection) for every seed."""
    py = sys.executable
    q = ["--quick"] if a.quick else []
    for s in a.seeds:
        root = P.fid_root(s)
        sh([py, repo.script("encoding_search.py"), "--dataset", P.ds, "--mode", P.mode, "--seed", str(s),
            "--objective", "fidelity", "--out-root", root] + q)
        sh([py, repo.script("select_encoder.py"), "--dataset", P.ds, "--mode", P.mode, "--seed", str(s),
            "--objective", "fidelity", "--out-root", root] + q)


def stage_arms(a, P):
    """Define the compared circuits ('arms'): retention picks, fidelity picks, the
    reference, the literature ansaetze (best one chosen on held-out data within the
    budget) and random circuits in the CNOT range of the retention picks.
    """
    import search_config as SCF
    import search_core as SC
    arms = {}
    ret_n2q = []
    for s in a.seeds:
        sel = R.jload(P.selected(P.retention_root(s)))
        if not sel:
            raise SystemExit(f"missing retention search for seed {s}: {P.selected(P.retention_root(s))} "
                             f"(run encoding_search.py / multiseed.py first)")
        ret_n2q.append(sel["gate_count"])
        arms[f"retention_s{s}"] = {"kind": "retention", "seed": s, "genome": sel["genome"],
                                   "n2q": sel["gate_count"], "holdout_retention": sel.get("holdout_retention"),
                                   "eval_root": P.searched_root(s),
                                   "clf_root": P.searched_root(s), "clf_source": "searched"}
        fsel = R.jload(P.selected(P.fid_root(s)))
        fsn = R.jload(P.selection(P.fid_root(s)))
        if not (fsel and fsn):
            raise SystemExit(f"missing fidelity search for seed {s} (run --stage search)")
        arms[f"fidelity_best_s{s}"] = {"kind": "fidelity_best", "seed": s, "genome": fsel["genome"],
                                       "n2q": fsel["gate_count"]}
        rows = fsn["rows"]
        cands = [r for r in rows if r["n2q"] <= sel["gate_count"]]
        if cands:
            m = max(cands, key=lambda r: (r["fid_mean"], -r["n2q"]))
            note = "best held-out fidelity with at most as many CNOTs as the retention pick"
        else:
            m = min(rows, key=lambda r: (abs(r["n2q"] - sel["gate_count"]), -r["fid_mean"]))
            note = "no fidelity-front circuit this cheap; closest CNOT count used"
        arms[f"fidelity_matched_s{s}"] = {"kind": "fidelity_matched", "seed": s, "genome": m["genome"],
                                          "n2q": m["n2q"], "note": note}
    d = R.jload(os.path.join(a.default_root, P.ds, "a_phase_check", P.mode, "results.json"))
    if d:
        arms["handref"] = {"kind": "handref", "seed": None, "genome": d["genome"],
                           "n2q": d["circuit"]["n_two_qubit"], "eval_root": a.default_root,
                           "clf_root": P.searched_root(0), "clf_source": "default",
                           "note": "hand-designed 31-CNOT reference (called 'default' in older outputs)"}
    import literature_ansatze as LA
    for name, spec in LA.literature_set().items():
        arms[name] = {"kind": "literature", "seed": None, "genome": spec["genome"],
                      "family": spec["family"], "depth": spec["depth"], "label": spec["label"],
                      "n2q": SC.n_two_qubit(spec["genome"])}

    lo, hi = min(ret_n2q) - a.random_margin, max(ret_n2q) + a.random_margin
    rng = np.random.default_rng(a.random_seed)
    seen = {SC.key(v["genome"]) for v in arms.values()}
    k, tries = 0, 0
    while k < a.n_random and tries < 200000:
        tries += 1
        g = SC.random_genome(rng)
        for _ in range(int(rng.integers(0, 4))):
            g = SC.mutate(g, rng)
        if not SC.valid(g):
            continue
        n2q = SC.n_two_qubit(g)
        if not lo <= n2q <= hi or SC.key(g) in seen:
            continue
        seen.add(SC.key(g))
        arms[f"random_{k:02d}"] = {"kind": "random", "seed": None, "genome": g, "n2q": n2q}
        k += 1
    if k < a.n_random:
        print(f"warning: only {k} random genomes found in the CNOT range [{lo}, {hi}]")

    for name, arm in arms.items():
        if "eval_root" not in arm:
            same = [n for n, o in arms.items() if n < name and "eval_root" in o
                    and SC.key(o["genome"]) == SC.key(arm["genome"])
                    and o["kind"] not in ("retention", "handref")]
            gid = hashlib.md5(SC.key(arm["genome"]).encode()).hexdigest()[:8]
            root = arms[same[0]]["eval_root"] if same else P.study(f"{name}_{gid}")
            arm.update({"eval_root": root, "clf_root": root, "clf_source": "searched"})
    os.makedirs(P.study(), exist_ok=True)
    lit_sel = literature_selection(a, P, {n: v for n, v in arms.items() if v["kind"] == "literature"})
    for n, r in lit_sel["scores"].items():
        arms[n]["holdout_retention"] = r["retention"]
        arms[n]["holdout_auc"] = r["auc"]
    # best literature ansatz WITHIN the CNOT budget of the searched circuits
    # (held-out retention, ties: fewer CNOTs); deeper ones are reported
    # separately as high-budget references
    budget = hi
    within = [n for n, r in lit_sel["scores"].items() if r["n2q"] <= budget]
    best_within = (max(within, key=lambda n: (lit_sel["scores"][n]["retention"], -lit_sel["scores"][n]["n2q"]))
                   if within else None)
    high_budget = [n for n, r in lit_sel["scores"].items() if r["n2q"] > budget]
    print(f"best literature ansatz within {budget} CNOTs (held-out): {best_within}; "
          f"high-budget references: {high_budget}")
    U.save_json({"dataset": P.ds, "mode": P.mode, "random_cnot_range": [lo, hi],
                 "cnot_budget": budget, "literature_best": best_within,
                 "literature_best_any_budget": lit_sel["best"], "literature_high_budget": high_budget,
                 "search_config_objective_default": SCF.OBJECTIVE, "arms": arms},
                P.study("arms.json"))
    print(f"{len(arms)} arms: " + ", ".join(f"{n} ({v['n2q']})" for n, v in arms.items()))
    return arms


def literature_selection(a, P, lit_arms):
    """Score the literature ansaetze on the held-out selection batch of the
    seed-0 search (same batch, folds, ceiling and fitting budget as
    select_encoder.py) and pick the best by held-out retention."""
    import diag_common as DC
    import search_config as SCF
    import search_core as SC
    path = P.study("literature_selection.json")
    keys = sorted(lit_arms)
    cached = R.jload(path)
    if cached and sorted(cached["scores"]) == keys and cached.get("quick") == a.quick:
        return cached
    if a.quick:
        from encoding_search import apply_quick
        apply_quick()
    SCF.SEED = 0
    print(f"\nscoring {len(keys)} literature ansaetze on the held-out selection batch...")
    _, select = SC.prepare_batches(P.ds, P.mode, seed=0)
    folds = DC.make_folds(select["y"], SCF.CV_FOLDS, SCF.SEED + 1)
    ceil = SC.ceiling(select["T"], select["y"], folds)
    ev = SC.Evaluator(select["T"], select["y"], folds, ceil["C"], ceil["auc"],
                      SCF.SELECT_FIT_STEPS, SCF.SELECT_FIT_RESTARTS, SCF.SEED + 29)
    try:
        res, _ = ev.evaluate([lit_arms[k]["genome"] for k in keys])
    finally:
        ev.close()
    scores = {k: {"retention": r["retention"], "auc": r["auc"], "auc_std": r["auc_std"],
                  "fid_mean": r["fid_mean"], "n2q": r["n2q"]} for k, r in zip(keys, res)}
    best = max(keys, key=lambda k: (scores[k]["retention"], -scores[k]["n2q"]))
    out = {"scores": scores, "best": best, "ceiling_auc": ceil["auc"], "quick": a.quick,
           "rule": "highest held-out retention on the seed-0 selection batch (ties: fewer CNOTs)"}
    U.save_json(out, path)
    for k in keys:
        print(f"  {k:16s} {scores[k]['n2q']:2d} CNOTs  held-out retention {scores[k]['retention']:.3f}"
              + ("   <- best" if k == best else ""))
    return out


def _evaluate_arm(a, P, name, arm, n_threads):
    """Encode the test split with one arm and train the classifiers on it (logged)."""
    py = sys.executable
    env = thread_env(n_threads)
    log = P.study(f"{name}.log")
    os.makedirs(P.study(), exist_ok=True)
    diag = os.path.join(arm["eval_root"], P.ds, "a_phase_check", P.mode, "results.json")
    if not os.path.exists(diag):
        os.makedirs(arm["eval_root"], exist_ok=True)
        gpath = os.path.join(arm["eval_root"], "genome.json")
        U.save_json(arm["genome"], gpath)
        sh([py, repo.script("diag_a_phase_check.py"), "--genome-json", gpath, "--out", arm["eval_root"],
            "--datasets", P.ds, "--modes", P.mode] + a.diag_args, env, log)
    cmd = [py, repo.script("train_on_encoded_states.py"), "--runs", f"{P.ds}:{P.mode}",
           "--states-root", arm["clf_root"], "--classifiers"] + a.classifiers + a.clf_args
    if arm["clf_source"] == "default":
        cmd += ["--compare-root", arm["eval_root"], "--sources", "default"]
    else:
        cmd += ["--compare-root", "", "--sources", "searched"]
    sh(cmd, env, log)
    return name


def stage_evaluate(a, P, arms):
    """Evaluate all arms that are not done yet, --jobs at a time."""
    todo, seen_roots = [], set()
    for name, arm in arms.items():
        key = (arm["eval_root"], arm["clf_root"], arm["clf_source"])
        if key in seen_roots:
            continue
        seen_roots.add(key)
        todo.append((name, arm))
    n_threads = max(1, (os.cpu_count() or 1) // a.jobs)
    print(f"\nevaluating {len(todo)} circuits, {a.jobs} at a time ({n_threads} threads each); "
          f"logs in {P.study()}/<arm>.log")
    t0 = time.time()
    with ThreadPoolExecutor(a.jobs) as pool:
        futs = [pool.submit(_evaluate_arm, a, P, n, arm, n_threads) for n, arm in todo]
        for f in futs:
            print(f"  done: {f.result()}  ({time.time() - t0:.0f}s)", flush=True)


# ----------------------------------------------------------------
# report
# ----------------------------------------------------------------

def collect(a, P, arms):
    """Load fidelity, kernel and classifier results of every evaluated arm."""
    out = {}
    for name, arm in arms.items():
        d = R.jload(os.path.join(arm["eval_root"], P.ds, "a_phase_check", P.mode, "results.json"))
        if not d:
            continue
        res = R.load_classifier_results(arm["clf_root"], P.ds, P.mode)
        clfs = {c: res.get((arm["clf_source"], c)) for c in a.classifiers}
        out[name] = {**arm, "test_fidelity": d["fidelity"]["mean"],
                     "kernel_auc": d["readouts"]["encoded_fidkernel_svm"]["auc"],
                     "kernel_bal_acc": d["readouts"]["encoded_fidkernel_svm"]["bal_acc"],
                     "ceiling_auc": d["readouts"]["target_fidkernel_svm"]["auc"],
                     "clf": {c: r for c, r in clfs.items() if r}}
    return out


KIND_ORDER = ["retention", "fidelity_matched", "fidelity_best", "literature", "handref", "random"]
KIND_LABEL = {"retention": "retention search (ours)", "fidelity_matched": "fidelity search, matched CNOTs",
              "fidelity_best": "fidelity search, own selection", "literature": "literature ansatz",
              "handref": "hand-designed 31-CNOT reference", "random": "random circuit"}


def lit_best_name(a, P, arms):
    """Name of the literature ansatz selected on held-out data (saved by stage_arms)."""
    saved = R.jload(P.study("arms.json"))
    return (saved or {}).get("literature_best")


def lit_high_budget(a, P):
    """Literature ansaetze evaluated at a higher CNOT budget (saved by stage_arms)."""
    saved = R.jload(P.study("arms.json"))
    return (saved or {}).get("literature_high_budget", [])


def lit_matched(E, n2q):
    """Per literature family, the member with CNOT count closest to n2q (ties: fewer CNOTs)."""
    fams = {}
    for n, e in E.items():
        if e["kind"] == "literature":
            fams.setdefault(e["family"], []).append(n)
    return {f: min(ns, key=lambda n: (abs(E[n]["n2q"] - n2q), E[n]["n2q"])) for f, ns in fams.items()}


def paired(a, x, y, clf):
    """Paired bootstrap of two arms for one classifier (None if not comparable)."""
    ra, rb = x["clf"].get(clf), y["clf"].get(clf)
    if not (ra and rb) or ra.get("training") != rb.get("training"):
        return None
    if ra.get("test_split_idx") != rb.get("test_split_idx") or ra["test_labels"] != rb["test_labels"]:
        return None
    yv = np.asarray(ra["test_labels"])
    return R.paired_bootstrap(yv, R.seed_probas(ra), R.seed_probas(rb), int(yv.max()) + 1,
                              a.bootstrap, 0)


def stage_report(a, P, arms):
    """Markdown/JSON report and figures: fidelity vs. kernel AUC across arms, retention vs.
    fidelity picks, literature and random baselines (Table IV, Fig. 4).
    """
    E = collect(a, P, arms)
    if not E:
        raise SystemExit("no evaluated arms found")
    os.makedirs(a.reports, exist_ok=True)
    plt = U.get_plt()
    ds = P.ds
    md = f"# Objective study: retention vs fidelity, literature and random circuits ({ds}, {P.mode})\n\n"
    md += ("All circuits are amplitude encoders from the same search space, evaluated on the same test "
           "images with the same classifiers. The retention and fidelity searches differ only in the "
           "objective.\n\n")
    ba = lambda e, c: e["clf"][c]["test_bal_acc_mean"] if c in e["clf"] else float("nan")
    head = ["circuit", "CNOTs", "test fidelity", "kernel AUC"] + [f"{c} bal. acc." for c in a.classifiers]
    order = sorted(E, key=lambda n: (KIND_ORDER.index(E[n]["kind"]), E[n]["n2q"], n))
    head = ["circuit", "type", "CNOTs", "held-out retention", "test fidelity", "kernel AUC"] + \
        [f"{c} bal. acc." for c in a.classifiers]
    hr = lambda e: f"{e['holdout_retention']:.3f}" if e.get("holdout_retention") is not None else "—"
    rows = [[n, KIND_LABEL[E[n]["kind"]], E[n]["n2q"], hr(E[n]), f"{E[n]['test_fidelity']:.3f}",
             f"{E[n]['kernel_auc']:.3f}"] + [f"{ba(E[n], c):.3f}" for c in a.classifiers]
            for n in order if E[n]["kind"] != "random"]
    md += "## Circuits compared\n\n" + R.table(head, rows)
    best_lit = lit_best_name(a, P, arms)
    high_budget = lit_high_budget(a, P)
    saved = R.jload(P.study("arms.json")) or {}
    if best_lit in E:
        md += (f"Best literature ansatz within the CNOT budget of the searched circuits "
               f"(≤ {saved.get('cnot_budget')}), chosen on the held-out batch, never on test: "
               f"**{best_lit}** ({E[best_lit]['n2q']} CNOTs). "
               + (f"Deeper literature circuits ({', '.join(high_budget)}) are reported as high-budget "
                  f"references." if high_budget else "") + "\n\n")
    ceil = np.nanmean([e["ceiling_auc"] for e in E.values()])
    md += f"Kernel AUC of the exact states (ceiling): {ceil:.3f}.\n\n"

    js = {"dataset": ds, "mode": P.mode, "arms": {n: {k: v for k, v in e.items() if k != "clf"}
                                                  | {"clf_bal_acc": {c: ba(e, c) for c in e["clf"]}}
                                                  for n, e in E.items()},
          "tests": {}, "random": {}, "claims": {}}

    # paired tests
    md += "## Paired comparisons (retention pick − other circuit)\n\n"
    md += "Δ balanced accuracy, paired bootstrap on the same test images, 95% CI, * = excludes 0.\n\n"
    trows, sig_counts = [], {}
    for s in a.seeds:
        x = E.get(f"retention_s{s}")
        if not x:
            continue
        others = [(f"fidelity_matched_s{s}", "fidelity_matched", "fidelity search, matched CNOTs"),
                  (f"fidelity_best_s{s}", "fidelity_best", "fidelity search, own selection")]
        if best_lit in E:
            others.append((best_lit, "literature_best", f"best literature ansatz within budget ({best_lit})"))
        for fam, n in lit_matched(E, x["n2q"]).items():
            others.append((n, "literature_matched", f"matched-cost literature ({n})"))
        for n in high_budget:
            if n in E:
                others.append((n, "literature_high_budget", f"high-budget literature ({n})"))
        if "handref" in E:
            others.append(("handref", "handref", "hand-designed 31-CNOT reference"))
        for other, group, label in others:
            y = E.get(other)
            if not y:
                continue
            row = [f"seed {s}: retention ({x['n2q']}) − {label} ({y['n2q']})"]
            for c in a.classifiers:
                t = paired(a, x, y, c)
                js["tests"][f"{s}|{other}|{c}"] = t
                row.append(R.fmt_ci(t["bal_acc"]) if t else "n/a")
                cnt = sig_counts.setdefault(group, [0, 0, 0])
                if t:
                    cnt[0] += 1
                    cnt[1] += int(t["bal_acc"]["significant"] and t["bal_acc"]["diff"] > 0)
                    cnt[2] += int(t["bal_acc"]["significant"] and t["bal_acc"]["diff"] < 0)
            trows.append(row)
    md += R.table(["comparison"] + a.classifiers, trows)

    # random reference distribution
    rnd = [e for e in E.values() if e["kind"] == "random"]
    if rnd:
        md += f"## Random circuits ({len(rnd)} circuits, CNOTs {min(e['n2q'] for e in rnd)}–{max(e['n2q'] for e in rnd)})\n\n"
        rrows = []
        metrics = [("kernel AUC", lambda e: e["kernel_auc"])] + [(f"{c} bal. acc.", (lambda c: lambda e: ba(e, c))(c))
                                                                   for c in a.classifiers]
        for lab, fn in metrics:
            vals = np.array([fn(e) for e in rnd], float)
            vals = vals[np.isfinite(vals)]
            if not len(vals):
                continue
            picks = [fn(E[f"retention_s{s}"]) for s in a.seeds if f"retention_s{s}" in E]
            pct = [float(np.mean(vals < p)) for p in picks]
            js["random"][lab] = {"mean": float(vals.mean()), "std": float(vals.std()),
                                 "min": float(vals.min()), "max": float(vals.max()),
                                 "retention_picks": picks, "share_of_random_beaten": pct}
            rrows.append([lab, f"{vals.mean():.3f} ± {vals.std():.3f}", f"{vals.min():.3f} – {vals.max():.3f}",
                          ", ".join(f"{p:.3f}" for p in picks), ", ".join(f"{q:.0%}" for q in pct)])
        md += R.table(["metric", "random mean ± std", "random range", "retention picks (seeds)",
                       "share of random circuits beaten"], rrows)

    # fidelity vs downstream across all circuits
    allE = list(E.values())
    fid = np.array([e["test_fidelity"] for e in allE])
    kauc = np.array([e["kernel_auc"] for e in allE])
    corr = {}
    if len(allE) >= 5:
        corr["fidelity_vs_kernel_auc"] = float(spearmanr(fid, kauc).correlation)
        for c in a.classifiers:
            v = np.array([ba(e, c) for e in allE])
            m = np.isfinite(v)
            if m.sum() >= 5:
                corr[f"fidelity_vs_{c}"] = float(spearmanr(fid[m], v[m]).correlation)
        if len(rnd) >= 5:
            corr["random_only_fidelity_vs_kernel_auc"] = float(
                spearmanr([e["test_fidelity"] for e in rnd], [e["kernel_auc"] for e in rnd]).correlation)
    js["spearman"] = corr
    if corr:
        md += "## Does fidelity predict downstream quality?\n\n"
        md += R.table(["Spearman correlation", "value"], [[k.replace("_", " "), f"{v:.2f}"] for k, v in corr.items()])

    colors = {"retention": "#c0504d", "fidelity_matched": "#3b6ea5", "fidelity_best": "#7fa6d1",
              "literature": "#4a9a5b", "handref": "#222222", "random": "#aaaaaa"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    c0 = a.classifiers[0]
    for kind in ("random", "handref", "literature", "fidelity_best", "fidelity_matched", "retention"):
        es = [e for e in allE if e["kind"] == kind]
        if not es:
            continue
        axes[0].scatter([e["test_fidelity"] for e in es], [e["kernel_auc"] for e in es], s=60 if kind != "random" else 30,
                        color=colors[kind], label=KIND_LABEL[kind], edgecolor="k", lw=0.4, zorder=3)
        axes[1].scatter([e["n2q"] for e in es], [ba(e, c0) for e in es], s=60 if kind != "random" else 30,
                        color=colors[kind], label=KIND_LABEL[kind], edgecolor="k", lw=0.4, zorder=3)
    axes[0].axhline(ceil, color="k", ls=":", lw=1, label="exact-state ceiling")
    axes[0].set_xlabel("test fidelity"); axes[0].set_ylabel("kernel SVM AUC (test)")
    axes[0].set_title("fidelity vs class information, all circuits")
    axes[1].set_xlabel("CNOT gates"); axes[1].set_ylabel(f"{c0} balanced accuracy (test)")
    axes[1].set_title(f"downstream accuracy vs cost ({c0})")
    axes[0].legend(fontsize=7)
    fig.suptitle(f"{ds}: objective study"); fig.tight_layout()
    f1 = f"fig_objective_scatter_{ds}.png"
    fig.savefig(os.path.join(a.reports, f1), dpi=150); plt.close(fig)
    md += f"\n![]({f1})\n\n"

    md += "## Claims\n\n"
    claims = []
    for kind, lab in (("fidelity_matched", "Retention pick beats the fidelity pick at matched CNOT count"),
                      ("fidelity_best", "Retention pick beats the fidelity search's own selection"),
                      ("literature_best", "Retention pick beats the best literature ansatz within budget (held-out selection)"),
                      ("literature_matched", "Retention pick beats matched-cost literature ansaetze"),
                      ("literature_high_budget", "Retention pick beats deeper (high-budget) literature ansaetze"),
                      ("handref", "Retention pick beats the hand-designed 31-CNOT reference")):
        if kind in sig_counts:
            n, pos, neg = sig_counts[kind]
            ok = n > 0 and pos >= (n + 1) // 2 and neg == 0
            js["claims"][kind] = {"tests": n, "better": pos, "worse": neg, "supported": ok}
            claims.append([lab, f"{'SUPPORTED' if ok else 'not supported'} ({pos} better, {neg} worse of {n} seed×classifier tests)"])
    if "kernel AUC" in js["random"]:
        pct = js["random"]["kernel AUC"]["share_of_random_beaten"]
        ok = bool(pct) and min(pct) >= 0.9
        js["claims"]["random"] = {"share_beaten": pct, "supported": ok}
        claims.append(["Retention picks beat ≥ 90% of random circuits of similar cost (kernel AUC)",
                       f"{'SUPPORTED' if ok else 'not supported'} ({', '.join(f'{p:.0%}' for p in pct)})"])
    if "fidelity_vs_kernel_auc" in corr:
        claims.append(["Test fidelity predicts class information across circuits (|Spearman| ≥ 0.7)",
                       f"{'yes' if abs(corr['fidelity_vs_kernel_auc']) >= 0.7 else 'no'} "
                       f"(ρ = {corr['fidelity_vs_kernel_auc']:.2f})"])
    md += R.table(["claim", "status"], claims)
    with open(os.path.join(a.reports, f"objective_{ds}.md"), "w") as fh:
        fh.write(md)
    U.save_json(js, os.path.join(a.reports, f"objective_{ds}.json"))
    print(md)
    print(f"report: {a.reports}/objective_{ds}.md")


def main():
    """Command line: run, seeds, random circuits, stage and parallelism."""
    p = argparse.ArgumentParser(description="experiment 1: objective study")
    p.add_argument("--run", default="bloodmnist:centered")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--n-random", type=int, default=12)
    p.add_argument("--random-margin", type=int, default=3, help="CNOT range around the retention picks")
    p.add_argument("--random-seed", type=int, default=1234)
    p.add_argument("--classifiers", nargs="*", default=["hea8", "hea8_probs", "qcnn"])
    p.add_argument("--retention-root", default="experiments", help="seed-0 retention search root")
    p.add_argument("--searched-root", default="experiments_searched", help="seed-0 evaluation root")
    p.add_argument("--default-root", default="experiments_handref")
    p.add_argument("--out-root", default="experiments_objective")
    p.add_argument("--reports", default="reports")
    p.add_argument("--stage", nargs="*", default=["search", "arms", "evaluate", "report"],
                   choices=["search", "arms", "evaluate", "report"])
    p.add_argument("--jobs", type=int, default=2, help="circuits evaluated in parallel")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--quick", action="store_true", help="tiny settings (pipeline test)")
    a = p.parse_args()
    a.diag_args = ["--n-train", "300", "--n-test", "150", "--steps", "60", "--restarts", "1"] if a.quick else []
    a.clf_args = ["--epochs", "4"] if a.quick else []
    P = Paths(a)
    arms = None
    if "search" in a.stage:
        stage_search(a, P)
    if "arms" in a.stage:
        arms = stage_arms(a, P)
    if arms is None:
        saved = R.jload(P.study("arms.json"))
        if not saved:
            raise SystemExit(f"{P.study('arms.json')} not found; run --stage arms first")
        arms = saved["arms"]
    if "evaluate" in a.stage:
        stage_evaluate(a, P, arms)
    if "report" in a.stage:
        stage_report(a, P, arms)


if __name__ == "__main__":
    main()
