#!/usr/bin/env python3
# ----------------------------------------------------------------
# noise_study.py  —  Experiment 2: do cheaper encoders keep their
# advantage when gates are noisy?
#
# Noise: two-qubit depolarizing noise with probability p after every
# two-qubit gate of the ENCODING, and single-qubit depolarizing noise with
# probability p * P1_RATIO after every single-qubit gate (noise.py). The
# classifier is kept noise-free so that only the encoders differ.
#
# Encoders (whatever exists is used):
#   aae_retention   selected retention-driven amplitude circuit (seed 0)
#   aae_handref     hand-designed 31-CNOT amplitude reference circuit
#   aae_fidelity    fidelity-selected circuit at matched cost (objective_study.py)
#   aae_literature  best literature ansatz within the searched circuits' CNOT
#                   budget, chosen on held-out data (objective_study.py)
#   aae_fidelity_deep   the fidelity search's own selection (deeper circuit)
#   aae_literature_deep the deepest literature ansatz (e.g. 56-CNOT hea_ry_8)
#   The two deep encoders test whether their noiseless advantage survives
#   gate noise, where every extra CNOT costs accuracy.
#   fourier_pick    Fourier search pick (seed 0)
#   fourier_ref     best literature reference encoding
#   fourier_zero    the entanglement-free Fourier pick, if there is one
# For each encoder and noise level:
#   - mean fidelity to the exact state (amplitude encoders)
#   - kernel SVM on the mixed-state kernel Tr(rho rho')
#   - a trained classifier (default hea8) on the noisy states, several seeds
#   - both again with a finite number of measurements (--shots): classifier
#     readouts from S computational-basis shots, kernel entries from S
#     swap-test shots
# Paired bootstrap tests compare aae_retention with aae_literature,
# aae_fidelity and aae_handref, and fourier_pick with fourier_ref, at every
# noise level.
#
# Usage (from the repository root):
#   python studies/noise_study.py                               # bloodmnist
#   python studies/noise_study.py --run pathmnist:centered --fourier-dataset pathmnist
#   python studies/noise_study.py --p 0 0.005 0.02 --workers 4
#   python studies/noise_study.py --report-only
#
# Output: experiments_noise/<dataset>/<encoder>/p<p>.json
#         reports/noise_<dataset>.md/.json, reports/fig_noise_<dataset>.png
# Runtime on 8 cores: roughly 30-60 min for the default settings.
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import classifiers as CL
import common as U
import config as C
import encoding_genome as G
import make_report as R
import noise as N


# ----------------------------------------------------------------
# encoders
# ----------------------------------------------------------------

def find_encoders(a):
    """Encoders to test: the retention pick and the hand-designed reference (amplitude),
    plus the Fourier picks and references, with the files they are loaded from.
    """
    ds, mode = a.run.split(":")
    enc = {}
    for name, root in (("aae_retention", a.searched_root), ("aae_handref", a.default_root)):
        p = os.path.join(root, ds, "a_phase_check", mode, "encoded_states.npz")
        if os.path.exists(p):
            enc[name] = {"kind": "aae", "npz": p,
                         "results": os.path.join(root, ds, "a_phase_check", mode, "results.json")}
    arms = R.jload(os.path.join(a.objective_root, ds, "arms.json"))
    deep_lit = None
    if arms and arms.get("literature_high_budget"):
        deep_lit = max(arms["literature_high_budget"], key=lambda n: arms["arms"][n]["n2q"])
    picks = {"aae_fidelity": "fidelity_matched_s0",
             "aae_literature": (arms or {}).get("literature_best"),
             "aae_fidelity_deep": "fidelity_best_s0",
             "aae_literature_deep": deep_lit}
    for name, arm_name in picks.items():
        if arms and arm_name in arms["arms"]:
            root = arms["arms"][arm_name]["eval_root"]
            p = os.path.join(root, ds, "a_phase_check", mode, "encoded_states.npz")
            if os.path.exists(p):
                enc[name] = {"kind": "aae", "npz": p, "arm": arm_name,
                             "results": os.path.join(root, ds, "a_phase_check", mode, "results.json")}
    fds = a.fourier_dataset
    if fds:
        four, _ = R.load_fourier(a.fourier_root, fds)
        picks = R.jload(os.path.join(a.fourier_root, fds, "search", "picks.json"))
        pick = R.pick_fourier_encoding(four)
        if pick:
            enc["fourier_pick"] = {"kind": "fourier", "genome": four[pick]["genome"], "name": pick}
            zero = [n for n, r in four.items() if n.startswith("search_") and r["cost"] == 0 and n != pick]
            if zero:
                enc["fourier_zero"] = {"kind": "fourier", "genome": four[zero[0]]["genome"], "name": zero[0]}
        if picks and picks.get("best_reference") in four:
            ref = picks["best_reference"]
            enc["fourier_ref"] = {"kind": "fourier", "genome": four[ref]["genome"], "name": ref}
    return enc


def seed_for(*parts):
    """Deterministic random seed from any combination of labels (stable across runs)."""
    return zlib.crc32("|".join(map(str, parts)).encode()) % (2 ** 31)


# ----------------------------------------------------------------
# one task = one encoder at one noise level
# ----------------------------------------------------------------

def _limit_threads():
    """One BLAS thread per worker process."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass


def run_aae(a, name, spec, p):
    """Noisy evaluation of an amplitude encoder at gate-error rate p: re-simulate the
    fitted circuits with depolarizing noise (trajectories), then fidelity, mixed-state
    kernel SVM, finite-shot kernel and trained classifiers on the noisy states.
    """
    from sklearn.model_selection import train_test_split
    from aae_sim import AAECircuit
    data = np.load(spec["npz"])
    res = U.load_json(spec["results"])
    circ = AAECircuit(res["genome"])
    y = data["labels"].astype(int)
    n_tr = int(data["n_train"])
    K = int(y.max()) + 1
    T = a.traj
    S = N.noisy_aae_states(circ, data["params"], p, p * a.p1_ratio, T, seed_for(name, p))
    idx = np.arange(n_tr)
    tr, va = train_test_split(idx, test_size=0.2, stratify=y[:n_tr], random_state=C.DATA_SEED)
    tr, va = np.sort(tr), np.sort(va)
    te = np.arange(n_tr, len(y))
    kidx = np.sort(U.stratified_sample(y[:n_tr], a.kernel_n, np.random.default_rng(C.DATA_SEED)))
    out = {"encoder": name, "p": p, "n2q": circ.n_two_qubit, "genome": res["genome"],
           "test_split_idx": np.asarray(data["eval_idx"]).tolist()}
    out["fidelity"] = float(N.mixed_fidelity(S[N.rows_of(te, T)], data["targets"][te], T).mean())
    Sk = S[N.rows_of(kidx, T)]
    Ktr = N.mixed_kernel(Sk, Sk, T)
    Kte = N.mixed_kernel(S[N.rows_of(te, T)], Sk, T)
    clf, Cv = U.fit_kernel_svm(Ktr, y[kidx])
    out["kernel"] = {**U.eval_kernel_svm(clf, Kte, y[te]), "C": Cv}
    out["classifiers"] = {}
    models = {}
    for kind in a.classifiers:
        cc = CL.build(kind, K)
        z = lambda n: np.zeros((n * T, 1))
        r = CL.train_seeds(cc, K, z(len(tr)), y[tr], z(len(va)), y[va], z(len(te)), y[te], C.CLF_SEEDS,
                           init_tr=S[N.rows_of(tr, T)], init_va=S[N.rows_of(va, T)],
                           init_te=S[N.rows_of(te, T)], n_traj=T)
        models[kind] = (cc, r)
        r["test_split_idx"] = out["test_split_idx"]
        out["classifiers"][kind] = r
    out["shots"] = shot_results(a, name, p, Ktr, Kte, y[kidx], y[te], models, K,
                                np.zeros((len(te) * T, 1)), S[N.rows_of(te, T)], T, out["test_split_idx"])
    for kind in models:
        for run in out["classifiers"][kind]["runs"]:
            run.pop("model", None)
    return out


def run_fourier(a, name, spec, p):
    """Noisy evaluation of a Fourier encoder at gate-error rate p (same models as for the
    amplitude encoders).
    """
    import train_downstream as TD
    C.OUT_ROOT = a.fourier_root
    feats, _ = U.load_or_build_features(a.fourier_dataset)
    D = TD.splits(feats)
    K = int(max(feats["train"][1].max(), feats["test"][1].max())) + 1
    T = a.traj
    circ = G.encoding_circuit(spec["genome"])
    Xk, yk = D["kernel"]
    sel = np.sort(U.stratified_sample(yk, a.kernel_n, np.random.default_rng(C.DATA_SEED)))
    Xk, yk = Xk[sel], yk[sel]
    states = {}
    for part, (X, _) in (("kernel", (Xk, yk)), ("fit", D["fit"]), ("val", D["val"]), ("test", D["test"])):
        states[part] = N.noisy_fourier_states(circ, X, p, p * a.p1_ratio, T, seed_for(name, p, part))
    yt = D["test"][1]
    out = {"encoder": name, "p": p, "n2q": G.two_qubit_cost(spec["genome"]), "genome": spec["genome"],
           "fourier_name": spec["name"], "test_split_idx": D["test_idx"].tolist()}
    Ktr = N.mixed_kernel(states["kernel"], states["kernel"], T)
    Kte = N.mixed_kernel(states["test"], states["kernel"], T)
    clf, Cv = U.fit_kernel_svm(Ktr, yk)
    out["kernel"] = {**U.eval_kernel_svm(clf, Kte, yt), "C": Cv}
    out["classifiers"] = {}
    models = {}
    for kind in a.classifiers:
        cc = CL.build(kind, K)
        z = lambda n: np.zeros((n * T, 1))
        (Xf, yf), (Xv, yv) = D["fit"], D["val"]
        r = CL.train_seeds(cc, K, z(len(yf)), yf, z(len(yv)), yv, z(len(yt)), yt, C.CLF_SEEDS,
                           init_tr=states["fit"], init_va=states["val"], init_te=states["test"], n_traj=T)
        models[kind] = (cc, r)
        r["test_split_idx"] = out["test_split_idx"]
        out["classifiers"][kind] = r
    out["shots"] = shot_results(a, name, p, Ktr, Kte, yk, yt, models, K,
                                np.zeros((len(yt) * T, 1)), states["test"], T, out["test_split_idx"])
    for kind in models:
        for run in out["classifiers"][kind]["runs"]:
            run.pop("model", None)
    return out


def shot_results(a, name, p, Ktr, Kte, ytr, yte, models, K, Xte_dummy, init_te, T, test_idx):
    """Same kernels and trained classifiers, evaluated with finite measurement budgets."""
    out = {}
    for shots in a.shots:
        rng = np.random.default_rng(seed_for(name, p, shots, "kernel"))
        Ktr_s = N.swap_test_kernel(Ktr, shots, rng, symmetric=True)
        Kte_s = N.swap_test_kernel(Kte, shots, rng)
        clf, Cv = U.fit_kernel_svm(Ktr_s, ytr)
        entry = {"kernel": {**U.eval_kernel_svm(clf, Kte_s, yte), "C": Cv}, "classifiers": {}}
        for kind, (cc, r) in models.items():
            runs = []
            for run in r["runs"]:
                te, Pr = CL.evaluate_shots(cc, run["model"], Xte_dummy, yte, K, shots,
                                           seed_for(name, p, shots, kind, run["seed"]),
                                           init=init_te, n_traj=T)
                runs.append({"seed": run["seed"], "test": te, "test_proba": np.round(Pr, 6).tolist()})
            ba = [x["test"]["bal_acc"] for x in runs]
            au = [x["test"]["auc"] for x in runs]
            entry["classifiers"][kind] = {
                "test_bal_acc_mean": float(np.mean(ba)), "test_bal_acc_std": float(np.std(ba)),
                "test_auc_mean": float(np.nanmean(au)), "runs": runs,
                "test_labels": np.asarray(yte).tolist(), "test_split_idx": test_idx}
        out[str(shots)] = entry
    return out


def _task(args):
    """Run one (encoder, p) evaluation in a worker and save it with the settings used."""
    a, name, spec, p, path = args
    _limit_threads()
    t = time.time()
    out = run_aae(a, name, spec, p) if spec["kind"] == "aae" else run_fourier(a, name, spec, p)
    out["seconds"] = round(time.time() - t, 1)
    out["settings"] = {"traj": a.traj, "p1_ratio": a.p1_ratio, "kernel_n": a.kernel_n, "shots": a.shots,
                       "classifiers": a.classifiers, "training": CL.training_signature()}
    U.save_json(out, path)
    return name, p, out["seconds"]


def result_path(a, name, p):
    """File of one (encoder, p) result."""
    ds = a.run.split(":")[0] if name.startswith("aae") else a.fourier_dataset
    return os.path.join(a.out_root, ds, name, f"p{p:g}.json")


def current(a, path):
    """True if a saved result exists and was made with the current settings."""
    r = R.jload(path)
    return (r is not None and r.get("settings", {}).get("traj") == a.traj
            and r["settings"].get("p1_ratio") == a.p1_ratio
            and r["settings"].get("kernel_n") == a.kernel_n
            and r["settings"].get("shots") == a.shots
            and r["settings"].get("training") == CL.training_signature()
            and all(k in r.get("classifiers", {}) for k in a.classifiers))


# ----------------------------------------------------------------
# report
# ----------------------------------------------------------------

def paired(a, ra, rb):
    """Paired bootstrap of two results on the same test images (None if not comparable)."""
    if not (ra and rb) or ra.get("test_split_idx") != rb.get("test_split_idx"):
        return None
    if ra["test_labels"] != rb["test_labels"]:
        return None
    y = np.asarray(ra["test_labels"])
    return R.paired_bootstrap(y, R.seed_probas(ra), R.seed_probas(rb), int(y.max()) + 1, a.bootstrap, 0)


def report(a, enc):
    """Markdown/JSON report: metrics vs. p for every encoder, crossover points and paired
    comparisons (reports/noise_<dataset>.md).
    """
    ds = a.run.split(":")[0]
    res = {n: {p: R.jload(result_path(a, n, p)) for p in a.p} for n in enc}
    res = {n: {p: r for p, r in v.items() if r} for n, v in res.items()}
    res = {n: v for n, v in res.items() if v}
    if not res:
        raise SystemExit("no noise results yet")
    os.makedirs(a.reports, exist_ok=True)
    c0 = a.classifiers[0]
    md = f"# Noise study ({ds})\n\n"
    md += (f"Two-qubit depolarizing noise p after every two-qubit gate of the encoding, single-qubit noise "
           f"{a.p1_ratio:g}·p after every single-qubit gate; {a.traj} noise trajectories per image; classifier "
           f"noise-free. Amplitude encoders: {a.run}, 800 test images. Fourier encoders: {a.fourier_dataset}, "
           f"full test split.\n\n")
    head = ["encoder", "CNOTs"] + [f"p = {p:g}" for p in a.p]
    for lab, fn in ((f"{c0} balanced accuracy (mean over seeds)", lambda r: r["classifiers"][c0]["test_bal_acc_mean"]),
                    ("kernel SVM balanced accuracy", lambda r: r["kernel"]["bal_acc"]),
                    ("mean fidelity to the exact state", lambda r: r.get("fidelity"))):
        rows = []
        for n, v in res.items():
            if lab.startswith("mean fidelity") and not n.startswith("aae"):
                continue
            n2q = next(iter(v.values()))["n2q"]
            rows.append([n, n2q] + [("—" if p not in v or fn(v[p]) is None else f"{fn(v[p]):.3f}") for p in a.p])
        if rows:
            md += f"## {lab}\n\n" + R.table(head, rows)

    js = {"dataset": ds, "p": a.p, "tests": {}, "results": {
        n: {str(p): {"n2q": r["n2q"], "fidelity": r.get("fidelity"), "kernel_bal_acc": r["kernel"]["bal_acc"],
                     "kernel_auc": r["kernel"]["auc"],
                     "clf": {k: {"bal_acc": c["test_bal_acc_mean"], "bal_acc_std": c["test_bal_acc_std"],
                                 "auc": c["test_auc_mean"]} for k, c in r["classifiers"].items()}}
            for p, r in v.items()} for n, v in res.items()}}
    pairs = [("aae_retention", "aae_literature"), ("aae_retention", "aae_fidelity"),
             ("aae_retention", "aae_fidelity_deep"), ("aae_retention", "aae_literature_deep"),
             ("aae_retention", "aae_handref"), ("fourier_pick", "fourier_ref")]
    trows = []
    for x, y in pairs:
        if x not in res or y not in res:
            continue
        row = [f"{x} ({next(iter(res[x].values()))['n2q']}) − {y} ({next(iter(res[y].values()))['n2q']})"]
        for p in a.p:
            t = None
            if p in res[x] and p in res[y]:
                t = paired(a, res[x][p]["classifiers"].get(c0), res[y][p]["classifiers"].get(c0))
            js["tests"][f"{x}|{y}|{p:g}"] = t
            row.append(R.fmt_ci(t["bal_acc"]) if t else "n/a")
        trows.append(row)
    if trows:
        md += (f"## Paired comparisons under noise ({c0}, Δ balanced accuracy, 95% CI, * = excludes 0)\n\n"
               + R.table(["comparison"] + [f"p = {p:g}" for p in a.p], trows))

    if a.shots:
        md += (f"## Finite number of measurements ({c0} and kernel SVM)\n\n"
               "Readouts estimated from S measurement shots of one setting (same trained models); kernel entries "
               "estimated with S swap-test shots each.\n\n")
        head2 = ["encoder", "CNOTs", "S"] + [f"p = {p:g}" for p in a.p]
        srows = []
        for n, v in res.items():
            n2q = next(iter(v.values()))["n2q"]
            for S_ in ["exact"] + [str(s) for s in a.shots]:
                row = [n, n2q, S_]
                for p in a.p:
                    r = v.get(p)
                    if not r:
                        row.append("—")
                    elif S_ == "exact":
                        row.append(f"{r['classifiers'][c0]['test_bal_acc_mean']:.3f} / {r['kernel']['bal_acc']:.3f}")
                    elif S_ in r.get("shots", {}):
                        e = r["shots"][S_]
                        row.append(f"{e['classifiers'][c0]['test_bal_acc_mean']:.3f} / {e['kernel']['bal_acc']:.3f}")
                    else:
                        row.append("—")
                srows.append(row)
        md += "Entries: classifier balanced accuracy / kernel SVM balanced accuracy.\n\n" + R.table(head2, srows)
        smin = str(min(a.shots))
        trows2 = []
        for x, y in pairs:
            if x not in res or y not in res:
                continue
            row = [f"{x} − {y}"]
            for p in a.p:
                t = None
                if p in res[x] and p in res[y] and smin in res[x][p].get("shots", {}) \
                        and smin in res[y][p].get("shots", {}):
                    t = paired(a, res[x][p]["shots"][smin]["classifiers"].get(c0),
                               res[y][p]["shots"][smin]["classifiers"].get(c0))
                js["tests"][f"{x}|{y}|{p:g}|S{smin}"] = t
                row.append(R.fmt_ci(t["bal_acc"]) if t else "n/a")
            trows2.append(row)
        if trows2:
            md += (f"Paired comparisons with S = {smin} shots ({c0}, Δ balanced accuracy, 95% CI):\n\n"
                   + R.table(["comparison"] + [f"p = {p:g}" for p in a.p], trows2))
        js["shots"] = {n: {str(p): {S_: {"clf_bal_acc": e["classifiers"][c0]["test_bal_acc_mean"],
                                         "kernel_bal_acc": e["kernel"]["bal_acc"]}
                                    for S_, e in r.get("shots", {}).items()}
                               for p, r in v.items()} for n, v in res.items()}

    plt = U.get_plt()
    groups = [("amplitude encoders", [n for n in res if n.startswith("aae")]),
              ("Fourier encoders", [n for n in res if n.startswith("fourier")])]
    groups = [g for g in groups if g[1]]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.6 * len(groups), 4), squeeze=False)
    for ax, (title, names) in zip(axes[0], groups):
        for n in names:
            ps = [p for p in a.p if p in res[n]]
            m = [res[n][p]["classifiers"][c0]["test_bal_acc_mean"] for p in ps]
            s = [res[n][p]["classifiers"][c0]["test_bal_acc_std"] for p in ps]
            k = [res[n][p]["kernel"]["bal_acc"] for p in ps]
            line = ax.errorbar(ps, m, yerr=s, marker="o", capsize=3,
                               label=f"{n} ({res[n][ps[0]]['n2q']} CNOTs), {c0}")
            ax.plot(ps, k, ls="--", color=line[0].get_color(), alpha=0.7)
        ax.set_xlabel("two-qubit depolarizing probability p"); ax.set_ylabel("test balanced accuracy")
        ax.set_title(f"{title} (solid: {c0}; dashed: kernel SVM)", fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fname = f"fig_noise_{ds}.png"
    fig.savefig(os.path.join(a.reports, fname), dpi=150); plt.close(fig)
    md += f"\n![]({fname})\n\n"

    claims = []
    for x, y in pairs:
        ts = [js["tests"].get(f"{x}|{y}|{p:g}") for p in a.p]
        ts = [t for t in ts if t]
        if not ts:
            continue
        pos = sum(t["bal_acc"]["significant"] and t["bal_acc"]["diff"] > 0 for t in ts)
        neg = sum(t["bal_acc"]["significant"] and t["bal_acc"]["diff"] < 0 for t in ts)
        first, last = ts[0]["bal_acc"]["diff"], ts[-1]["bal_acc"]["diff"]
        claims.append([f"{x} better than {y} at every noise level",
                       f"{'SUPPORTED' if pos == len(ts) else 'not supported'} ({pos}/{len(ts)} significant, {neg} worse); "
                       f"Δ {first:+.3f} at p = {a.p[0]:g} → {last:+.3f} at p = {a.p[-1]:g}"])
    for n, v in res.items():
        ps = sorted(v)
        if len(ps) >= 2:
            drop = v[ps[0]]["classifiers"][c0]["test_bal_acc_mean"] - v[ps[-1]]["classifiers"][c0]["test_bal_acc_mean"]
            js.setdefault("drop", {})[n] = drop
    if claims:
        md += "## Claims\n\n" + R.table(["claim", "status"], claims)
    if js.get("drop"):
        md += ("Accuracy lost between the lowest and highest noise level: "
               + ", ".join(f"{n} {d:+.3f}" for n, d in js["drop"].items()) + ".\n")
    with open(os.path.join(a.reports, f"noise_{ds}.md"), "w") as fh:
        fh.write(md)
    U.save_json(js, os.path.join(a.reports, f"noise_{ds}.json"))
    print(md)
    print(f"report: {a.reports}/noise_{ds}.md")


def main():
    """Command line: noise levels, trajectories, shots, encoders, workers; runs what is
    missing, then reports.
    """
    p = argparse.ArgumentParser(description="experiment 2: noise study")
    p.add_argument("--run", default="bloodmnist:centered", help="amplitude-encoding dataset:mode")
    p.add_argument("--fourier-dataset", default="bloodmnist", help="'' to skip Fourier encoders")
    p.add_argument("--p", nargs="*", type=float, default=[0.0, 0.002, 0.005, 0.01, 0.02])
    p.add_argument("--p1-ratio", type=float, default=0.1)
    p.add_argument("--traj", type=int, default=4, help="noise trajectories per image")
    p.add_argument("--kernel-n", type=int, default=1000, help="training images for the kernel SVM")
    p.add_argument("--classifiers", nargs="*", default=["hea8"])
    p.add_argument("--shots", nargs="*", type=int, default=[1000, 100],
                   help="finite measurement budgets per readout (classifiers) and per kernel entry (swap test)")
    p.add_argument("--encoders", nargs="*", default=None, help="subset of encoder names")
    p.add_argument("--searched-root", default="experiments_searched")
    p.add_argument("--default-root", default="experiments_handref")
    p.add_argument("--objective-root", default="experiments_objective")
    p.add_argument("--fourier-root", default="experiments_fourier")
    p.add_argument("--out-root", default="experiments_noise")
    p.add_argument("--reports", default="reports")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--report-only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true", help="tiny settings (pipeline test)")
    a = p.parse_args()
    a.p = sorted(a.p)
    if a.quick:
        C.VQC_EPOCHS, C.CLF_SEEDS = 3, [0]
        C.KERNEL_N_TRAIN, C.VQC_N_TRAIN, C.VQC_N_VAL, C.TEST_MAX = 300, 200, 100, 300
        a.kernel_n = min(a.kernel_n, 200)
    enc = find_encoders(a)
    if a.encoders:
        enc = {k: v for k, v in enc.items() if k in a.encoders}
    print("encoders: " + ", ".join(enc) if enc else "no encoders found")
    if not enc:
        return
    if not a.report_only:
        tasks = []
        for n, spec in enc.items():
            for pv in a.p:
                path = result_path(a, n, pv)
                if a.force or not current(a, path):
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    tasks.append((a, n, spec, pv, path))
        workers = a.workers or min(8, os.cpu_count() or 1, max(1, len(tasks)))
        print(f"{len(tasks)} encoder×noise tasks on {workers} worker(s)")
        t0 = time.time()
        if workers > 1 and len(tasks) > 1:
            with ProcessPoolExecutor(workers) as pool:
                for n, pv, sec in pool.map(_task, tasks):
                    print(f"  {n} p={pv:g} done ({sec:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
        else:
            for t in tasks:
                n, pv, sec = _task(t)
                print(f"  {n} p={pv:g} done ({sec:.0f}s)", flush=True)
    report(a, enc)


if __name__ == "__main__":
    main()
