#!/usr/bin/env python3
# ----------------------------------------------------------------
# train_downstream.py  —  do the searched Fourier encodings work for
# downstream classifiers? Everything here is scored on the TEST split.
#
# Encodings: the search picks (experiments_fourier/<dataset>/search/
# picks.json) plus the literature references in encoding_genome.py.
# For each encoding:
#   quantum kernel SVMs   fidelity kernel |<psi(x)|psi(x')>|^2
#                         projected kernel (RBF on single-qubit Paulis,
#                         local measurements only)
#   trained classifiers   reupload: W0 [S(x) W] x R, Z/ZZ readout (Schuld et al.)
#                         qcnn:     S(x) once, then a hierarqcal-style QCNN
#                         hea8:     S(x) once, then an 8-layer HEA
#                         (mean +- std over config.CLF_SEEDS)
#   classical baselines   logistic regression and RBF SVM on the same d features;
#                         a classical Fourier model restricted to the encoding's
#                         own frequency spectrum (random Fourier features over
#                         that spectrum). If it matches the quantum models, the
#                         encoding's function class is classically reachable.
# and once per dataset: logistic regression / RBF SVM on all features.
#
# Usage (from the repository root, after search_encoding.py):
#   python search_fourier/train_downstream.py
#   python search_fourier/train_downstream.py --dataset bloodmnist
#   python search_fourier/train_downstream.py --no-references      # picks only
#   python search_fourier/train_downstream.py --quick              # match search_encoding.py --quick
#
# Output: experiments_fourier/<dataset>/downstream/
#   <encoding>.json              all metrics, per-seed curves, trained weights
#   encodings/<encoding>.json    genome + best re-uploading classifier
#                                (load with pennylane_encoding.py)
#   classical_all_features.json  summary.csv  results.png
# and experiments_fourier/summary_downstream.csv
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import csv
import glob
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import classifiers as CL
import common as U
import config as C
import encoding_genome as G


def fourier_baseline(g, Xtr, ytr, Xte, yte, seed):
    """Classical baseline with the SAME frequency spectrum as the encoding: logistic
    regression on cos/sin of all accessible frequency vectors.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sets = G.frequency_sets(g)
    d = g["n_features"]
    enc = [f for f in range(d) if len(sets[f]) > 1]
    if not enc:
        return None
    rng = np.random.default_rng(seed)
    Om = np.zeros((C.RFF_N_FEATURES, d))
    for f in enc:
        Om[:, f] = rng.choice(sets[f], C.RFF_N_FEATURES)

    def phi(X):
        """Random-Fourier-style feature map cos(Omega x), sin(Omega x)."""
        Z = X[:, :d] @ Om.T
        return np.hstack([np.cos(Z), np.sin(Z)])

    sc = StandardScaler().fit(phi(Xtr))
    lr = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    lr.fit(sc.transform(phi(Xtr)), ytr)
    B = sc.transform(phi(Xte))
    out = U.summarise(yte, lr.predict(B), lr.predict_proba(B), lr.classes_)
    if any(l["kind"] == "zz" for l in g["layers"]):
        out["note"] = "per-feature spectrum only; ZZ product terms not included"
    return out


def splits(feats):
    """Kernel-training, fit, validation and (optionally capped) test sets for the
    downstream models.
    """
    rng = np.random.default_rng(C.DATA_SEED + 7)
    Xtr, ytr = feats["train"]
    Xte, yte = feats["test"]
    test_idx = np.arange(len(yte))
    if C.TEST_MAX and len(yte) > C.TEST_MAX:
        test_idx = np.sort(U.stratified_sample(yte, C.TEST_MAX, rng))
        Xte, yte = Xte[test_idx], yte[test_idx]
    k = U.stratified_sample(ytr, C.KERNEL_N_TRAIN, rng)
    f = U.stratified_sample(ytr, C.VQC_N_TRAIN, rng)
    if "val" in feats:
        Xva, yva = feats["val"]
        v = U.stratified_sample(yva, C.VQC_N_VAL, rng)
        val = (Xva[v], yva[v])
    else:
        rest = np.setdiff1d(np.arange(len(ytr)), f)
        v = rest[U.stratified_sample(ytr[rest], C.VQC_N_VAL, rng)]
        val = (Xtr[v], ytr[v])
    return {"kernel": (Xtr[k], ytr[k]), "fit": (Xtr[f], ytr[f]), "val": val, "test": (Xte, yte),
            "test_idx": test_idx}


def load_encodings(dataset, with_refs):
    """Encodings to evaluate: the search picks plus (optionally) the reference encodings,
    with their held-out search AUC.
    """
    encs, sel = {}, {}
    p = os.path.join(C.OUT_ROOT, dataset, "search", "picks.json")
    if os.path.exists(p):
        pk = U.load_json(p)
        for r in pk["all"]:
            sel[G.key(r["genome"])] = r["sel_auc"]
        for r in pk["picks"]:
            encs[r["name"]] = r["genome"]
    else:
        print(f"  note: {p} not found — run search_encoding.py first; using references only")
    if with_refs or not encs:
        encs.update(G.references())
    return encs, sel


def _train_kind(kind, g, D, K):
    """Train one classifier type on top of the encoding (all seeds)."""
    (Xf, yf), (Xv, yv), (Xt, yt) = D["fit"], D["val"], D["test"]
    circ = CL.build(kind, K, prefix_ops=G.encoding_ops(g))
    res = CL.train_seeds(circ, K, Xf, yf, Xv, yv, Xt, yt, C.CLF_SEEDS)
    res["n_params"] = circ.n_params
    res["spec_version"] = CL.spec_version(kind)
    if kind == "reupload":
        res["model_mean_degree"] = G.spectrum_summary(g)["mean_degree"] * C.VQC_REUPLOADS
    for r in res["runs"]:
        r["model"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                      for k, v in r["model"].items()}
    return res


def missing_classifiers(out):
    """Classifier types whose cached result is missing or outdated."""
    return [k for k in C.CLASSIFIERS if not CL.result_is_current(out.get(k), k)]


def evaluate_encoding(name, g, D, K, prev=None):
    """Full evaluation, or only the classifiers missing from `prev`."""
    t0 = time.time()
    (Xk, yk), (Xf, yf), (Xv, yv), (Xt, yt) = D["kernel"], D["fit"], D["val"], D["test"]
    if prev is not None and prev.get("genome") == g:
        out = prev
        out.setdefault("test_split_idx", D["test_idx"].tolist())
        for kind in missing_classifiers(out):
            out[kind] = _train_kind(kind, g, D, K)
        out["seconds"] = round(out.get("seconds", 0) + time.time() - t0, 1)
        return out
    enc = G.encoding_circuit(g)
    Sk, St = enc.states(Xk), enc.states(Xt)
    out = {"name": name, "genome": g, "description": G.describe(g),
           "cost": G.two_qubit_cost(g), "spectrum": G.spectrum_summary(g),
           "test_split_idx": D["test_idx"].tolist()}

    clf, Cv = U.fit_kernel_svm(U.fidelity_kernel(Sk, Sk), yk)
    out["qk_fidelity"] = {**U.eval_kernel_svm(clf, U.fidelity_kernel(St, Sk), yt), "C": Cv}
    Pk, Pt = enc.pauli_features(Sk), enc.pauli_features(St)
    gam = U.median_gamma(Pk)
    clf, Cv = U.fit_kernel_svm(U.rbf_kernel(Pk, Pk, gam), yk)
    out["qk_projected"] = {**U.eval_kernel_svm(clf, U.rbf_kernel(Pt, Pk, gam), yt), "C": Cv}

    for kind in C.CLASSIFIERS:
        out[kind] = _train_kind(kind, g, D, K)

    d = g["n_features"]
    out["classical_same_features"] = U.classical_models(Xk[:, :d], yk, Xt[:, :d], yt)
    out["classical_fourier_same_spectrum"] = fourier_baseline(g, Xk, yk, Xt, yt, C.DATA_SEED)
    out["seconds"] = round(time.time() - t0, 1)
    return out


_W = {}


def _init_worker(D, K):
    """Worker-process setup: one BLAS thread per worker and the shared data."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:
        pass
    _W.update(D=D, K=K)


def _work(item):
    """Evaluate one encoding in a worker process."""
    name, g, prev = item
    return evaluate_encoding(name, g, _W["D"], _W["K"], prev)


def summary_row(out, sel_auc):
    """One summary-table row per encoding (all models, test metrics)."""
    r = {"encoding": out["name"], "cost": out["cost"], "n_features": out["genome"]["n_features"],
         "mean_degree": out["spectrum"]["mean_degree"], "held_out_search_auc": sel_auc}
    for k in ("qk_fidelity", "qk_projected"):
        r[f"{k}_balacc"], r[f"{k}_auc"] = out[k]["bal_acc"], out[k]["auc"]
    for k in C.CLASSIFIERS:
        if k in out:
            r[f"{k}_balacc"] = out[k]["test_bal_acc_mean"]
            r[f"{k}_balacc_std"] = out[k]["test_bal_acc_std"]
            r[f"{k}_auc"] = out[k]["test_auc_mean"]
    cs = out["classical_same_features"]
    r.update({"cl_logreg_balacc": cs["logreg"]["bal_acc"], "cl_logreg_auc": cs["logreg"]["auc"],
              "cl_rbf_balacc": cs["rbf_svm"]["bal_acc"], "cl_rbf_auc": cs["rbf_svm"]["auc"]})
    cf = out["classical_fourier_same_spectrum"]
    if cf:
        r.update({"cl_fourier_balacc": cf["bal_acc"], "cl_fourier_auc": cf["auc"]})
    r["description"] = out["description"]
    return r


def export(ddir, out):
    """Save the encoding (and trained re-uploading parameters) as a standalone JSON file
    that tools/pennylane_encoding.py can rebuild.
    """
    edir = os.path.join(ddir, "encodings")
    os.makedirs(edir, exist_ok=True)
    exp = {"name": out["name"], "genome": out["genome"], "description": out["description"],
           "two_qubit_cost": out["cost"], "n_features": out["genome"]["n_features"],
           "features": "experiments_fourier/<dataset>/features/<split>_features.npy, first n_features columns"}
    if "reupload" in out:
        best = max(out["reupload"]["runs"], key=lambda r: r["val_bal_acc"])
        exp["reupload_classifier"] = {"reuploads": C.VQC_REUPLOADS, "seed": best["seed"],
                                      "val_bal_acc": best["val_bal_acc"],
                                      "test_bal_acc": best["test"]["bal_acc"], **best["model"]}
    U.save_json(exp, os.path.join(edir, f"{out['name']}.json"))


def plot(ddir, rows, ref, dataset):
    """Test balanced accuracy and AUC of every model for every encoding."""
    plt = U.get_plt()
    models = [("qk_fidelity", "kernel: fidelity"), ("qk_projected", "kernel: projected")]
    models += [(k, f"trained: {k}") for k in C.CLASSIFIERS]
    models += [("cl_rbf", "classical RBF (same d)"), ("cl_fourier", "classical Fourier (same spectrum)")]

    def val(r, key):
        """Numeric value of a summary field (NaN if missing)."""
        v = r.get(key)
        return float(v) if v not in ("", None) else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(14, 1.0 + 0.45 * len(rows) * len(models) / 2.5))
    h = 0.8 / len(models)
    for ax, met in zip(axes, ("balacc", "auc")):
        allv = []
        for j, (m, lab) in enumerate(models):
            vals = [val(r, f"{m}_{met}") for r in rows]
            allv += vals
            ax.barh(np.arange(len(rows)) + (j - (len(models) - 1) / 2) * h, vals, h, label=lab)
        ax.axvline(ref[met], color="k", ls="--", lw=1, label="classical RBF, all features")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['encoding']} (cost {r['cost']})" for r in rows], fontsize=8)
        ax.set_title(f"test {'balanced accuracy' if met == 'balacc' else 'AUC'}")
        ax.set_xlim(max(0.0, np.nanmin(allv) - 0.05), 1.0)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle(f"{dataset}: downstream performance of the encodings")
    fig.tight_layout(); fig.savefig(os.path.join(ddir, "results.png"), dpi=150); plt.close(fig)


def run(dataset, args):
    """Evaluate every encoding of one dataset (resumable) and write the summary and plots."""
    t0 = time.time()
    print(f"\n{'=' * 70}\n[{dataset}] downstream evaluation")
    feats, _ = U.load_or_build_features(dataset)
    if "test" not in feats:
        raise SystemExit(f"{dataset}: no test split")
    D = splits(feats)
    K = int(max(feats["train"][1].max(), feats["test"][1].max())) + 1
    encs, sel = load_encodings(dataset, not args.no_references)
    ddir = U.out_dir(dataset, "downstream")
    print(f"  {K} classes | kernel train {len(D['kernel'][1])}, classifier fit {len(D['fit'][1])}, "
          f"val {len(D['val'][1])}, test {len(D['test'][1])} | seeds {C.CLF_SEEDS}")

    ref_path = os.path.join(ddir, "classical_all_features.json")
    if os.path.exists(ref_path) and not args.force:
        ref_all = U.load_json(ref_path)
    else:
        (Xk, yk), (Xt, yt) = D["kernel"], D["test"]
        ref_all = U.classical_models(Xk, yk, Xt, yt)
        U.save_json(ref_all, ref_path)
    print(f"  classical on all {feats['train'][0].shape[1]} features (bal_acc/AUC): logreg "
          f"{ref_all['logreg']['bal_acc']:.3f}/{ref_all['logreg']['auc']:.3f}, RBF SVM "
          f"{ref_all['rbf_svm']['bal_acc']:.3f}/{ref_all['rbf_svm']['auc']:.3f}")

    todo = []
    for name, g in encs.items():
        path = os.path.join(ddir, f"{name}.json")
        prev = None if args.force or not os.path.exists(path) else U.load_json(path)
        if prev is not None and prev.get("genome") != g:
            prev = None
        if prev is None or missing_classifiers(prev):
            todo.append((name, g, prev))
    n_workers = min(C.N_WORKERS or min(8, os.cpu_count() or 1), max(1, len(todo)))
    if todo:
        n_full = sum(1 for t in todo if t[2] is None)
        print(f"  {len(todo)} encodings to (re)train on {n_workers} worker(s): {n_full} in full, "
              f"{len(todo) - n_full} only for missing classifiers "
              f"({', '.join(sorted({k for t in todo if t[2] for k in missing_classifiers(t[2])}))})")
        if n_workers > 1:
            with ProcessPoolExecutor(n_workers, initializer=_init_worker, initargs=(D, K)) as pool:
                results = pool.map(_work, todo)
                for out in results:
                    U.save_json(out, os.path.join(ddir, f"{out['name']}.json"))
                    export(ddir, out)
                    print(f"    finished {out['name']} ({out['seconds']:.0f}s)")
        else:
            _init_worker(D, K)
            for item in todo:
                out = _work(item)
                U.save_json(out, os.path.join(ddir, f"{out['name']}.json"))
                export(ddir, out)
                print(f"    finished {out['name']} ({out['seconds']:.0f}s)")

    rows = []
    for name, g in encs.items():
        out = U.load_json(os.path.join(ddir, f"{name}.json"))
        r = summary_row(out, sel.get(G.key(g)))
        rows.append(r)
        keys = ["qk_fidelity", "qk_projected"] + C.CLASSIFIERS + ["cl_rbf", "cl_fourier"]
        msg = " | ".join(f"{k} {r[f'{k}_balacc']:.3f}" for k in keys if f"{k}_balacc" in r)
        print(f"  {name:26s} cost {r['cost']:2d} d={r['n_features']:2d} | {msg}")

    for r in rows:
        r["dataset"] = dataset
        r["ref_all_features_rbf_balacc"] = ref_all["rbf_svm"]["bal_acc"]
        r["ref_all_features_rbf_auc"] = ref_all["rbf_svm"]["auc"]
    U.write_csv(rows, os.path.join(ddir, "summary.csv"))
    plot(ddir, rows, {"balacc": ref_all["rbf_svm"]["bal_acc"], "auc": ref_all["rbf_svm"]["auc"]}, dataset)
    print(f"  done in {time.time() - t0:.0f}s -> {ddir}")


def main():
    """Command line: datasets, references, quick mode, output folder."""
    p = argparse.ArgumentParser(description="downstream evaluation of Fourier encodings")
    p.add_argument("--dataset", nargs="*", default=None)
    p.add_argument("--no-references", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--out-root", default=None, help="results root (default config.OUT_ROOT)")
    a = p.parse_args()
    if a.quick:
        from search_encoding import apply_quick
        apply_quick()
        C.CLF_SEEDS = [0]
    if a.out_root:
        C.OUT_ROOT = a.out_root
    names = a.dataset or C.RUNS
    U.check_datasets(names)
    for d in names:
        run(d, a)
    rows = []
    for f in sorted(glob.glob(os.path.join(C.OUT_ROOT, "*", "downstream", "summary.csv"))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    U.write_csv(rows, os.path.join(C.OUT_ROOT, "summary_downstream.csv"))
    print(f"\nsummary: {C.OUT_ROOT}/summary_downstream.csv")


if __name__ == "__main__":
    main()
