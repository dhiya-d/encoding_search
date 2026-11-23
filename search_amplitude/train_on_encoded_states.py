#!/usr/bin/env python3
# ----------------------------------------------------------------
# train_on_encoded_states.py  —  downstream test of the searched
# AMPLITUDE encoders: can a measurement-based classifier use them?
#
# Uses the encoded states saved by diag_a_phase_check.py
# (encoded_states.npz: 2000 train + 800 test images for bloodmnist,
# the full splits for breastmnist). For each input source
#   exact     the exact amplitude states (the ceiling for any encoder)
#   searched  the states from the searched circuit
#   default   the states from the default 31-CNOT circuit (if available)
# it trains these classifiers (default), each for every seed in config.CLF_SEEDS:
#   hea, hea8, hea12   hardware-efficient ansatz with 4 / 8 / 12 layers,
#                      readout <Z_q>, <Z_q Z_q+1> + linear layer
#   hea8_probs         8 layers, readout = all 256 basis probabilities (scaled)
#   qcnn               hierarqcal-style QCNN, readout = basis probabilities of
#                      the remaining qubits + linear layer
# Validation (early stopping) is a stratified 20% of the training part;
# the test part is never used for training or model selection.
#
# Usage (from the repository root):
#   python search_amplitude/train_on_encoded_states.py
#   python search_amplitude/train_on_encoded_states.py --runs bloodmnist:centered
#   python search_amplitude/train_on_encoded_states.py --compare-root experiments_handref
#   python search_amplitude/train_on_encoded_states.py --classifiers hea8 hea16 --epochs 80 --patience 15
#
# Cached results are reused only if the classifier definition, the training
# settings and the saved test predictions all match (see
# classifiers.result_is_current); otherwise they are recomputed.
#
# Output: experiments_searched/<dataset>/downstream_classifiers/<mode>/
#   <source>_<classifier>.json   per-seed metrics, curves, trained weights
#   summary.csv   results.png
# and experiments_searched/summary_downstream_classifiers.csv
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import glob
import os
import time

import numpy as np

import classifiers as CL
import common as U
import config as C

DEFAULT_RUNS = ["bloodmnist:centered", "breastmnist:uncentered"]


def load_npz(root, dataset, mode):
    """Encoded states saved by diag_a_phase_check.py for one run, and the file path."""
    p = os.path.join(root, dataset, "a_phase_check", mode, "encoded_states.npz")
    return (np.load(p), p) if os.path.exists(p) else (None, p)


def load_diag_a(root, dataset, mode):
    """Results of diag_a_phase_check.py for one run (None if missing)."""
    p = os.path.join(root, dataset, "a_phase_check", mode, "results.json")
    return U.load_json(p) if os.path.exists(p) else None


def split_train_val(y, seed, frac=0.2):
    """Stratified split of the training indices into fit and validation parts."""
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=frac, stratify=y, random_state=seed)
    return np.sort(tr), np.sort(va)


def plot(odir, rows, tag, kernel_ref):
    """Bar chart of test balanced accuracy and AUC per classifier and state source, with
    the fidelity-kernel SVM results as reference lines.
    """
    plt = U.get_plt()
    sources = [s for s in ("exact", "searched", "default") if any(r["source"] == s for r in rows)]
    clfs = list(dict.fromkeys(r["classifier"] for r in rows))
    fig, axes = plt.subplots(1, 2, figsize=(max(11, 2.2 * len(clfs) + 4), 4.2))
    w = 0.8 / len(sources)
    for ax, key in zip(axes, ("bal_acc", "auc")):
        for i, s in enumerate(sources):
            vals = [next((r for r in rows if r["source"] == s and r["classifier"] == c), None) for c in clfs]
            ax.bar(np.arange(len(clfs)) + (i - (len(sources) - 1) / 2) * w,
                   [v[f"test_{key}_mean"] if v else 0 for v in vals], w,
                   yerr=[v[f"test_{key}_std"] if v else 0 for v in vals], label=s, capsize=3)
        for name, val in kernel_ref.get(key, {}).items():
            ax.axhline(val, ls="--", lw=0.9, color="k" if name == "exact" else "grey")
            ax.text(len(clfs) - 0.5, val, f" kernel SVM ({name})", fontsize=7, va="bottom", ha="right")
        chance = rows[0]["chance"] if key == "bal_acc" else 0.5
        ax.axhline(chance, ls=":", color="r", lw=0.9)
        ax.set_xticks(range(len(clfs))); ax.set_xticklabels(clfs); ax.set_title(f"test {key}")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{tag}: measurement-based classifiers on amplitude-encoded states")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "results.png"), dpi=150); plt.close(fig)


def run(args, dataset, mode):
    """Train every classifier on every state source (exact targets, searched circuit,
    reference circuit) of one run; results are cached per (source, classifier).
    """
    tag = f"{dataset}/{mode}"
    data, path = load_npz(args.states_root, dataset, mode)
    if data is None:
        print(f"[{tag}] {path} not found — run diag_a_phase_check.py with --genome-root first")
        return []
    odir = os.path.join(args.states_root, dataset, args.out_name, mode)
    os.makedirs(odir, exist_ok=True)
    n_tr = int(data["n_train"])
    y = data["labels"].astype(int)
    K = int(y.max()) + 1
    # State sources: exact targets (ceiling), the searched circuit, and the
    # hand-designed reference circuit (key "default", kept for compatibility
    # with existing result files).
    sources = {"exact": np.asarray(data["targets"]), "searched": np.asarray(data["states"])}
    if args.compare_root:
        cmp_, cpath = load_npz(args.compare_root, dataset, mode)
        if cmp_ is not None and np.array_equal(cmp_["labels"], data["labels"]) \
                and np.array_equal(cmp_["eval_idx"], data["eval_idx"]):
            sources["default"] = np.asarray(cmp_["states"])
        elif cmp_ is not None:
            print(f"  [{tag}] {cpath} uses a different image sample — skipping 'default'")
    sources = {k: v for k, v in sources.items() if k in args.sources}
    tr, va = split_train_val(y[:n_tr], C.DATA_SEED)
    te = np.arange(n_tr, len(y))
    Xdummy = np.zeros((len(y), 1))
    print(f"\n[{tag}] {K} classes | fit {len(tr)}, val {len(va)}, test {len(te)} | "
          f"sources {list(sources)} | seeds {C.CLF_SEEDS}")

    rows = []
    for src, S in sources.items():
        for clf in args.classifiers:
            fpath = os.path.join(odir, f"{src}_{clf}.json")
            res = U.load_json(fpath) if os.path.exists(fpath) else None
            if not args.force and CL.result_is_current(res, clf):
                print(f"  {src:8s} {clf:10s} (cached) test bal_acc {res['test_bal_acc_mean']:.3f} "
                      f"± {res['test_bal_acc_std']:.3f}, AUC {res['test_auc_mean']:.3f}")
            else:
                t = time.time()
                circ = CL.build(clf, K)
                res = CL.train_seeds(circ, K, Xdummy[tr], y[tr], Xdummy[va], y[va],
                                     Xdummy[te], y[te], C.CLF_SEEDS,
                                     init_tr=S[tr], init_va=S[va], init_te=S[te])
                res.update({"dataset": dataset, "mode": mode, "source": src, "classifier": clf,
                            "spec_version": CL.spec_version(clf),
                            "n_params": circ.n_params, "readout_dim": int(circ.readout.shape[0]),
                            "test_split_idx": np.asarray(data["eval_idx"]).tolist(),
                            "seconds": round(time.time() - t, 1)})
                for r in res["runs"]:
                    r["model"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                  for k, v in r["model"].items()}
                U.save_json(res, fpath)
                print(f"  {src:8s} {clf:10s} test bal_acc {res['test_bal_acc_mean']:.3f} "
                      f"± {res['test_bal_acc_std']:.3f}, AUC {res['test_auc_mean']:.3f} "
                      f"± {res['test_auc_std']:.3f}  ({res['seconds']:.0f}s)")
            rows.append({"dataset": dataset, "mode": mode, "source": src, "classifier": clf,
                         "chance": 1.0 / K, "n_params": res["n_params"],
                         **{k: res[k] for k in ("test_bal_acc_mean", "test_bal_acc_std",
                                                "test_auc_mean", "test_auc_std")}})

    kernel_ref = {}
    d = load_diag_a(args.states_root, dataset, mode)
    if d:
        ro = d["readouts"]
        kernel_ref = {"bal_acc": {"exact": ro["target_fidkernel_svm"]["bal_acc"],
                                  "searched": ro["encoded_fidkernel_svm"]["bal_acc"]},
                      "auc": {"exact": ro["target_fidkernel_svm"]["auc"],
                              "searched": ro["encoded_fidkernel_svm"]["auc"]}}
        for r in rows:
            r["kernel_svm_exact_balacc"] = kernel_ref["bal_acc"]["exact"]
            r["kernel_svm_searched_balacc"] = kernel_ref["bal_acc"]["searched"]
    U.write_csv(rows, os.path.join(odir, "summary.csv"))
    plot(odir, rows, tag, kernel_ref)
    return rows


def main():
    """Command line: runs, state folders, classifiers and training settings."""
    p = argparse.ArgumentParser(description="classifiers on searched amplitude encodings")
    p.add_argument("--runs", nargs="*", default=DEFAULT_RUNS, help="dataset:mode pairs")
    p.add_argument("--states-root", default="experiments_searched")
    p.add_argument("--compare-root", default="experiments_handref",
                   help="diag_a output made with the default circuit ('' to skip)")
    p.add_argument("--classifiers", nargs="*", default=["hea", "hea8", "hea12", "hea8_probs", "qcnn"],
                   help="hea, qcnn, hea<L> (L layers), and *_probs variants, e.g. hea8_probs")
    p.add_argument("--sources", nargs="*", default=["exact", "searched", "default"],
                   choices=["exact", "searched", "default"])
    p.add_argument("--epochs", type=int, default=None, help="override config.VQC_EPOCHS")
    p.add_argument("--patience", type=int, default=None, help="override config.VQC_PATIENCE")
    p.add_argument("--force", action="store_true")
    p.add_argument("--quick", action="store_true", help="few epochs, one seed")
    a = p.parse_args()
    a.out_name = "downstream_classifiers"
    if a.epochs:
        C.VQC_EPOCHS = a.epochs
    if a.patience:
        C.VQC_PATIENCE = a.patience
    if a.quick:
        C.VQC_EPOCHS, C.CLF_SEEDS = 3, [0]
        a.out_name = "downstream_classifiers_quick"      # never mixed with real results
    if a.compare_root and not os.path.isdir(a.compare_root):
        print(f"note: --compare-root {a.compare_root} not found, 'default' source skipped")
        a.compare_root = ""
    for spec in a.runs:
        ds, mode = spec.split(":")
        run(a, ds, mode)
    rows = []
    import csv
    for f in sorted(glob.glob(os.path.join(a.states_root, "*", a.out_name, "*", "summary.csv"))):
        with open(f) as fh:
            rows += list(csv.DictReader(fh))
    out = os.path.join(a.states_root, f"summary_{a.out_name}.csv")
    U.write_csv(rows, out)
    print(f"\nsummary: {out}")


if __name__ == "__main__":
    main()
