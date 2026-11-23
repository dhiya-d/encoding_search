#!/usr/bin/env python3
# ----------------------------------------------------------------
# diag_b_ceiling.py  —  Diagnostic (b): what is the best ANY
# approximate-amplitude-encoding circuit could possibly deliver?
#
# An AAE circuit can at most reproduce the exact state |t>. Any quantum
# model that reads the state through measurement is linear in
# rho = |t><t|, so it only ever sees phase-invariant quantities. The
# natural ceiling is therefore an SVM on the exact fidelity kernel
# K(t, t') = |<t|t'>|^2 — no circuit search can beat it, only approach it.
#
# Readouts per (dataset, mode), all on the same class-balanced sample:
#   amp_linear          linear probe on the 256 amplitudes (the repo's
#                       "raw" column; NOT physically readable, not
#                       phase-invariant — upper reference only)
#   amp_rbf_svm         classical RBF SVM on the same unit vectors
#   zbasis_linear       linear probe on |t|^2 (what computational-basis
#                       sampling gives you)
#   fidkernel_svm       SVM on |<t|t'>|^2  <- the physical ceiling
# plus kernel-target alignment (repo definition, label-permutation null)
# for the fidelity kernel and an RBF reference kernel, and kernel
# concentration statistics (uncentered MedMNIST states overlap ~0.9, a
# nearly constant kernel carries little usable structure).
#
# Classical references computed once per dataset (mode-independent):
#   pixels_native_linear   grayscale at native resolution
#   pixels16_linear        16x16 grayscale, unnormalised
#   pixels16_rbf_svm       16x16 grayscale, RBF SVM
#   rgb_native_linear      RGB at native resolution        (RGB datasets only)
#   rgb16_linear           16x16 RGB, 768 features         (RGB datasets only)
#   rgb16_rbf_svm          16x16 RGB, RBF SVM              (RGB datasets only)
# These separate "information lost by grayscale", "... by downsampling"
# and "... by amplitude encoding". RGB rows use each channel resized with
# the same interpolation as the grayscale pipeline.
#
# Re-running WITHOUT --force on an earlier output folder only computes
# references that are missing (e.g. the RGB ones) and refreshes the
# summary rows and verdicts of finished modes; kernels are not redone.
#
# Usage (from the repository root):
#   python diagnostics/diag_b_ceiling.py
#   python diagnostics/diag_b_ceiling.py --datasets breastmnist --modes centered whitened
#
# Output: experiments/<dataset>/b_ceiling/
#   classical_reference/results.json
#   <mode>/results.json, summary_row.json, per_class_recall.csv,
#   <mode>/readouts.png, kernel_heatmap.png, kernel_value_hist.png
# ----------------------------------------------------------------

# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import os
import time

import numpy as np

import diag_common as D

DIAG = "b_ceiling"


def rbf_kernel(A, B, gamma):
    """Gaussian kernel exp(-gamma |a - b|^2)."""
    sq = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T
    return np.exp(-gamma * np.maximum(sq, 0.0))


REF_KEYS = ("pixels_native_linear", "pixels16_linear", "pixels16_rbf_svm",
            "rgb_native_linear", "rgb16_linear", "rgb16_rbf_svm")
MAX_NATIVE_FEATURES = 8192


def classical_reference(args, dataset, tr_x, tr_y, ev_x, ev_y, tr_vecs, ev_vecs):
    """Mode-independent classical references. Incremental: an existing
    results.json is kept and only missing references are computed (so the
    RGB references can be added to an earlier run without --force)."""
    odir = D.out_dir(args.out, dataset, DIAG, "classical_reference")
    path = os.path.join(odir, "results.json")
    res = D.load_json(path) if (os.path.exists(path) and not args.force) else {"dataset": dataset}
    missing = [k for k in REF_KEYS if k not in res]
    if not missing:
        return res
    print(f"[{dataset}] classical references: computing {missing}")
    rng = np.random.default_rng(args.seed)          # same sample as the modes
    tr_idx = D.stratified_sample(tr_y, args.n_train, rng)
    ev_idx = D.stratified_sample(ev_y, args.n_test, rng)
    ytr, yev = tr_y[tr_idx], ev_y[ev_idx]
    rgb = D.is_rgb(tr_x)
    no_rgb = {"skipped": f"images are not RGB (shape {tr_x.shape[1:]})"}

    if "pixels_native_linear" in missing:
        g_tr, g_ev = D.to_grayscale(tr_x[tr_idx]), D.to_grayscale(ev_x[ev_idx])
        if g_tr[0].size <= MAX_NATIVE_FEATURES:
            res["pixels_native_linear"] = D.linear_probe(
                g_tr.reshape(len(g_tr), -1), ytr, g_ev.reshape(len(g_ev), -1), yev)
            res["pixels_native_linear"]["shape"] = list(g_tr.shape[1:])
        else:
            res["pixels_native_linear"] = {"skipped": f"native size {g_tr.shape[1:]} too large"}
    if "pixels16_linear" in missing:
        res["pixels16_linear"] = D.linear_probe(tr_vecs[tr_idx], ytr, ev_vecs[ev_idx], yev)
    if "pixels16_rbf_svm" in missing:
        res["pixels16_rbf_svm"] = D.rbf_svm(tr_vecs[tr_idx], ytr, ev_vecs[ev_idx], yev)

    if "rgb_native_linear" in missing:
        if not rgb:
            res["rgb_native_linear"] = no_rgb
        else:
            r_tr, r_ev = D.to_rgb_float(tr_x[tr_idx]), D.to_rgb_float(ev_x[ev_idx])
            if r_tr[0].size <= MAX_NATIVE_FEATURES:
                res["rgb_native_linear"] = D.linear_probe(
                    r_tr.reshape(len(r_tr), -1), ytr, r_ev.reshape(len(r_ev), -1), yev)
                res["rgb_native_linear"]["shape"] = list(r_tr.shape[1:])
            else:
                res["rgb_native_linear"] = {"skipped": f"native size {r_tr.shape[1:]} too large"}
    if "rgb16_linear" in missing or "rgb16_rbf_svm" in missing:
        if not rgb:
            res.setdefault("rgb16_linear", no_rgb)
            res.setdefault("rgb16_rbf_svm", no_rgb)
        else:
            c_tr = D.images_to_rgb_vectors(tr_x[tr_idx])
            c_ev = D.images_to_rgb_vectors(ev_x[ev_idx])
            if "rgb16_linear" in missing:
                res["rgb16_linear"] = D.linear_probe(c_tr, ytr, c_ev, yev)
            if "rgb16_rbf_svm" in missing:
                res["rgb16_rbf_svm"] = D.rbf_svm(c_tr, ytr, c_ev, yev)

    D.save_json(res, path)
    for k in REF_KEYS:
        if "bal_acc" in res[k]:
            print(f"  {k:22s} bal_acc {res[k]['bal_acc']:.3f}  auc {res[k]['auc']:.3f}")
        else:
            print(f"  {k:22s} {res[k]['skipped']}")
    return res


def ref_columns(classical):
    """Classical reference scores as flat summary columns (ref_<model>_balacc/auc)."""
    cols = {}
    for k in REF_KEYS:
        if "bal_acc" in classical.get(k, {}):
            cols[f"ref_{k}_balacc"] = classical[k]["bal_acc"]
            cols[f"ref_{k}_auc"] = classical[k]["auc"]
    return cols


def verdict(r, classical):
    """Plain-language reading: is the exact-state ceiling at chance, below or close to the
    classical references?
    """
    out = []
    fk = r["readouts"]["fidkernel_svm"]
    chance = fk["chance"]
    ref = classical["pixels16_rbf_svm"]
    if fk["bal_acc"] < chance + 0.05 or (np.isfinite(fk["auc"]) and fk["auc"] < 0.6):
        out.append(f"CEILING AT CHANCE: even the exact fidelity kernel reaches only "
                   f"{fk['bal_acc']:.3f} bal_acc (chance {chance:.3f}). In this mode amplitude "
                   f"encoding itself is the bottleneck — no circuit search can fix it.")
    else:
        gap = ref["bal_acc"] - fk["bal_acc"]
        out.append(f"Physical ceiling {fk['bal_acc']:.3f} bal_acc / AUC {fk['auc']:.3f} "
                   f"vs classical 16x16 RBF {ref['bal_acc']:.3f} (gap {gap:+.3f}). "
                   f"A good AAE circuit should approach {fk['bal_acc']:.3f}.")
    if len(r["train_class_counts"]) == 2 and r["mode"] != "uncentered":
        out.append("BINARY + CENTERING: subtracting the train mean makes the two class means "
                   "exactly antiparallel (mu0 - m = (1-p)(mu0 - mu1), mu1 - m = -p(mu0 - mu1)). "
                   "|<t|t'>|^2 cannot tell t from -t, so the class-mean signal is invisible to "
                   "any measurement-based model in this mode. Compare with the uncentered row.")
    ks = r["kernel_stats"]
    if ks["offdiag_std"] < 0.03:
        out.append(f"Kernel is concentrated: off-diagonal mean {ks['offdiag_mean']:.3f}, "
                   f"std {ks['offdiag_std']:.3f}.")
    kt = r["kta"]["fidelity_kernel"]
    out.append(f"Fidelity-kernel KTA above null {kt['kta_above_null']:+.4f} "
               f"(z = {kt['z_score']:.1f}); RBF reference "
               f"{r['kta']['rbf_reference']['kta_above_null']:+.4f}.")
    if "bal_acc" in classical.get("rgb16_rbf_svm", {}):
        d = classical["rgb16_rbf_svm"]["bal_acc"] - classical["pixels16_rbf_svm"]["bal_acc"]
        da = classical["rgb16_rbf_svm"]["auc"] - classical["pixels16_rbf_svm"]["auc"]
        if d > 0.03:
            out.append(f"COLOUR MATTERS: 16x16 RGB RBF beats 16x16 grayscale RBF by {d:+.3f} bal_acc "
                       f"({da:+.3f} AUC). Grayscale conversion discards class signal before any "
                       f"quantum step; the ceiling above is relative to grayscale.")
        else:
            out.append(f"Colour adds little at 16x16 (RGB vs grayscale RBF: {d:+.3f} bal_acc, "
                       f"{da:+.3f} AUC); grayscale conversion is not the bottleneck.")
    out.append("Heuristic thresholds — read the numbers, not just this text.")
    return out


def make_figures(odir, readouts, classical, K_sub, y_sub, K_full, y_full, tag):
    """Bar chart of the exact-state readouts against the classical references, and the
    fidelity kernel sorted by class.
    """
    plt = D.get_plt()
    entries = [(f"classical: {k}", classical[k]) for k in REF_KEYS
               if "bal_acc" in classical.get(k, {})]
    entries += [(k, v) for k, v in readouts.items()]
    names = [e[0] for e in entries]
    colors = ["#7f7f7f"] * (len(entries) - len(readouts)) + ["#4f81bd", "#9bbb59", "#f79646", "#c0504d"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ("bal_acc", "auc")):
        vals = [e[1][key] for e in entries]
        ax.barh(names, vals, color=colors)
        ax.axvline(readouts["fidkernel_svm"]["chance"] if key == "bal_acc" else 0.5,
                   color="k", ls=":", label="chance")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.05); ax.set_title(key); ax.legend(fontsize=8)
    axes[1].set_yticklabels([])
    fig.suptitle(f"{tag}: ceiling of exact amplitude encoding (red = physical ceiling)")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "readouts.png"), dpi=150); plt.close(fig)

    order = np.argsort(y_sub, kind="stable")
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(K_sub[np.ix_(order, order)], cmap="viridis")
    bounds = np.flatnonzero(np.diff(y_sub[order])) + 0.5
    for b in bounds:
        ax.axhline(b, color="w", lw=0.6); ax.axvline(b, color="w", lw=0.6)
    fig.colorbar(im, ax=ax, label="|<t_i|t_j>|^2")
    ax.set_title(f"{tag}: fidelity kernel, sorted by class")
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(); fig.savefig(os.path.join(odir, "kernel_heatmap.png"), dpi=150); plt.close(fig)

    same = y_full[:, None] == y_full[None, :]
    off = ~np.eye(len(y_full), dtype=bool)
    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.linspace(K_full[off].min(), K_full[off].max(), 60)
    ax.hist(K_full[same & off], bins=bins, density=True, alpha=0.6, label="same class")
    ax.hist(K_full[~same], bins=bins, density=True, alpha=0.6, label="different class")
    ax.set_xlabel("|<t_i|t_j>|^2"); ax.set_ylabel("density"); ax.legend()
    ax.set_title(f"{tag}: kernel values within vs between classes")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "kernel_value_hist.png"), dpi=150); plt.close(fig)


def refresh_existing(odir, classical):
    """Update a finished combo with newly added classical references
    without recomputing its kernels."""
    result = D.load_json(os.path.join(odir, "results.json"))
    result["verdict"] = verdict(result, classical)
    D.save_json(result, os.path.join(odir, "results.json"))
    rp = os.path.join(odir, "summary_row.json")
    row = D.load_json(rp)
    old = {k: v for k, v in row.items() if not k.startswith("ref_")}
    head = {k: old.pop(k) for k in ("dataset", "mode", "chance") if k in old}
    D.save_json({**head, **ref_columns(classical), **old}, rp)
    for line in result["verdict"]:
        if "COLOUR" in line or "Colour" in line:
            print("  > " + line)


def run_combo(args, dataset, mode, data, classical):
    """Diagnostic (b) for one (dataset, mode): readouts on the EXACT amplitude-encoded
    states (perfect circuit), kernel-target alignment with a permutation null.
    """
    odir = D.out_dir(args.out, dataset, DIAG, mode)
    rpath = os.path.join(odir, "results.json")
    if os.path.exists(rpath) and not args.force:
        refresh_existing(odir, classical)
        print(f"[{dataset}/{mode}] results exist, refreshed reference columns and verdict "
              f"(use --force to recompute; figures keep the old reference bars)")
        return
    tag = f"{dataset}/{mode}"
    tr_vecs, tr_y, ev_vecs, ev_y, ev_name = data
    t0 = time.time()
    print(f"\n[{tag}] building exact amplitude states...")
    mean, std = D.compute_stats(tr_vecs, mode)
    rng = np.random.default_rng(args.seed)          # same sample as diag (a)
    tr_idx = D.stratified_sample(tr_y, args.n_train, rng)
    ev_idx = D.stratified_sample(ev_y, args.n_test, rng)
    T_tr, y_tr = D.apply_stats(tr_vecs[tr_idx], mean, std), tr_y[tr_idx]
    T_ev, y_ev = D.apply_stats(ev_vecs[ev_idx], mean, std), ev_y[ev_idx]

    K_tr = D.fidelity_kernel(T_tr, T_tr)
    K_ev = D.fidelity_kernel(T_ev, T_tr)
    print("  readouts...")
    readouts = {
        "amp_linear": D.linear_probe(T_tr, y_tr, T_ev, y_ev),
        "amp_rbf_svm": D.rbf_svm(T_tr, y_tr, T_ev, y_ev),
        "zbasis_linear": D.linear_probe(T_tr ** 2, y_tr, T_ev ** 2, y_ev),
        "fidkernel_svm": D.kernel_svm(K_tr, y_tr, K_ev, y_ev),
    }

    print(f"  KTA with {args.n_perm} label permutations...")
    krng = np.random.default_rng(args.seed + 5)
    k_idx = D.stratified_sample(y_tr, args.kta_n, krng)
    K_k, y_k = K_tr[np.ix_(k_idx, k_idx)], y_tr[k_idx]
    Tk = T_tr[k_idx]
    sq = (Tk ** 2).sum(1)[:, None] + (Tk ** 2).sum(1)[None, :] - 2 * Tk @ Tk.T
    gamma = 1.0 / np.median(sq[~np.eye(len(Tk), dtype=bool)])
    kta = {"fidelity_kernel": D.kta_with_null(K_k, y_k, args.n_perm, krng),
           "rbf_reference": D.kta_with_null(rbf_kernel(Tk, Tk, gamma), y_k, args.n_perm, krng)}
    kta["rbf_reference"]["gamma"] = float(gamma)

    result = {
        "dataset": dataset, "mode": mode, "eval_split": ev_name,
        "n_train": int(len(T_tr)), "n_eval": int(len(T_ev)), "kta_n": int(len(k_idx)),
        "train_class_counts": D.class_counts(y_tr), "eval_class_counts": D.class_counts(y_ev),
        "readouts": readouts, "kta": kta, "kernel_stats": D.kernel_stats(K_tr),
        "runtime_s": round(time.time() - t0, 1),
    }
    result["verdict"] = verdict(result, classical)

    row = {"dataset": dataset, "mode": mode, "chance": readouts["fidkernel_svm"]["chance"],
           **ref_columns(classical)}
    for k, v in readouts.items():
        row[f"{k}_balacc"] = v["bal_acc"]
        row[f"{k}_auc"] = v["auc"]
    row["fid_kta_above_null"] = kta["fidelity_kernel"]["kta_above_null"]
    row["fid_kta_z"] = kta["fidelity_kernel"]["z_score"]
    row["rbf_kta_above_null"] = kta["rbf_reference"]["kta_above_null"]
    row["kernel_offdiag_mean"] = result["kernel_stats"]["offdiag_mean"]
    row["kernel_offdiag_std"] = result["kernel_stats"]["offdiag_std"]

    D.save_json(result, os.path.join(odir, "results.json"))
    D.save_json(row, os.path.join(odir, "summary_row.json"))
    classes = sorted(readouts["amp_linear"]["recall"])
    D.write_csv([{"class": c, "support_eval": int((y_ev == c).sum()),
                  **{k: v["recall"].get(c) for k, v in readouts.items()}} for c in classes],
                os.path.join(odir, "per_class_recall.csv"))
    sub = D.stratified_sample(y_k, min(300, len(y_k)), np.random.default_rng(1))
    make_figures(odir, readouts, classical, K_k[np.ix_(sub, sub)], y_k[sub], K_k, y_k, tag)

    for k, v in readouts.items():
        print(f"  {k:16s} bal_acc {v['bal_acc']:.3f}  auc {v['auc']:.3f}")
    for line in result["verdict"]:
        print("  > " + line)
    print(f"  saved to {odir}  ({result['runtime_s']} s)")


def main():
    """Command line: sample sizes and KTA settings; runs every dataset/mode."""
    p = D.add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--kta-n", type=int, default=1000, help="train subset used for KTA")
    p.add_argument("--n-perm", type=int, default=200, help="label permutations for the KTA null")
    args = p.parse_args()

    datasets = D.resolve_datasets(args.data_root, args.datasets)
    print(f"datasets: {datasets} | modes: {args.modes}")
    for ds in datasets:
        tr_x, tr_y, ev_x, ev_y, ev_name = D.load_train_eval(args.data_root, ds)
        tr_vecs, ev_vecs = D.images_to_vectors(tr_x), D.images_to_vectors(ev_x)
        classical = classical_reference(args, ds, tr_x, tr_y, ev_x, ev_y, tr_vecs, ev_vecs)
        for mode in args.modes:
            run_combo(args, ds, mode, (tr_vecs, tr_y, ev_vecs, ev_y, ev_name), classical)
            D.collect_summaries(args.out, DIAG)
    print(f"\nsummaries: {args.out}/<dataset>/summary_{DIAG}.csv and {args.out}/summary_{DIAG}.csv")


if __name__ == "__main__":
    main()
