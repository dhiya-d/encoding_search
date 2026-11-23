#!/usr/bin/env python3
# ----------------------------------------------------------------
# diag_c_spectral.py  —  Diagnostic (c): where does the class
# information live, and can a fidelity-driven encoder afford to lose it?
#
# With amplitude encoding the state IS the 16x16 image viewed as a
# function on Z_16 x Z_16 (wires 0-3 = row, 4-7 = column). Two group
# Fourier transforms are natural here:
#   * the 2D discrete Fourier transform (QFT on the row register tensor
#     QFT on the column register) — spatial frequencies;
#   * the Walsh-Hadamard transform (H on every qubit, the Z_2^8
#     transform of Belis et al. Sec. II) — parities of bit subsets.
# Both use the orthonormal convention, so each unit-norm state has total
# energy 1 in every basis (Plancherel), and fidelity is basis-invariant.
#
# Energy decomposition (class-balanced weights, so every class counts
# equally, matching balanced accuracy). For every frequency k:
#   total_k   = mean over classes of E|x_k|^2
#   shared_k  = |mu_k|^2                       (mean of class means)
#   between_k = mean_c |mu_{c,k} - mu_k|^2     (class-mean differences)
#   within_k  = mean_c E|x_k - mu_{c,k}|^2     (within-class spread)
# and total = shared + between + within exactly, summing to 1 over k.
#
# The key number is B = sum_k between_k: the share of each state's norm
# that carries the class-MEAN differences a linear or kernel readout
# relies on. An encoder at fidelity F throws away up to 1 - F of the
# norm; if B < 1 - F it can match the shared + within components and
# lose all of the class-mean signal while still scoring high fidelity.
#
# Also reported:
#   Fisher score per frequency (between_k / within_k), on the complex
#   coefficients and on their magnitudes (shift-insensitive version);
#   radial profile (low vs mid vs high spatial frequency);
#   Walsh profile by Hamming weight (the "order" of the parity);
#   "greedy retention" curves: keep coefficients in order of decreasing
#   energy — an idealised fidelity-greedy compressor — and track how
#   much of B survives as kept energy rises. Done in the Fourier, Walsh
#   and pixel bases, plus a strict low-pass (by radius) curve.
#
# If diag_a_phase_check.py has already run, its measured fidelities are
# pulled in automatically and compared against B. Pass --genome-root to
# also compare against the search's holdout fidelities.
#
# Usage (from the repository root):
#   python diagnostics/diag_c_spectral.py
#   python diagnostics/diag_c_spectral.py --datasets retinamnist --max-n 5000
#
# Output: experiments/<dataset>/c_spectral/
#   decomposition_by_mode.png                     (all modes side by side)
#   <mode>/results.json, summary_row.json, spectra.npz,
#   <mode>/class_means.png, fourier_maps.png, radial_profile.png,
#   <mode>/retention_curves.png, walsh_order.png
# ----------------------------------------------------------------

# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import os

import numpy as np
from scipy.linalg import hadamard

import diag_common as D

DIAG = "c_spectral"
SIDE = D.IMG_SIDE
EPS_LEVELS = (0.01, 0.02, 0.03, 0.05, 0.10, 0.20)


def fourier2(T):
    """2-D unitary DFT of every 16x16 target (rows of T)."""
    return np.fft.fft2(T.reshape(-1, SIDE, SIDE), norm="ortho").reshape(len(T), -1)


def walsh(T):
    """Walsh-Hadamard transform of every 256-dimensional target (rows of T)."""
    H = hadamard(D.STATE_DIM) / np.sqrt(D.STATE_DIM)   # H[i,j] = (-1)^{popcount(i&j)}/16
    return T @ H


def decompose(X, y):
    """Per-coefficient energy split into shared (|class-average mean|^2), between-class and
    within-class parts, plus the Fisher ratio between/within.
    """
    classes = np.unique(y)
    mu_c = np.stack([X[y == c].mean(0) for c in classes])
    mu = mu_c.mean(0)
    shared = np.abs(mu) ** 2
    between = (np.abs(mu_c - mu[None]) ** 2).mean(0)
    within = np.stack([(np.abs(X[y == c] - mu_c[i]) ** 2).mean(0)
                       for i, c in enumerate(classes)]).mean(0)
    total = np.stack([(np.abs(X[y == c]) ** 2).mean(0) for c in classes]).mean(0)
    return {"total": total, "shared": shared, "between": between, "within": within,
            "fisher": between / (within + 1e-12), "class_means": mu_c}


def magnitude_fisher(X, y):
    """Fisher ratio of the coefficient magnitudes (phase-blind)."""
    d = decompose(np.abs(X), y)
    return d["fisher"]


def greedy_curve(dec, order=None):
    """Keep coefficients in order of decreasing energy (or a given order); returns the
    order and the cumulative fractions of total, between-class and shared energy.
    """
    if order is None:
        order = np.argsort(-dec["total"], kind="stable")
    cum_E = np.cumsum(dec["total"][order]) / dec["total"].sum()
    cum_B = np.cumsum(dec["between"][order]) / max(dec["between"].sum(), 1e-15)
    cum_S = np.cumsum(dec["shared"][order]) / max(dec["shared"].sum(), 1e-15)
    return order, cum_E, cum_B, cum_S


def retention_table(cum_E, cum_B):
    """For each infidelity level eps: number of coefficients needed to keep 1 - eps of the
    energy, and the fraction of between-class energy they retain.
    """
    rows = {}
    for eps in EPS_LEVELS:
        m = int(np.searchsorted(cum_E, 1 - eps - 1e-12) + 1)
        m = min(m, len(cum_E))
        rows[f"{1 - eps:.2f}"] = {"n_coeffs": m, "between_retained": float(cum_B[m - 1])}
    return rows


def radial_bins():
    """Radial frequency |k| (rounded) of every 2-D DFT coefficient, and the axis frequencies."""
    ky = np.fft.fftfreq(SIDE) * SIDE
    kx = np.fft.fftfreq(SIDE) * SIDE
    r = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2).reshape(-1)
    return np.rint(r).astype(int), ky, kx


def load_measured_fidelity(args, dataset, mode):
    """Fidelities actually reached by the circuit (from diagnostic (a) and, if given, the
    search's held-out fidelity) for comparison.
    """
    out = {}
    p = os.path.join(args.out, dataset, "a_phase_check", mode, "results.json")
    if os.path.exists(p):
        r = D.load_json(p)
        out["diag_a_mean_fidelity"] = r["fidelity"]["mean"]
        out["diag_a_two_qubit_gates"] = r["circuit"]["n_two_qubit"]
    if args.genome_root:
        gp, _ = D.find_selected_genome_file(args.genome_root, dataset, mode)
        if gp:
            g = D.load_json(gp)
            if isinstance(g, dict) and g.get("holdout_fidelity") is not None:
                if mode == "uncentered" or bool(g.get("whitened", False)) == (mode == "whitened"):
                    out["search_holdout_fidelity"] = g["holdout_fidelity"]
    return out


def make_figures(odir, tag, dec_f, dec_w, dec_p, magF, curves, rbin, ky, kx, fids):
    """Class means, energy spectra (pixel, Fourier, Walsh), Fisher maps and the retention
    curves.
    """
    plt = D.get_plt()
    sh = lambda v: np.fft.fftshift(v.reshape(SIDE, SIDE))
    ext = [-8.5, 7.5, 7.5, -8.5]          # fftshifted: k from -8 to 7, DC at centre

    # class means in direct space
    cm = dec_p["class_means"].real
    k = len(cm)
    fig, axes = plt.subplots(1, k + 1, figsize=(1.8 * (k + 1), 2.2))
    vmax = np.abs(cm).max()
    axes[0].imshow((cm.mean(0)).reshape(SIDE, SIDE), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    axes[0].set_title("shared", fontsize=8)
    for i in range(k):
        axes[i + 1].imshow((cm[i] - cm.mean(0)).reshape(SIDE, SIDE), cmap="RdBu_r",
                           vmin=-np.abs(cm - cm.mean(0)).max(), vmax=np.abs(cm - cm.mean(0)).max())
        axes[i + 1].set_title(f"class {i} - shared", fontsize=8)
    for a in axes:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{tag}: class-mean amplitude images", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(odir, "class_means.png"), dpi=150); plt.close(fig)

    # Fourier maps
    panels = [("log10 total energy", np.log10(dec_f["total"] + 1e-12)),
              ("log10 shared energy", np.log10(dec_f["shared"] + 1e-12)),
              ("log10 between-class energy", np.log10(dec_f["between"] + 1e-12)),
              ("log10 within-class energy", np.log10(dec_f["within"] + 1e-12)),
              ("Fisher (complex coeff.)", dec_f["fisher"]),
              ("Fisher (|coeff.|)", magF)]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.5))
    for ax, (title, v) in zip(axes.ravel(), panels):
        im = ax.imshow(sh(v), cmap="magma", extent=ext)
        ax.set_title(title, fontsize=9); ax.set_xlabel("k_col"); ax.set_ylabel("k_row")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"{tag}: 2D Fourier spectrum of the amplitude states (centre = DC)")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "fourier_maps.png"), dpi=150); plt.close(fig)

    # radial profile
    rs = np.arange(rbin.max() + 1)
    agg = lambda v: np.array([v[rbin == r].sum() for r in rs])
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, lab in (("total", "total energy"), ("shared", "shared"), ("between", "between-class"),
                     ("within", "within-class")):
        ax.plot(rs, agg(dec_f[key]) / max(dec_f[key].sum(), 1e-15), marker="o", label=lab)
    ax.set_xlabel("radial spatial frequency |k|"); ax.set_ylabel("fraction of that component")
    ax2 = ax.twinx()
    ax2.plot(rs, [dec_f["fisher"][rbin == r].mean() for r in rs], "k--", label="mean Fisher")
    ax2.set_ylabel("mean Fisher score")
    ax.legend(loc="upper right", fontsize=8); ax2.legend(loc="center right", fontsize=8)
    ax.set_title(f"{tag}: radial profile")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "radial_profile.png"), dpi=150); plt.close(fig)

    # retention curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    styles = ["-", "--", "-", ":"]
    ax = axes[0]
    for (name, (cE, cB)), ls in zip(curves.items(), styles):
        ax.plot(np.concatenate([[0], cE]), np.concatenate([[0], cB]), ls, label=name)
    ax.plot([0, 1], [0, 1], "k:", lw=0.8, label="proportional")
    for lab, f in fids.items():
        ax.axvline(f, color="grey", ls="--", lw=0.8)
        ax.text(f, 0.02, f" {lab} F={f:.3f}", fontsize=7, rotation=90, va="bottom")
    ax.set_xlim(0, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of total energy kept (~ fidelity)")
    ax.set_ylabel("fraction of between-class energy kept")
    ax.legend(fontsize=8)
    ax = axes[1]
    for (name, (cE, cB)), ls in zip(curves.items(), styles):
        disc = np.clip(1 - np.concatenate([[0], cE]), 1e-7, 1)
        ax.step(disc, np.concatenate([[0], cB]), ls, where="post", label=name)
    for lab, f in fids.items():
        ax.axvline(max(1 - f, 1e-7), color="grey", ls="--", lw=0.8)
        ax.text(max(1 - f, 1e-7), 0.02, f" {lab} 1-F={1 - f:.3g}", fontsize=7, rotation=90, va="bottom")
    ax.set_xscale("log"); ax.invert_xaxis(); ax.set_ylim(0, 1.02)
    ax.set_xlabel("energy discarded, log scale (~ infidelity budget 1-F)")
    ax.set_title("same curves: how small must 1-F be to keep the class signal?")
    fig.suptitle(f"{tag}: what an energy-greedy compressor keeps of the class signal")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "retention_curves.png"), dpi=150); plt.close(fig)

    # Walsh by order
    k = np.arange(D.STATE_DIM)
    pc = lambda a: np.array([bin(int(v)).count("1") for v in a])
    wt, wr, wc = pc(k), pc(k >> 4), pc(k & 15)
    orders = np.arange(9)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    width = 0.27
    for j, (key, lab) in enumerate((("total", "total"), ("shared", "shared"), ("between", "between-class"))):
        vals = [dec_w[key][wt == o].sum() / max(dec_w[key].sum(), 1e-15) for o in orders]
        axes[0].bar(orders + (j - 1) * width, vals, width, label=lab)
    axes[0].set_xlabel("Hamming weight of Walsh index (parity order)")
    axes[0].set_ylabel("fraction of component"); axes[0].legend(fontsize=8)
    grid = np.zeros((5, 5))
    for a in range(5):
        for b in range(5):
            m = (wr == a) & (wc == b)
            grid[a, b] = dec_w["between"][m].sum() / max(dec_w["between"].sum(), 1e-15)
    im = axes[1].imshow(grid, cmap="magma", origin="lower")
    for a in range(5):
        for b in range(5):
            axes[1].text(b, a, f"{grid[a, b]:.2f}", ha="center", va="center", fontsize=7, color="w")
    axes[1].set_xlabel("column-register weight"); axes[1].set_ylabel("row-register weight")
    axes[1].set_title("between-class energy share")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.suptitle(f"{tag}: Walsh-Hadamard (Z_2^8) spectrum")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "walsh_order.png"), dpi=150); plt.close(fig)


def run_combo(args, dataset, mode, tr_vecs, tr_y):
    """Diagnostic (c) for one (dataset, mode): where the class signal sits in the state and
    how much of it survives a fidelity budget.
    """
    odir = D.out_dir(args.out, dataset, DIAG, mode)
    if os.path.exists(os.path.join(odir, "results.json")) and not args.force:
        print(f"[{dataset}/{mode}] results exist, skipping (use --force)")
        return D.load_json(os.path.join(odir, "results.json"))
    tag = f"{dataset}/{mode}"
    print(f"\n[{tag}] spectral decomposition...")
    mean, std = D.compute_stats(tr_vecs, mode)
    idx = np.arange(len(tr_y))
    if args.max_n and len(idx) > args.max_n:
        idx = D.stratified_sample(tr_y, args.max_n, np.random.default_rng(args.seed))
    T, y = D.apply_stats(tr_vecs[idx], mean, std), tr_y[idx]

    Xf, Xw = fourier2(T), walsh(T)
    dec_p, dec_f, dec_w = decompose(T, y), decompose(Xf, y), decompose(Xw, y)
    magF = magnitude_fisher(Xf, y)
    rbin, ky, kx = radial_bins()

    parts = {k: float(dec_f[k].sum()) for k in ("total", "shared", "between", "within")}
    check = max(abs(parts[k] - float(dec_p[k].sum())) for k in parts)
    parts["plancherel_check_max_abs_diff"] = check

    curves, tables = {}, {}
    for name, dec in (("Fourier greedy", dec_f), ("Walsh greedy", dec_w), ("pixel greedy", dec_p)):
        _, cE, cB, _ = greedy_curve(dec)
        curves[name] = (cE, cB)
        tables[name] = retention_table(cE, cB)
    lp_order = np.lexsort((-dec_f["total"], rbin))
    _, cE, cB, _ = greedy_curve(dec_f, lp_order)
    curves["Fourier low-pass (by radius)"] = (cE, cB)
    tables["Fourier low-pass (by radius)"] = retention_table(cE, cB)

    radial = {}
    for r in range(rbin.max() + 1):
        m = rbin == r
        radial[r] = {"n_freqs": int(m.sum()),
                     **{k: float(dec_f[k][m].sum()) for k in ("total", "shared", "between", "within")},
                     "mean_fisher": float(dec_f["fisher"][m].mean())}
    k = np.arange(D.STATE_DIM)
    wt = np.array([bin(int(v)).count("1") for v in k])
    walsh_orders = {int(o): {kk: float(dec_w[kk][wt == o].sum())
                             for kk in ("total", "shared", "between", "within")}
                    for o in range(9)}
    top = np.argsort(-dec_f["fisher"])[:15]
    ky_full = np.repeat(ky, SIDE).astype(int)
    kx_full = np.tile(kx, SIDE).astype(int)
    top_list = [{"k_row": int(ky_full[i]), "k_col": int(kx_full[i]), "radius": int(rbin[i]),
                 "fisher": float(dec_f["fisher"][i]), "energy": float(dec_f["total"][i]),
                 "between": float(dec_f["between"][i])} for i in top]

    B = parts["between"]
    # Born-rule sign ambiguity: rho(t) = rho(-t), so class means pointing in
    # opposite directions are indistinguishable to a measurement-based model.
    cm = dec_p["class_means"]
    cmn = cm / np.maximum(np.linalg.norm(cm, axis=1, keepdims=True), 1e-15)
    cos = cmn @ cmn.T
    iu = np.triu_indices(len(cm), 1)
    min_cos = float(cos[iu].min()) if len(iu[0]) else float("nan")
    fids = load_measured_fidelity(args, dataset, mode)
    fid_compare = {}
    for lab, F in fids.items():
        if "fidelity" in lab:
            infid = 1 - F
            fid_compare[lab] = {"fidelity": F, "infidelity": infid,
                                "between_over_infidelity": B / infid if infid > 0 else float("inf"),
                                "between_could_be_entirely_lost": bool(B < infid)}

    lines = [f"Energy split: shared {parts['shared']:.3f}, between-class {B:.4f}, "
             f"within-class {parts['within']:.3f}."]
    r97 = tables["Fourier greedy"]["0.97"]
    lines.append(f"Energy-greedy Fourier compressor at 97% energy keeps {r97['n_coeffs']} of 256 "
                 f"coefficients and {r97['between_retained']:.2f} of the between-class energy.")
    low = sum(radial[r]["between"] for r in radial if r <= 2) / max(B, 1e-15)
    lines.append(f"{low:.2f} of between-class energy sits at radius <= 2 (low frequency); "
                 f"{1 - low:.2f} above.")
    if min_cos < -0.5:
        lines.append(f"Some class means are nearly antiparallel (min cosine {min_cos:.2f}); since "
                     f"|<t|t'>|^2 cannot tell t from -t, that part of the class signal is invisible "
                     f"to measurement-based models" + (" (always exactly -1 for two classes after "
                     f"centering)" if len(cm) == 2 else "") + ".")
    for lab, c in fid_compare.items():
        lines.append(f"{lab}: 1-F = {c['infidelity']:.4f} vs B = {B:.4f} "
                     f"({'B < 1-F: the class-mean signal can be lost entirely at this fidelity' if c['between_could_be_entirely_lost'] else 'B > 1-F: this fidelity cannot erase all of it'}).")

    result = {"dataset": dataset, "mode": mode, "n_images": int(len(y)),
              "class_counts": D.class_counts(y), "energy": parts,
              "fisher_total_between_over_within": B / max(parts["within"], 1e-15),
              "class_mean_cosines": cos, "min_class_mean_cosine": min_cos,
              "retention": tables, "radial_profile": radial, "walsh_by_order": walsh_orders,
              "top_fisher_frequencies": top_list, "fidelity_comparison": fid_compare,
              "summary": lines}

    row = {"dataset": dataset, "mode": mode, "n_images": int(len(y)),
           "shared": parts["shared"], "between": B, "within": parts["within"],
           "between_frac_radius_le2": low, "min_class_mean_cosine": min_cos}
    for name, key in (("Fourier greedy", "fourier"), ("Walsh greedy", "walsh"),
                      ("pixel greedy", "pixel"), ("Fourier low-pass (by radius)", "lowpass")):
        row[f"{key}_ncoef_97"] = tables[name]["0.97"]["n_coeffs"]
        row[f"{key}_between_kept_97"] = tables[name]["0.97"]["between_retained"]
        row[f"{key}_between_kept_90"] = tables[name]["0.90"]["between_retained"]
    for lab, c in fid_compare.items():
        row[f"{lab}"] = c["fidelity"]
        row[f"{lab}_B_over_infid"] = c["between_over_infidelity"]

    D.save_json(result, os.path.join(odir, "results.json"))
    D.save_json(row, os.path.join(odir, "summary_row.json"))
    np.savez_compressed(os.path.join(odir, "spectra.npz"),
                        **{f"fourier_{k}": dec_f[k] for k in ("total", "shared", "between", "within", "fisher")},
                        **{f"walsh_{k}": dec_w[k] for k in ("total", "shared", "between", "within", "fisher")},
                        **{f"pixel_{k}": dec_p[k] for k in ("total", "shared", "between", "within", "fisher")},
                        fourier_mag_fisher=magF, fourier_class_means=dec_f["class_means"],
                        pixel_class_means=dec_p["class_means"], radius=rbin, k_row=ky_full, k_col=kx_full)
    make_figures(odir, tag, dec_f, dec_w, dec_p, magF, curves, rbin, ky, kx,
                 {lab.replace("_mean_fidelity", "").replace("_holdout_fidelity", ""): c["fidelity"]
                  for lab, c in fid_compare.items()})
    for line in lines:
        print("  > " + line)
    if check > 1e-8:
        print(f"  WARNING: Plancherel check off by {check:.2e}")
    print(f"  saved to {odir}")
    return result


def decomposition_figure(args, dataset, results):
    """Shared / between / within energy fractions for all modes of one dataset."""
    plt = D.get_plt()
    modes = [m for m in D.MODES if m in results]
    if not modes:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    bottoms = np.zeros(len(modes))
    for key, col in (("shared", "#7f7f7f"), ("between", "#c0504d"), ("within", "#4f81bd")):
        vals = np.array([results[m]["energy"][key] for m in modes])
        axes[0].bar(modes, vals, bottom=bottoms, color=col, label=key)
        bottoms += vals
    axes[0].set_ylabel("fraction of state norm"); axes[0].legend(fontsize=8)
    axes[0].set_title("energy decomposition")
    B = [results[m]["energy"]["between"] for m in modes]
    axes[1].bar(modes, B, color="#c0504d", label="between-class B")
    for i, m in enumerate(modes):
        for lab, c in results[m]["fidelity_comparison"].items():
            axes[1].scatter(i, c["infidelity"], marker="x", color="k", zorder=3)
            short = lab.replace("_mean_fidelity", "").replace("_holdout_fidelity", "")
            axes[1].annotate(short, (i, c["infidelity"]), fontsize=7,
                             xytext=(4, 2), textcoords="offset points")
    axes[1].set_ylabel("fraction of state norm")
    axes[1].set_title("B vs measured infidelity 1-F (x)")
    axes[1].set_yscale("log"); axes[1].legend(fontsize=8)
    fig.suptitle(f"{dataset}: how much of each state carries class-mean signal")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, dataset, DIAG, "decomposition_by_mode.png"), dpi=150)
    plt.close(fig)


def main():
    """Command line: image cap and optional search folder; runs every dataset/mode."""
    p = D.add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--max-n", type=int, default=0,
                   help="cap images per dataset (0 = full train split)")
    p.add_argument("--genome-root", default=None,
                   help="search output folder (e.g. experiments), to compare against "
                        "the search's held-out fidelities")
    args = p.parse_args()

    D.check_genome_root(args.genome_root)
    datasets = D.resolve_datasets(args.data_root, args.datasets)
    print(f"datasets: {datasets} | modes: {args.modes}")
    for ds in datasets:
        tr_x, tr_y = D.load_split(args.data_root, ds, "train")
        tr_vecs = D.images_to_vectors(tr_x)
        results = {}
        for mode in args.modes:
            results[mode] = run_combo(args, ds, mode, tr_vecs, tr_y)
            D.collect_summaries(args.out, DIAG)
        for mode in D.MODES:          # include modes from earlier runs in the figure
            rp = os.path.join(args.out, ds, DIAG, mode, "results.json")
            if mode not in results and os.path.exists(rp):
                results[mode] = D.load_json(rp)
        decomposition_figure(args, ds, results)
    print(f"\nsummaries: {args.out}/<dataset>/summary_{DIAG}.csv and {args.out}/summary_{DIAG}.csv")


if __name__ == "__main__":
    main()
