#!/usr/bin/env python3
# ----------------------------------------------------------------
# diag_a_phase_check.py  —  Diagnostic (a): is the chance-level linear
# probe on encoded states a GLOBAL SIGN / PHASE artefact?
#
# Fidelity |<t|psi>|^2 cannot see a global phase, so per-image training
# from random initial angles lands on  psi ~ e^{-i phi} t  with an
# arbitrary phi (for real RY/CNOT/CZ circuits: psi ~ +t or -t). A linear
# probe that reads the raw amplitudes then sees randomly sign-flipped
# feature vectors, which can push it to chance regardless of fidelity.
# Physically this is irrelevant: expectation values and the fidelity
# kernel are phase-invariant.
#
# For each (dataset, preprocessing mode) this script:
#   1. loads the searched architecture (or the hand-designed reference),
#   2. re-encodes a class-balanced train/test sample with per-image
#      training (same recipe as the repo: random init, Adam, restarts),
#   3. measures the phase distribution of <psi|t> overall and per class,
#   4. runs six readouts on the SAME sample:
#        target_linear              linear probe, exact targets (reference)
#        encoded_linear_asis        linear probe, encoded states as produced
#                                   (reproduces the earlier probe setup)
#        encoded_linear_aligned     same, after removing each state's
#                                   global phase relative to its target
#        control_target_randphase   exact targets multiplied by the
#                                   OBSERVED phases, shuffled across images
#                                   (isolates the phase effect, no infidelity)
#        encoded_fidkernel_svm      SVM on |<psi_i|psi_j>|^2 (phase-invariant,
#                                   the physically meaningful readout)
#        target_fidkernel_svm       same kernel on exact targets (ceiling)
#
# Reading the result:
#   aligned >> as-is and control ~ as-is   -> the earlier negative result
#                                             was (largely) a phase artefact
#   aligned ~ as-is ~ chance, but encoded_fidkernel ~ target_fidkernel
#                                          -> the linear probe is the wrong
#                                             instrument; info survives
#   encoded_fidkernel << target_fidkernel  -> genuine encoding loss
#
# Usage (from the repository root):
#   python diagnostics/diag_a_phase_check.py
#   python diagnostics/diag_a_phase_check.py --out experiments_handref      # reference circuit
#   python diagnostics/diag_a_phase_check.py --genome-root experiments --out experiments_searched
#   python diagnostics/diag_a_phase_check.py --datasets bloodmnist --modes uncentered --genome-source kta
#   python diagnostics/diag_a_phase_check.py --genome-json path/to/selected_genome.json
#
# --genome-root accepts the search output folder (experiments/, layout
# <dataset>/search/<mode>/selected_genome.json) or the repo layout
# <root>/<dataset>_<uncentered|centered>/selected_genome[_kta].json. Datasets/modes without a searched genome
# fall back to DEFAULT_GENOME in aae_sim.py (the hand-designed reference);
# the source is recorded in every results.json.
#
# Output: experiments/<dataset>/a_phase_check/<mode>/
#   results.json, summary_row.json, per_class_recall.csv,
#   fidelity_hist.png, phase_by_class.png, readouts.png,
#   encoded_states.npz (targets, states, aligned states, params, labels)
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
from aae_sim import AAECircuit, DEFAULT_GENOME, fit_states, load_genome_file

DIAG = "a_phase_check"


def find_genome(args, dataset, mode):
    """Circuit to use for one (dataset, mode): --genome-json, else the selected genome
    under --genome-root, else the hand-designed reference (DEFAULT_GENOME). Returns
    (genome, info about its source).
    """
    if args.genome_json:
        genome, meta = load_genome_file(args.genome_json)
        return genome, {"source": os.path.abspath(args.genome_json),
                        "search_holdout_fidelity": meta.get("holdout_fidelity")}
    info = {"source": "DEFAULT_GENOME"}
    if args.genome_root:
        path, tried = D.find_selected_genome_file(args.genome_root, dataset, mode,
                                                  kta=args.genome_source == "kta")
        if path:
            genome, meta = load_genome_file(path)
            info = {"source": os.path.abspath(path),
                    "search_holdout_fidelity": meta.get("holdout_fidelity"),
                    "search_holdout_retention": meta.get("holdout_retention"),
                    "search_gate_count": meta.get("gate_count")}
            if mode != "uncentered" and "whitened" in meta:
                if bool(meta["whitened"]) != (mode == "whitened"):
                    info["warning"] = (f"genome was searched with whitened={meta['whitened']} "
                                       f"but is being evaluated in mode '{mode}'")
            return genome, info
        info["note"] = f"no selected genome found (tried {tried})"
        print(f"  NOTE: [{dataset}/{mode}] no selected genome found, using DEFAULT_GENOME")
    return DEFAULT_GENOME, info


def phase_stats(phases, labels, is_complex):
    """Distribution of the global phases arg<t|psi>: resultant length (1 = all equal, 0 =
    uniform), fraction of sign flips, and the same per class.
    """
    z = np.exp(1j * phases)
    out = {"resultant_length": float(np.abs(z.mean())),
           "frac_sign_flipped": float(np.mean(np.cos(phases) < 0))}
    per = {}
    for c in np.unique(labels):
        m = labels == c
        per[int(c)] = {"resultant_length": float(np.abs(z[m].mean())),
                       "frac_sign_flipped": float(np.mean(np.cos(phases[m]) < 0))}
    out["per_class"] = per
    out["note"] = ("resultant_length = |mean(exp(i*phase))|: 1 means every state has "
                   "the same global phase, 0 means phases are uniformly scattered"
                   + ("" if is_complex else "; real circuit, so phase is 0 or pi"))
    return out


def verdict(r):
    """Plain-language reading of the six readouts (phase artefact / wrong instrument /
    genuine encoding loss), following the rules in the header.
    """
    lines = []
    asis, ali = r["readouts"]["encoded_linear_asis"], r["readouts"]["encoded_linear_aligned"]
    ctrl = r["readouts"]["control_target_randphase"]
    tgt = r["readouts"]["target_linear"]
    ek, tk = r["readouts"]["encoded_fidkernel_svm"], r["readouts"]["target_fidkernel_svm"]
    R = r["phase"]["resultant_length"]
    gain = ali["bal_acc"] - asis["bal_acc"]
    if R < 0.9 and gain > 0.05:
        lines.append(f"PHASE ARTEFACT LIKELY: phases scattered (R={R:.2f}) and aligning "
                     f"them raises probe bal_acc by {gain:+.3f}.")
        if ctrl["bal_acc"] < tgt["bal_acc"] - 0.05:
            lines.append(f"Control confirms it: exact targets with the same shuffled phases "
                         f"drop from {tgt['bal_acc']:.3f} to {ctrl['bal_acc']:.3f}.")
    elif R >= 0.9:
        lines.append(f"Phases are consistent (R={R:.2f}); the as-is probe is not a phase artefact.")
    else:
        lines.append(f"Phases scattered (R={R:.2f}) but alignment changes bal_acc by only "
                     f"{gain:+.3f}; look at the kernel readouts.")
    lines.append(f"Phase-invariant readout: encoded {ek['bal_acc']:.3f} vs exact-target "
                 f"{tk['bal_acc']:.3f} bal_acc (AUC {ek['auc']:.3f} vs {tk['auc']:.3f}); "
                 f"physical encoding loss = {tk['bal_acc'] - ek['bal_acc']:+.3f} bal_acc.")
    lines.append("Heuristic thresholds (0.9, 0.05) — read the numbers, not just this text.")
    return lines


def make_figures(odir, fids, phases, labels, readouts, is_complex, tag):
    """Fidelity histogram, phase distribution per class and bar chart of the readouts."""
    plt = D.get_plt()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(fids, bins=40, color="#3b6ea5")
    ax.axvline(fids.mean(), color="k", ls="--", label=f"mean {fids.mean():.3f}")
    ax.set_xlabel("fidelity |<t|psi>|^2"); ax.set_ylabel("images"); ax.legend()
    ax.set_title(f"{tag}: per-image fidelity")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "fidelity_hist.png"), dpi=150); plt.close(fig)

    classes = np.unique(labels)
    fig, ax = plt.subplots(figsize=(7, 4))
    if is_complex:
        for c in classes:
            ax.hist(np.mod(phases[labels == c], 2 * np.pi), bins=36, histtype="step", label=f"class {c}")
        ax.set_xlabel("global phase arg<psi|t> (rad)"); ax.set_ylabel("images")
    else:
        flip = [np.mean(np.cos(phases[labels == c]) < 0) for c in classes]
        ax.bar([str(c) for c in classes], flip, color="#c0504d")
        ax.axhline(0.5, color="k", ls=":", label="coin flip")
        ax.set_ylim(0, 1); ax.set_xlabel("class"); ax.set_ylabel("fraction with psi ~ -t")
    ax.legend(fontsize=7); ax.set_title(f"{tag}: global phase of encoded states")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "phase_by_class.png"), dpi=150); plt.close(fig)

    names = list(readouts)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, key in zip(axes, ("bal_acc", "auc")):
        vals = [readouts[n][key] for n in names]
        ax.barh(names, vals, color=["#7f7f7f", "#c0504d", "#4f81bd", "#9bbb59", "#8064a2", "#f79646"])
        ax.axvline(readouts[names[0]]["chance"] if key == "bal_acc" else 0.5, color="k", ls=":", label="chance")
        for i, v in enumerate(vals):
            ax.text(v, i, f" {v:.3f}", va="center", fontsize=8)
        ax.set_xlim(0, 1.05); ax.set_title(key); ax.legend(fontsize=8)
    axes[1].set_yticklabels([])
    fig.suptitle(f"{tag}: readouts on the same sample")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "readouts.png"), dpi=150); plt.close(fig)


def run_combo(args, dataset, mode, data):
    """Diagnostic (a) for one (dataset, mode): encode the train and evaluation samples,
    measure phases, run the six readouts and save results, figures and states.
    """
    odir = D.out_dir(args.out, dataset, DIAG, mode)
    if os.path.exists(os.path.join(odir, "results.json")) and not args.force:
        print(f"[{dataset}/{mode}] results exist, skipping (use --force)")
        return
    tag = f"{dataset}/{mode}"
    tr_vecs, tr_y, ev_vecs, ev_y, ev_name = data
    t0 = time.time()

    mean, std = D.compute_stats(tr_vecs, mode)
    rng = np.random.default_rng(args.seed)
    tr_idx = D.stratified_sample(tr_y, args.n_train, rng)
    ev_idx = D.stratified_sample(ev_y, args.n_test, rng)
    T_tr, y_tr = D.apply_stats(tr_vecs[tr_idx], mean, std), tr_y[tr_idx]
    T_ev, y_ev = D.apply_stats(ev_vecs[ev_idx], mean, std), ev_y[ev_idx]

    genome, ginfo = find_genome(args, dataset, mode)
    circ = AAECircuit(genome)
    print(f"\n[{tag}] genome from {ginfo['source']}: {circ.n_params} params, "
          f"{circ.n_two_qubit} two-qubit gates, complex={circ.is_complex}")
    if "warning" in ginfo:
        print(f"  WARNING: {ginfo['warning']}")
    print(f"  encoding {len(T_tr)} train + {len(T_ev)} {ev_name} images "
          f"({args.steps} steps x {args.restarts} restarts)...")

    T_all = np.concatenate([T_tr, T_ev])
    fids, params, states = fit_states(circ, T_all, steps=args.steps, lr=args.lr,
                                      restarts=args.restarts, seed=args.seed + 17,
                                      chunk=args.chunk)
    ov = np.sum(states.conj() * T_all, axis=1)                   # <psi|t>
    phases = np.angle(ov)
    aligned = states * np.exp(1j * phases)[:, None]             # <aligned|t> = |ov| >= 0
    if not circ.is_complex:
        aligned = aligned.real
    n = len(T_tr)
    y_all = np.concatenate([y_tr, y_ev])

    shuffled = np.random.default_rng(args.seed + 99).permutation(phases)
    ctrl = T_all * np.exp(-1j * shuffled)[:, None]              # psi ~ e^{-i phi} t
    if not circ.is_complex:
        ctrl = ctrl.real

    print("  running readouts...")
    F = D.amp_features
    readouts = {
        "target_linear": D.linear_probe(T_tr, y_tr, T_ev, y_ev),
        "encoded_linear_asis": D.linear_probe(F(states[:n]), y_tr, F(states[n:]), y_ev),
        "encoded_linear_aligned": D.linear_probe(F(aligned[:n]), y_tr, F(aligned[n:]), y_ev),
        "control_target_randphase": D.linear_probe(F(ctrl[:n]), y_tr, F(ctrl[n:]), y_ev),
        "encoded_fidkernel_svm": D.kernel_svm(D.fidelity_kernel(states[:n], states[:n]), y_tr,
                                              D.fidelity_kernel(states[n:], states[:n]), y_ev),
        "target_fidkernel_svm": D.kernel_svm(D.fidelity_kernel(T_tr, T_tr), y_tr,
                                             D.fidelity_kernel(T_ev, T_tr), y_ev),
    }

    fid_by_class = {int(c): float(fids[y_all == c].mean()) for c in np.unique(y_all)}
    result = {
        "dataset": dataset, "mode": mode, "eval_split": ev_name,
        "n_train": int(n), "n_eval": int(len(T_ev)),
        "train_class_counts": D.class_counts(y_tr), "eval_class_counts": D.class_counts(y_ev),
        "genome": genome, "genome_info": ginfo,
        "circuit": {"n_params": circ.n_params, "n_two_qubit": circ.n_two_qubit,
                    "complex": circ.is_complex, "two_qubit_edges": circ.edges_description()},
        "fit": {"steps": args.steps, "lr": args.lr, "restarts": args.restarts},
        "fidelity": {"mean": float(fids.mean()), "median": float(np.median(fids)),
                     "p10": float(np.percentile(fids, 10)), "p90": float(np.percentile(fids, 90)),
                     "train_mean": float(fids[:n].mean()), "eval_mean": float(fids[n:].mean()),
                     "per_class_mean": fid_by_class},
        "phase": phase_stats(phases, y_all, circ.is_complex),
        "readouts": readouts,
        "runtime_s": round(time.time() - t0, 1),
    }
    result["verdict"] = verdict(result)

    row = {"dataset": dataset, "mode": mode, "genome_source": os.path.basename(str(ginfo["source"])),
           "two_qubit_gates": circ.n_two_qubit, "fid_mean": result["fidelity"]["mean"],
           "phase_R": result["phase"]["resultant_length"],
           "frac_flipped": result["phase"]["frac_sign_flipped"],
           "chance": readouts["target_linear"]["chance"]}
    for k, v in readouts.items():
        row[f"{k}_balacc"] = v["bal_acc"]
        row[f"{k}_auc"] = v["auc"]

    D.save_json(result, os.path.join(odir, "results.json"))
    D.save_json(row, os.path.join(odir, "summary_row.json"))
    classes = sorted(readouts["target_linear"]["recall"])
    D.write_csv([{"class": c, "support_eval": int((y_ev == c).sum()),
                  **{k: v["recall"].get(c) for k, v in readouts.items()}} for c in classes],
                os.path.join(odir, "per_class_recall.csv"))
    if not args.no_save_states:
        np.savez_compressed(os.path.join(odir, "encoded_states.npz"),
                            targets=T_all, states=states, aligned=aligned, params=params,
                            fidelity=fids, overlap=ov, labels=y_all, n_train=n,
                            train_idx=tr_idx, eval_idx=ev_idx)
    make_figures(odir, fids, phases, y_all, readouts, circ.is_complex, tag)

    print(f"  mean fidelity {fids.mean():.4f} | phase R {result['phase']['resultant_length']:.3f} "
          f"| flipped {result['phase']['frac_sign_flipped']:.2f}")
    for k, v in readouts.items():
        print(f"  {k:26s} bal_acc {v['bal_acc']:.3f}  auc {v['auc']:.3f}")
    for line in result["verdict"]:
        print("  > " + line)
    print(f"  saved to {odir}  ({result['runtime_s']} s)")


def main():
    """Command line: sample sizes, fitting budget and circuit choice; runs every dataset/mode."""
    p = D.add_common_args(argparse.ArgumentParser(description=__doc__))
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=800)
    p.add_argument("--steps", type=int, default=250, help="Adam steps per image (repo ENC_SELECT_STEPS)")
    p.add_argument("--lr", type=float, default=0.15, help="repo INNER_LR")
    p.add_argument("--restarts", type=int, default=2, help="repo ENC_SELECT_RESTARTS")
    p.add_argument("--chunk", type=int, default=1024, help="images trained simultaneously")
    p.add_argument("--genome-root", default=None,
                   help="repo experiments/encoding_benchmark folder with <dataset>_<mode>/ subfolders")
    p.add_argument("--genome-source", choices=["fidelity", "kta"], default="fidelity")
    p.add_argument("--genome-json", default=None, help="use this genome for every dataset/mode")
    p.add_argument("--no-save-states", action="store_true")
    args = p.parse_args()

    D.check_genome_root(args.genome_root)
    datasets = D.resolve_datasets(args.data_root, args.datasets)
    print(f"datasets: {datasets} | modes: {args.modes}")
    for ds in datasets:
        tr_x, tr_y, ev_x, ev_y, ev_name = D.load_train_eval(args.data_root, ds)
        data = (D.images_to_vectors(tr_x), tr_y, D.images_to_vectors(ev_x), ev_y, ev_name)
        for mode in args.modes:
            run_combo(args, ds, mode, data)
            D.collect_summaries(args.out, DIAG)
    print(f"\nsummaries: {args.out}/<dataset>/summary_{DIAG}.csv and {args.out}/summary_{DIAG}.csv")


if __name__ == "__main__":
    main()
