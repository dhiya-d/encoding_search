#!/usr/bin/env python3
# ----------------------------------------------------------------
# select_encoder.py  —  Pick the encoder to report.
#
# The search's fitness was measured on one batch with a small fitting
# budget, so the front is optimistically biased. This script re-scores
# every front genome on a DISJOINT batch (val split + train images the
# search never saw) with the full fitting budget, then selects the
# cheapest genome whose retention is within SELECT_TOLERANCE of the best.
#
# The result is written as selected_genome.json in the same format the
# benchmark repo used (plus retention fields), so the diagnostics pick it
# up directly for the final TEST-split report:
#   python diag_a_phase_check.py --genome-root experiments --out experiments_searched
#
# Usage (from the repository root, after encoding_search.py):
#   python search_amplitude/select_encoder.py
#   python search_amplitude/select_encoder.py --dataset bloodmnist --mode centered
#
# Output: experiments/<dataset>/search/<mode>/
#   selection.json  selection.csv  selected_genome.json  selection_front.png
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
import search_config as S
import search_core as SC


def plot(odir, rows, chosen, ceil, tag):
    """Re-evaluated front (retention vs. two-qubit gates) with the chosen genome marked."""
    plt = D.get_plt()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = [r["n2q"] for r in rows]
    ax.errorbar(x, [r["retention"] for r in rows],
                yerr=[r["auc_std"] / ceil["auc"] for r in rows], fmt="o-",
                label="re-evaluated front (± fold std)")
    ax.scatter(x, [r["search_retention"] for r in rows], marker="x", color="grey",
               label="retention during search")
    ax.scatter([chosen["n2q"]], [chosen["retention"]], s=200, facecolors="none",
               edgecolors="r", linewidths=2, label="selected")
    ax.axhline(1.0, color="k", ls=":", label=f"ceiling (AUC {ceil['auc']:.3f})")
    ax2 = ax.twinx()
    ax2.plot(x, [r["fid_mean"] for r in rows], "g--", alpha=0.6)
    ax2.set_ylabel("mean fidelity", color="g")
    ax.set_xlabel("two-qubit gates"); ax.set_ylabel("retention")
    ax.legend(fontsize=8, loc="lower right"); ax.set_title(f"{tag}: selection on a held-out batch")
    fig.tight_layout(); fig.savefig(os.path.join(odir, "selection_front.png"), dpi=150); plt.close(fig)


def run(dataset, mode, force=False):
    """Re-evaluate the final front of one run on the held-out selection batch with a larger
    fitting budget and pick the cheapest genome within SELECT_TOLERANCE of the best.
    """
    tag = f"{dataset}/{mode}"
    odir = os.path.join(S.OUT_ROOT, dataset, "search", mode)
    fpath = os.path.join(odir, "front_final.json")
    if not os.path.exists(fpath):
        raise SystemExit(f"{fpath} not found — run encoding_search.py for {tag} first")
    spath = os.path.join(odir, "selected_genome.json")
    if (not force and os.path.exists(spath) and os.path.getmtime(spath) >= os.path.getmtime(fpath)
            and D.load_json(spath).get("objective", "retention") == S.OBJECTIVE):
        sel = D.load_json(spath)
        print(f"[{tag}] selection is up to date: {sel['gate_count']} CNOTs "
              f"(use --force to redo)")
        return
    front = D.load_json(fpath)["front"]
    t0 = time.time()
    print(f"\n{'=' * 64}\n[{tag}] re-evaluating {len(front)} front genomes on a held-out batch...")

    _, select = SC.prepare_batches(dataset, mode)
    folds = D.make_folds(select["y"], S.CV_FOLDS, S.SEED + 1)
    ceil = SC.ceiling(select["T"], select["y"], folds)
    print(f"  {len(select['T'])} images, classes {D.class_counts(select['y'])}")
    print(f"  ceiling: CV AUC {ceil['auc']:.3f} ± {ceil['auc_std']:.3f}, bal_acc {ceil['bal_acc']:.3f}")

    ev = SC.Evaluator(select["T"], select["y"], folds, ceil["C"], ceil["auc"],
                      S.SELECT_FIT_STEPS, S.SELECT_FIT_RESTARTS, S.SEED + 29)
    try:
        res, _ = ev.evaluate([f["genome"] for f in front])
    finally:
        ev.close()

    rows = []
    for f, r in zip(front, res):
        rows.append({"n2q": r["n2q"], "retention": r["retention"], "auc": r["auc"],
                     "auc_std": r["auc_std"], "bal_acc": r["bal_acc"],
                     "fid_mean": r["fid_mean"], "fid_median": r["fid_median"],
                     "search_retention": f["retention"], "search_fid_mean": f["fid_mean"],
                     "genome": f["genome"]})
    rows.sort(key=lambda r: r["n2q"])
    key = "fid_mean" if S.OBJECTIVE == "fidelity" else "retention"
    tol = S.SELECT_TOLERANCE_FIDELITY if S.OBJECTIVE == "fidelity" else S.SELECT_TOLERANCE
    best = max(r[key] for r in rows)
    chosen = min((r for r in rows if r[key] >= best - tol),
                 key=lambda r: (r["n2q"], -r[key]))

    selected = {
        "dataset": dataset, "mode": mode,
        "centered": mode == "centered", "whitened": False,
        "genome": chosen["genome"], "gate_count": chosen["n2q"],
        "holdout_fidelity": chosen["fid_mean"],
        "holdout_retention": chosen["retention"],
        "holdout_auc": chosen["auc"], "holdout_bal_acc": chosen["bal_acc"],
        "holdout_ceiling_auc": ceil["auc"],
        "objective": S.OBJECTIVE,
        "selection_rule": f"cheapest within {tol} of best {S.OBJECTIVE} ({best:.3f})",
    }
    D.save_json(selected, os.path.join(odir, "selected_genome.json"))
    D.save_json({"ceiling": ceil, "rows": rows, "selected": selected,
                 "batch_source_counts": {s: sum(1 for a, _ in select["source"] if a == s)
                                         for s in ("train", "val")},
                 "runtime_s": round(time.time() - t0, 1)},
                os.path.join(odir, "selection.json"))
    D.write_csv([{**{k: v for k, v in r.items() if k != "genome"}, "genome": str(r["genome"])}
                 for r in rows], os.path.join(odir, "selection.csv"))
    plot(odir, rows, chosen, ceil, tag)

    for r in rows:
        mark = "  <- selected" if r is chosen else ""
        print(f"  {r['n2q']:3d} gates: retention {r['retention']:.3f} (search {r['search_retention']:.3f}), "
              f"AUC {r['auc']:.3f}, F {r['fid_mean']:.3f}{mark}")
    print(f"  saved selected_genome.json to {odir}  ({time.time() - t0:.0f}s)")


def main():
    """Command line: select runs, seed and objective, then run the selection for each."""
    p = argparse.ArgumentParser(description="select the encoder from the final front")
    p.add_argument("--dataset", default=None)
    p.add_argument("--mode", default=None, choices=SC.VALID_MODES)
    p.add_argument("--seed", type=int, default=None, help="search seed (default search_config.SEED)")
    p.add_argument("--objective", choices=["retention", "fidelity"], default=None,
                   help="must match the objective the search used")
    p.add_argument("--out-root", default=None, help="output root (default search_config.OUT_ROOT)")
    p.add_argument("--quick", action="store_true", help="match encoding_search.py --quick")
    p.add_argument("--force", action="store_true", help="redo the selection even if it is up to date")
    args = p.parse_args()
    if args.quick:
        from encoding_search import apply_quick
        apply_quick()
        S.OUT_ROOT = "experiments_quick"
    if args.seed is not None:
        S.SEED = args.seed
    if args.out_root:
        S.OUT_ROOT = args.out_root
    if args.objective:
        S.OBJECTIVE = args.objective
    if args.dataset and not args.mode:
        raise SystemExit("--dataset needs --mode")
    for ds, mode in ([(args.dataset, args.mode)] if args.dataset else S.RUNS):
        run(ds, mode, args.force)


if __name__ == "__main__":
    main()
