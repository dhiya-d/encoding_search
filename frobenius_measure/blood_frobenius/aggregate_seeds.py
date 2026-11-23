"""Aggregate cost_signal_curve.csv across seeds.

Different seeds produce Pareto fronts at DIFFERENT gate counts, so the curves
cannot be averaged row-wise. This script does two things instead:

  1. SATURATION POINT per seed -- the smallest gate count g* such that the mean
     probe over all candidates with gates >= g* is within `tolerance` of the raw
     target baseline. Using a suffix mean (rather than a single point) makes it
     robust to the +/-0.03 bounce of a small test split. Reported as mean +/- sd
     across seeds: this is the headline number.

  2. BINNED CURVES -- probe and frobenius averaged inside gate bins, with error
     bars, so the two-regime shape can be plotted across seeds.

Example:
    python aggregate_seeds.py --dirs experiments/froblo_seed0 experiments/froblo_seed1 \
        experiments/froblo_seed2 experiments/froblo_seed3 experiments/froblo_seed4 \
        --tolerance 0.03 --outdir experiments/frob_aggregate
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics as st
import sys

from plot_safety import safe_render

DEFAULT_BINS = [(1, 5), (6, 10), (11, 15), (16, 25), (26, 40), (41, 60), (61, 200)]


def read_curve(path):
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (int(v) if k in ("gates", "cry", "params", "depth")
                             else float(v)) for k, v in r.items()})
    return sorted(rows, key=lambda r: r["gates"])


def saturation_point(rows, tolerance):
    """Smallest gate count whose SUFFIX mean probe is within tolerance of baseline."""
    base = rows[0]["probe_raw"]
    for i, r in enumerate(rows):
        suffix = [x["probe_encoded"] for x in rows[i:]]
        if st.mean(suffix) >= base - tolerance:
            return r["gates"], base
    return None, base


def main():
    ap = argparse.ArgumentParser(description="Aggregate cost-signal curves across seeds.")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="seed dirs (or a glob like 'experiments/froblo_seed*')")
    ap.add_argument("--tolerance", type=float, default=0.03,
                    help="probe within this of baseline counts as preserved")
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()

    dirs = []
    for d in a.dirs:
        dirs.extend(sorted(glob.glob(d)) if any(c in d for c in "*?[") else [d])

    curves, sats, bases = {}, [], []
    for d in dirs:
        p = os.path.join(d, "cost_signal_curve.csv")
        if not os.path.exists(p):
            print(f"  (skip {d}: no cost_signal_curve.csv)")
            continue
        rows = read_curve(p)
        curves[d] = rows
        g, base = saturation_point(rows, a.tolerance)
        sats.append(g)
        bases.append(base)
        print(f"  {os.path.basename(d):>18}: {len(rows):>2} candidates | "
              f"saturation at {g} gates | baseline {base:.4f}")

    if not curves:
        print("no curves found")
        sys.exit(1)

    print("\n" + "=" * 60)
    valid = [s for s in sats if s is not None]
    if valid:
        if len(valid) > 1:
            print(f" SATURATION POINT : {st.mean(valid):.1f} +/- {st.stdev(valid):.1f} gates "
                  f"(n={len(valid)})")
        else:
            print(f" SATURATION POINT : {valid[0]} gates (n=1, no error bar yet)")
        print(f" raw-target baseline : {st.mean(bases):.4f}")
        print(f" exact encoding cost : ~{2 ** curves[dirs[0]][0].get('n_qubits', 8)} gates "
              f"-> {2 ** 8 / st.mean(valid):.0f}x cheaper")
    print("=" * 60)

    # binned curves
    print(f"\n {'gate bin':>10} {'n':>3} {'frob':>16} {'probe':>16}")
    binned = []
    for lo, hi in DEFAULT_BINS:
        fs = [r["frob_test"] for rows in curves.values() for r in rows if lo <= r["gates"] <= hi]
        ps = [r["probe_encoded"] for rows in curves.values() for r in rows if lo <= r["gates"] <= hi]
        if not ps:
            continue
        fm = st.mean(fs); fsd = st.stdev(fs) if len(fs) > 1 else 0.0
        pm = st.mean(ps); psd = st.stdev(ps) if len(ps) > 1 else 0.0
        binned.append({"bin_lo": lo, "bin_hi": hi, "n": len(ps),
                       "frob_mean": fm, "frob_sd": fsd, "probe_mean": pm, "probe_sd": psd})
        print(f" {f'{lo}-{hi}':>10} {len(ps):>3}   {fm:.4f} +/- {fsd:.4f}   {pm:.4f} +/- {psd:.4f}")

    if a.outdir:
        os.makedirs(a.outdir, exist_ok=True)
        cp = os.path.join(a.outdir, "aggregate_bins.csv")
        with open(cp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(binned[0].keys()))
            w.writeheader(); w.writerows(binned)
        print(f"\nsaved -> {cp}")
        plot_path = os.path.join(a.outdir, "aggregate_curve.png")
        if safe_render("plot", _plot, curves, binned, st.mean(bases), plot_path):
            print(f"plot  -> {plot_path}")


def _plot(curves, binned, base, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    for rows in curves.values():
        ax.plot([r["gates"] for r in rows], [r["probe_encoded"] for r in rows],
                "-", color="#0E9384", alpha=0.25, lw=1)
    x = [(b["bin_lo"] + min(b["bin_hi"], 80)) / 2 for b in binned]
    y = [b["probe_mean"] for b in binned]
    e = [b["probe_sd"] for b in binned]
    ax.errorbar(x, y, yerr=e, fmt="-s", color="#0E9384", capsize=4, lw=2,
                label="probe on encoded states (binned mean)")
    ax.axhline(base, ls="--", color="#B45309", lw=1.3, label=f"probe on raw targets ({base:.3f})")
    ax.axhline(0.5, ls=":", color="#C0392B", lw=1.2, label="chance")
    ax.set_xlabel("two-qubit gate count")
    ax.set_ylabel("balanced accuracy")
    ax.set_title("Class signal vs encoding cost (across seeds)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
