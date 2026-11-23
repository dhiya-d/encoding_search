#!/usr/bin/env python3
# ----------------------------------------------------------------
# run_all.py  —  reproduce every result in the paper, in order.
#
# Run from the repository root. Every stage is resumable: finished work is
# detected and reused, so after an interruption rerun the same command.
#
# Stages (in dependency order):
#   data         download the MedMNIST datasets into data/ (tools/prepare_dataset.py)
#   diagnostics  diagnostics (a)-(c) on the four small datasets with the
#                hand-designed 31-CNOT reference circuit: sign ambiguity,
#                exact-state ceilings, class-signal budget (paper Sec. V A, Fig. 2)
#   amplitude    amplitude-encoder search per dataset: reference circuit on the
#                test split, retention-driven search + held-out selection
#                (seed 0), test-split encoding, trained classifiers
#   fourier      Fourier (angle) encoding search + downstream evaluation (seed 0)
#   multiseed    search seeds 1 and 2 for both searches (Table IV, Fig. 5)
#   objective    retention vs fidelity search vs literature vs random circuits
#                (Table II, Fig. 3)
#   noise        gate noise and finite measurement budgets (Table V, Fig. 6)
#   kernel       exact classical kernel of entanglement-free encodings
#   report       claims report over all runs (reporting/make_report.py)
#   tables       LaTeX tables (reporting/paper_tables.py)
#   figures      paper figures (reporting/paper_figures.py)
#
# Usage:
#   python run_all.py --dry-run                     # list every command
#   python run_all.py                               # everything
#   python run_all.py --stages data diagnostics     # selected stages
#   python run_all.py --quick                       # tiny settings, pipeline test only
#
# --quick runs the whole chain with tiny settings (few images, generations
# and epochs, one extra search seed, 3 random circuits) inside a separate
# working folder quick_run/ (with quick_run/data linked to data/), so its
# results can never be mixed up with real ones. It checks the installation
# and the pipeline in roughly an hour on 8 cores; its numbers mean nothing.
#
# Datasets (defaults reproduce the paper):
#   --main        bloodmnist and pathmnist: full pipeline incl. multiseed,
#                 objective and noise studies (<dataset>:<mode>)
#   --amp-extra   breastmnist (uncentered): amplitude search only
#   --fourier-extra  breastmnist, dermamnist, retinamnist: Fourier search only
#
# Rough runtime on 8 cores: diagnostics ~1 h; per main dataset amplitude
# ~1 h, Fourier ~20 min, multiseed ~1.5 h, objective ~3-4 h, noise ~1 h.
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "core"))
import repo  # noqa: E402,F401

import argparse
import os
import subprocess
import sys
import time

STAGES = ["data", "diagnostics", "amplitude", "fourier", "multiseed", "objective",
          "noise", "kernel", "report", "tables", "figures"]

# the four small datasets used by the diagnostics (paper Fig. 2)
SMALL = ["bloodmnist", "breastmnist", "dermamnist", "retinamnist"]

# training/validation caps for large datasets (stratified); test split always complete
CAPS = {"pathmnist": (20000, 5000)}


def show(cmd):
    """Command line for printing, with the script shown relative to the repository root."""
    return " ".join([os.path.relpath(cmd[1], repo.ROOT)] + cmd[2:])


def build_commands(a):
    """List of (stage, command) pairs for the selected stages."""
    py = sys.executable
    main = [tuple(x.split(":")) for x in a.main]                 # (dataset, mode)
    amp_runs = main + [tuple(x.split(":")) for x in a.amp_extra]
    fourier_sets = [ds for ds, _ in main] + a.fourier_extra

    # Extra arguments per script. In quick mode every script gets its own
    # small settings, and the searches are told to write to the STANDARD
    # folder names (they would otherwise use *_quick folders), so that every
    # later stage finds their results under its default paths. Quick mode
    # runs in its own working folder, so nothing is overwritten.
    Q = a.quick
    x_amp = ["--quick", "--out-root", "experiments"] if Q else []
    x_four = ["--quick", "--out-root", "experiments_fourier"] if Q else []
    # (diag_a and classifier settings match the quick settings of the studies,
    #  which reuse these results; otherwise reruns would retrain them)
    x_diag_a = ["--n-train", "300", "--n-test", "150", "--steps", "60", "--restarts", "1"] if Q else []
    x_diag_b = ["--n-train", "300", "--n-test", "150", "--kta-n", "200", "--n-perm", "20"] if Q else []
    x_diag_c = ["--max-n", "300"] if Q else []
    x_clf = ["--epochs", "4"] if Q else []
    x_study = ["--quick"] if Q else []
    x_objective = ["--quick", "--n-random", "3"] if Q else []
    cmds = []

    def add(stage, cmd):
        """Keep the command only if its stage was selected."""
        if stage in a.stages:
            cmds.append((stage, cmd))

    # data: every dataset that is used anywhere and is not yet in data/
    needed = sorted(set(SMALL) | {ds for ds, _ in amp_runs} | set(fourier_sets))
    for ds in needed:
        if not os.path.exists(os.path.join(repo.ROOT, "data", ds, "test_labels.npy")):
            cmd = [py, repo.script("prepare_dataset.py"), ds, "--out", os.path.join(repo.ROOT, "data")]
            if ds in CAPS:
                cmd += ["--max-train", str(CAPS[ds][0]), "--max-val", str(CAPS[ds][1])]
            add("data", cmd)

    # diagnostics on the small datasets, all preprocessing modes, reference circuit
    add("diagnostics", [py, repo.script("diag_b_ceiling.py"), "--datasets"] + SMALL + x_diag_b)
    add("diagnostics", [py, repo.script("diag_a_phase_check.py"), "--datasets"] + SMALL
        + ["--out", "experiments_handref"] + x_diag_a)
    add("diagnostics", [py, repo.script("diag_c_spectral.py"), "--datasets"] + SMALL
        + ["--out", "experiments_handref"] + x_diag_c)

    # amplitude-encoder search per run
    for ds, mode in amp_runs:
        # reference circuit on the test split (skipped if the diagnostics already did it)
        add("amplitude", [py, repo.script("diag_a_phase_check.py"), "--datasets", ds, "--modes", mode,
                          "--out", "experiments_handref"] + x_diag_a)
        add("amplitude", [py, repo.script("encoding_search.py"), "--dataset", ds, "--mode", mode] + x_amp)
        add("amplitude", [py, repo.script("select_encoder.py"), "--dataset", ds, "--mode", mode] + x_amp)
        # the selected circuit on the test split (encodes the images, saves the states)
        add("amplitude", [py, repo.script("diag_a_phase_check.py"), "--genome-root", "experiments",
                          "--out", "experiments_searched", "--datasets", ds, "--modes", mode] + x_diag_a)
        add("amplitude", [py, repo.script("train_on_encoded_states.py"), "--runs", f"{ds}:{mode}"] + x_clf)

    # Fourier search per dataset
    for ds in fourier_sets:
        add("fourier", [py, repo.script("search_encoding.py"), "--dataset", ds] + x_four)
        add("fourier", [py, repo.script("train_downstream.py"), "--dataset", ds] + x_four)

    # studies on the main datasets
    for ds, mode in main:
        run = f"{ds}:{mode}"
        seeds = [str(s) for s in a.seeds]
        add("multiseed", [py, repo.script("multiseed.py"), "--aae-run", run, "--fourier-dataset", ds,
                          "--seeds"] + seeds + x_study)
        add("objective", [py, repo.script("objective_study.py"), "--run", run, "--seeds", "0"]
            + seeds + ["--jobs", str(a.jobs)] + x_objective)
        add("noise", [py, repo.script("noise_study.py"), "--run", run, "--fourier-dataset", ds] + x_study)
        add("kernel", [py, repo.script("product_kernel_check.py"), "--dataset", ds,
                       "--seeds", "0"] + seeds + x_study)

    # reports, tables and figures
    add("report", [py, repo.script("make_report.py"), "--runs"] + [f"{d}:{m}" for d, m in amp_runs])
    add("tables", [py, repo.script("paper_tables.py"), "--pairs"] + [f"{d}:{m}:{d}" for d, m in main])
    for ds, _ in main:
        add("figures", [py, repo.script("paper_figures.py"), "--dataset", ds,
                        "--out", os.path.join("paper_figures", ds)])
    return cmds


def main():
    """Parse the options, set up quick mode, print the commands and run them in order."""
    p = argparse.ArgumentParser(description="reproduce all results of the paper")
    p.add_argument("--stages", nargs="*", default=STAGES, choices=STAGES)
    p.add_argument("--main", nargs="*", default=["bloodmnist:centered", "pathmnist:centered"],
                   help="datasets for the full pipeline, as <dataset>:<mode>")
    p.add_argument("--amp-extra", nargs="*", default=["breastmnist:uncentered"],
                   help="further amplitude-search runs (<dataset>:<mode>)")
    p.add_argument("--fourier-extra", nargs="*", default=["breastmnist", "dermamnist", "retinamnist"],
                   help="further datasets for the Fourier search only")
    p.add_argument("--seeds", nargs="*", type=int, default=[1, 2], help="extra search seeds")
    p.add_argument("--jobs", type=int, default=2, help="circuits evaluated in parallel (objective study)")
    p.add_argument("--quick", action="store_true", help="tiny settings everywhere (pipeline test only)")
    p.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    p.add_argument("--keep-going", action="store_true", help="continue with the next command after a failure")
    a = p.parse_args()

    workdir = None
    if a.quick:
        # isolated working folder; the datasets are shared through a link
        workdir = os.path.join(repo.ROOT, "quick_run")
        os.makedirs(workdir, exist_ok=True)
        link = os.path.join(workdir, "data")
        if not os.path.exists(link):
            os.symlink(os.path.join(repo.ROOT, "data"), link)
        if a.seeds == [1, 2]:
            a.seeds = [1]                       # one extra search seed is enough for a test
        print(f"quick mode: working folder {workdir}")

    cmds = build_commands(a)
    print(f"{len(cmds)} commands:")
    for stage, cmd in cmds:
        print(f"  [{stage}] " + show(cmd))
    if a.dry_run:
        return

    failed, t0 = [], time.time()
    for i, (stage, cmd) in enumerate(cmds, 1):
        print(f"\n{'=' * 78}\n[{i}/{len(cmds)}] ({stage}) {show(cmd)}\n{'=' * 78}", flush=True)
        t = time.time()
        r = subprocess.run(cmd, cwd=workdir)
        print(f"-> exit {r.returncode} after {time.time() - t:.0f}s (total {(time.time() - t0) / 60:.0f} min)",
              flush=True)
        if r.returncode != 0:
            failed.append(show(cmd))
            if not a.keep_going:
                print("stopping (rerun the same command to resume, or use --keep-going)")
                break
    if failed:
        print("\nfailed:\n  " + "\n  ".join(failed))
        sys.exit(1)
    print(f"\nall done in {(time.time() - t0) / 60:.0f} min")


if __name__ == "__main__":
    main()
