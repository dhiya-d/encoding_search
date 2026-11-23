"""Evaluate a saved genome on the held-out split.

Reports two things that answer different questions:

  1. RECONSTRUCTION -- mean/max ||t - s||_F and the equivalent overlap on the
     test split. This is what the search optimised.

  2. CLASS SIGNAL   -- a linear probe (balanced accuracy) fitted on the ENCODED
     STATES, compared against the same probe on the raw amplitude TARGETS.
     This is the check the Frobenius objective cannot make for itself: a circuit
     can reconstruct the image well and still destroy the class information.
     If probe(encoded) << probe(targets), the encoding is label-blind.

Example:
    python evaluate.py --genome experiments/r_seed0/best_genome.json \
        --data-dir retinamnist --size 16 --fit-steps 400 --restarts 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np                                       # noqa: E402
import torch                                             # noqa: E402
from sklearn.linear_model import LogisticRegression      # noqa: E402
from sklearn.metrics import balanced_accuracy_score      # noqa: E402

from data_local import load                              # noqa: E402
from encoder import fit_batch                            # noqa: E402
from genome import Genome                                # noqa: E402


def probe(Xtr, ytr, Xte, yte, seed=0):
    m = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
    m.fit(Xtr, ytr)
    return float(balanced_accuracy_score(yte, m.predict(Xte)))


def main():
    ap = argparse.ArgumentParser(description="Evaluate a Frobenius-searched genome.")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--data-dir", default="retinamnist")
    ap.add_argument("--merge", default="retina4", choices=("retina4",))
    ap.add_argument("--channel", default="gray", choices=("gray", "green", "rgb"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--no-center", dest="center", action="store_false", default=True)
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--fit-lr", type=float, default=0.08)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--max-images", type=int, default=1200,
                    help="cap per split (memory); breastmnist is small so all fit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    g = Genome.from_dict(json.load(open(a.genome)))
    d = load(a.data_dir, channel=a.channel, size=a.size, center=a.center, merge=a.merge)

    enc = {}
    for split, xk, yk in (("train", "Xtr", "ytr"), ("test", "Xte", "yte")):
        X = torch.as_tensor(d[xk][:a.max_images])
        r = fit_batch(g, X, steps=a.fit_steps, lr=a.fit_lr, restarts=a.restarts,
                      seed=a.seed, device=device, return_states=True)
        enc[split] = {"states": r["states"].numpy(), "y": d[yk][:a.max_images],
                      "targets": d[xk][:a.max_images], "r": r}

    te = enc["test"]["r"]
    p_target = probe(enc["train"]["targets"], enc["train"]["y"],
                     enc["test"]["targets"], enc["test"]["y"], a.seed)
    p_state = probe(enc["train"]["states"], enc["train"]["y"],
                    enc["test"]["states"], enc["test"]["y"], a.seed)

    print("=" * 66)
    print(f" genome : {g.summary()}")
    print(f" data   : {a.data_dir} {a.channel}/{a.size}px merge={a.merge} | "
          f"{d['n_classes']} classes | chance={d['chance']:.3f}")
    print("-" * 66)
    print(" RECONSTRUCTION (test split)")
    print(f"   mean ||t-s||_F      : {te['mean_frobenius']:.4f}   (0 = perfect, 2 = worst)")
    print(f"   max  ||t-s||_F      : {te['max_frobenius']:.4f}")
    print(f"   mean overlap <t,s>  : {te['mean_overlap']:.4f}")
    print("-" * 66)
    print(" CLASS SIGNAL (linear probe, balanced accuracy)")
    print(f"   on raw targets      : {p_target:.4f}")
    print(f"   on encoded states   : {p_state:.4f}   ({p_state - p_target:+.4f})")
    lost = p_target - p_state
    if lost > 0.05:
        print("   -> encoding LOST class signal despite its reconstruction score.")
    else:
        print("   -> class signal preserved through the encoding.")
    print("=" * 66)


if __name__ == "__main__":
    main()
