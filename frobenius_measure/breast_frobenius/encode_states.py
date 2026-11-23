"""Stage 1 -> Stage 2 bridge: encode every image with the chosen encoding
genome and cache the resulting states to disk.

This must happen once, before the QCNN search. Re-fitting per-image encoder
angles inside every QCNN fitness evaluation would be both enormously wasteful
and confounding -- the QCNN search must see a FIXED input representation so that
differences in accuracy are attributable to the QCNN architecture alone.

Requires the Frobenius encoding suite (encoder.py, genome.py, data_local.py) --
point --frob-dir at it.

Example:
    python encode_states.py \
        --frob-dir . \
        --genome experiments/b_seed0/best_genome.json \
        --data-dir breastmnist \
        --merge none --fit-steps 400 --restarts 3 \
        --out encoded/breast_seed0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser(description="Cache encoded states for the QCNN stage.")
    ap.add_argument("--frob-dir", required=True, help="path to the Frobenius encoding suite")
    ap.add_argument("--genome", required=True, help="chosen encoding genome JSON")
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--channel", default="gray", choices=("gray", "green", "rgb"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--no-center", dest="center", action="store_false", default=True)
    ap.add_argument("--merge", default="none", choices=("none",))
    ap.add_argument("--fit-steps", type=int, default=400)
    ap.add_argument("--fit-lr", type=float, default=0.08)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=512, help="images per encoding batch")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", required=True, help="output directory for the cache")
    a = ap.parse_args()

    sys.path.insert(0, os.path.abspath(a.frob_dir))
    from data_local import load                 # noqa: E402
    from encoder import fit_batch               # noqa: E402
    from genome import Genome                   # noqa: E402

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    g = Genome.from_dict(json.load(open(a.genome)))
    d = load(a.data_dir, channel=a.channel, size=a.size, center=a.center, merge=a.merge)

    print(f"[encoder] {g.summary()}")
    print(f"[data]    {a.data_dir} {a.channel}/{a.size}px merge={a.merge} | "
          f"{d['n_classes']} classes | {d['n_qubits']} qubits | device={device}")

    os.makedirs(a.out, exist_ok=True)
    meta = {"encoder_genome": g.to_dict(), "encoder_summary": g.summary(),
            "data_dir": a.data_dir, "channel": a.channel, "size": a.size,
            "center": a.center, "merge": a.merge, "n_classes": d["n_classes"],
            "n_qubits": d["n_qubits"], "chance": d["chance"],
            "fit_steps": a.fit_steps, "restarts": a.restarts, "seed": a.seed}

    for split, xk, yk in (("train", "Xtr", "ytr"), ("val", "Xval", "yval"),
                          ("test", "Xte", "yte")):
        X = d[xk]
        states, frobs = [], []
        for k in range(0, len(X), a.chunk):
            xb = torch.as_tensor(X[k:k + a.chunk])
            r = fit_batch(g, xb, steps=a.fit_steps, lr=a.fit_lr, restarts=a.restarts,
                          seed=a.seed, device=device, return_states=True)
            states.append(r["states"].numpy())
            frobs.append(r["frobenius"].numpy())
            print(f"  {split}: {min(k + a.chunk, len(X))}/{len(X)}  "
                  f"mean|χ|={float(np.concatenate(frobs).mean()):.4f}", end="\r")
        S = np.concatenate(states).astype(np.float32)
        Fr = np.concatenate(frobs).astype(np.float32)
        np.save(os.path.join(a.out, f"{split}_states.npy"), S)
        np.save(os.path.join(a.out, f"{split}_labels.npy"), d[yk].astype(np.int64))
        np.save(os.path.join(a.out, f"{split}_frobenius.npy"), Fr)
        meta[f"{split}_n"] = int(len(S))
        meta[f"{split}_mean_frobenius"] = float(Fr.mean())
        print(f"  {split}: {len(S)} states | mean |χ|={Fr.mean():.4f} "
              f"| overlap={1 - Fr.mean() ** 2 / 2:.4f}          ")

    with open(os.path.join(a.out, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"\ncached -> {a.out}")


if __name__ == "__main__":
    main()
