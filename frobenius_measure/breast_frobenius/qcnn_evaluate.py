"""Stage 2 final evaluation: train a chosen QCNN genome properly and report
HELD-OUT TEST balanced accuracy.

The search selects on validation, so the validation number it reports is
selection-biased. On RetinaMNIST (120 validation images, smallest class n=12)
that bias measured out to ~0.15-0.17. BreastMNIST was re-split 75/15/10
(resplit_data.py) specifically to make this validation set bigger and better
balanced (118 images, smallest class n=32) than the stock MedMNIST split (78
images, smallest class n=21) -- expect a smaller gap here, but still check it
with this script rather than assuming. This script is the honest number.

For context it also fits a linear probe on the same encoded states, so the
QCNN can be compared against a linear readout of exactly the same input.

Example:
    python qcnn_evaluate.py --genome experiments/qcnn_breast/best_qcnn.json \
        --data encoded/breast_seed0 --target-qubits 1 --epochs 150 --seeds 5 \
        --linear-baseline 0.83
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, recall_score

from qcnn_genome import QCNNGenome, configure
from qcnn_model import TrainConfig, predict, train_and_score
from qcnn_search import load_cache


def main():
    ap = argparse.ArgumentParser(description="Final test evaluation of a QCNN genome.")
    ap.add_argument("--genome", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--use-head", action="store_true")
    ap.add_argument("--target-qubits", type=int, default=2)
    ap.add_argument("--free-pooling", action="store_true")
    ap.add_argument("--seeds", type=int, default=5, help="independent training runs")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--linear-baseline", type=float, default=None)
    a = ap.parse_args()

    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    gd = json.load(open(a.genome))
    configure(gd.get("n_qubits", 8), gd.get("target_qubits", a.target_qubits),
              free_pooling=gd.get("free_pooling", a.free_pooling))
    g = QCNNGenome.from_dict(gd)
    data, meta = load_cache(a.data)
    K, chance = int(meta["n_classes"]), float(meta["chance"])

    # merge train+val for the final fit, evaluate once on test
    Xfull = np.concatenate([data["Xtr"], data["Xval"]])
    yfull = np.concatenate([data["ytr"], data["yval"]])
    full = {"Xtr": Xfull, "ytr": yfull, "Xval": data["Xte"], "yval": data["yte"]}

    accs, recalls = [], []
    for s in range(a.seeds):
        cfg = TrainConfig(epochs=a.epochs, lr=a.lr, batch_size=a.batch_size,
                          seed=s, use_head=a.use_head)
        r = train_and_score(g, full, cfg, device=device, n_classes=K,
                            return_params=True)
        accs.append(r["bal_acc"])
        th = torch.as_tensor(r["theta"], device=device)
        pred = predict(g, th, data["Xte"], device=device)
        recalls.append(recall_score(data["yte"], pred, average=None,
                                    zero_division=0).round(3).tolist())
        print(f"  seed {s}: test bal_acc = {r['bal_acc']:.4f}")

    # linear probe on the same encoded states, for context
    lp = LogisticRegression(max_iter=3000, class_weight="balanced")
    lp.fit(data["Xtr"], data["ytr"])
    lin_enc = balanced_accuracy_score(data["yte"], lp.predict(data["Xte"]))

    print("\n" + "=" * 68)
    print(f" genome  : {g.summary()}")
    print(f" encoder : {meta['encoder_summary']}")
    print(f" data    : {meta['merge']} | {K} classes | chance = {chance:.3f}")
    print("-" * 68)
    print(f"  QCNN TEST balanced accuracy : {np.mean(accs):.4f} +/- {np.std(accs):.4f} "
          f"(n={len(accs)})")
    print(f"  per-class recall (seed 0)   : {recalls[0]}")
    print(f"  linear probe on same states : {lin_enc:.4f}")
    if a.linear_baseline is not None:
        print(f"  linear probe on raw targets : {a.linear_baseline:.4f}")
    print("-" * 68)
    m = float(np.mean(accs))
    print(f"  vs chance                   : {m - chance:+.4f}")
    print(f"  vs linear-on-encoded-states : {m - lin_enc:+.4f}"
          f"   {'BEATS' if m > lin_enc else 'below'}")
    if a.linear_baseline is not None:
        print(f"  vs linear-on-raw-targets    : {m - a.linear_baseline:+.4f}"
              f"   {'BEATS' if m > a.linear_baseline else 'below'}")
    print("=" * 68)


if __name__ == "__main__":
    main()