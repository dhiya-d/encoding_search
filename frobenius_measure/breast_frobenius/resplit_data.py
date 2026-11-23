"""One-off: re-split breastmnist/{train,val,test}_{images,labels}.npy into new
75/15/10 train/val/test proportions, stratified by class.

The shipped MedMNIST split is 546/78/156 (70/10/20). This project's QCNN
search showed that split's 78-image validation set (smallest class n=21)
produces a large winner's-curse gap between validation and held-out test
accuracy -- a bigger, class-stratified validation split should shrink that gap.

Pools all three original splits (780 images total), shuffles per class with a
fixed seed, and reassigns 75% train / 15% val / 10% test. Originals are backed
up to breastmnist_original_split/ before anything is overwritten.

Usage:
    python resplit_data.py --data-dir breastmnist --seed 0
"""
from __future__ import annotations

import argparse
import os
import shutil

import numpy as np


def main():
    ap = argparse.ArgumentParser(description="Re-split BreastMNIST 75/15/10, stratified.")
    ap.add_argument("--data-dir", default="breastmnist")
    ap.add_argument("--train-frac", type=float, default=0.75)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backup-dir", default=None,
                    help="default: <data-dir>_original_split")
    a = ap.parse_args()

    assert abs(a.train_frac + a.val_frac + a.test_frac - 1.0) < 1e-9, \
        "fractions must sum to 1.0"

    backup = a.backup_dir or f"{a.data_dir.rstrip('/')}_original_split"
    if os.path.exists(backup):
        raise SystemExit(f"backup dir {backup} already exists -- refusing to "
                          f"overwrite; delete it first if you want to re-run.")
    shutil.copytree(a.data_dir, backup)
    print(f"[backup] original splits copied to {backup}/")

    # pool everything
    Xs, ys = [], []
    for split in ("train", "val", "test"):
        Xs.append(np.load(os.path.join(a.data_dir, f"{split}_images.npy")))
        ys.append(np.load(os.path.join(a.data_dir, f"{split}_labels.npy")))
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    n = len(y)
    print(f"[pool] {n} images total (was {[len(a) for a in ys]} train/val/test)")

    rng = np.random.RandomState(a.seed)
    y_flat = y.ravel()
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y_flat):
        idx = np.where(y_flat == c)[0]
        rng.shuffle(idx)
        n_c = len(idx)
        n_tr = int(round(n_c * a.train_frac))
        n_va = int(round(n_c * a.val_frac))
        # test gets the remainder so rounding always sums to n_c exactly
        train_idx.append(idx[:n_tr])
        val_idx.append(idx[n_tr:n_tr + n_va])
        test_idx.append(idx[n_tr + n_va:])
        print(f"  class {c}: n={n_c} -> train={n_tr} val={n_va} "
              f"test={n_c - n_tr - n_va}")

    def gather(idxs):
        idx = np.concatenate(idxs)
        rng.shuffle(idx)
        return X[idx], y[idx]

    for split, idxs in (("train", train_idx), ("val", val_idx), ("test", test_idx)):
        Xs_, ys_ = gather(idxs)
        np.save(os.path.join(a.data_dir, f"{split}_images.npy"), Xs_)
        np.save(os.path.join(a.data_dir, f"{split}_labels.npy"), ys_)
        vals, counts = np.unique(ys_, return_counts=True)
        print(f"[write] {split}: n={len(ys_)} class_counts={dict(zip(vals.tolist(), counts.tolist()))}")

    print(f"\ndone. Original split preserved in {backup}/")


if __name__ == "__main__":
    main()
