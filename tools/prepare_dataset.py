#!/usr/bin/env python3
# ----------------------------------------------------------------
# prepare_dataset.py  —  download a MedMNIST dataset into the layout the
# pipeline expects:  <out>/<name>/{train,val,test}_{images,labels}.npy
#
# Uses the metadata of the official `medmnist` package (pip install medmnist;
# PyTorch is not needed) and downloads the .npz from Zenodo with an MD5
# check. --npz uses a file you downloaded yourself. Large
# training sets can be capped with a class-stratified subsample; the test
# split is kept complete by default so results stay comparable to the
# MedMNIST benchmark. Multi-label datasets (chestmnist) are rejected, since
# the pipeline is single-label.
#
# Usage (from the repository root):
#   python tools/prepare_dataset.py pathmnist --max-train 20000 --max-val 5000
#   python tools/prepare_dataset.py organamnist
#
# Recommended second dataset: pathmnist (colon pathology, colour, 9 classes,
# 89,996 / 10,004 / 7,180 images). organamnist (abdominal CT, grayscale,
# 11 classes) is a good grayscale alternative.
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import json
import os

import numpy as np


def stratified_cap(labels, n, seed):
    """Indices of a stratified random subset of size n (class proportions kept, at least
    one image per class).
    """
    if n is None or n >= len(labels):
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(labels, return_counts=True)
    exact = counts / counts.sum() * n                  # keep class proportions
    quota = np.maximum(1, np.floor(exact).astype(int))
    for i in np.argsort(-(exact - np.floor(exact))):   # hand out the remainder exactly
        if quota.sum() >= n:
            break
        quota[i] += 1
    idx = []
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(labels == c)
        idx.extend(rng.choice(pool, min(q, len(pool)), replace=False).tolist())
    return np.sort(np.array(idx[:n] if len(idx) > n else idx))


def dataset_info(flag):
    """MedMNIST metadata of a dataset (task, labels, number of samples)."""
    try:
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):     # medmnist prints a torch warning
            from medmnist.info import INFO
    except ImportError:
        raise SystemExit("pip install medmnist   (only its metadata is used; torch is not needed)")
    if flag not in INFO:
        raise SystemExit(f"unknown MedMNIST dataset {flag!r}; options: {sorted(INFO)}")
    info = INFO[flag]
    if info["task"].startswith("multi-label") or "3D" in info.get("python_class", ""):
        raise SystemExit(f"{flag} ({info['task']}, {info['python_class']}) is not supported; "
                         f"this pipeline needs single-label 2D data")
    return info


def fetch_npz(flag, info, cache, size, npz=None):
    """Path of the dataset's .npz, downloading it (with MD5 check) if needed."""
    import hashlib
    import urllib.request
    if npz:
        return npz
    suffix = "" if size == 28 else f"_{size}"
    path = os.path.join(cache, f"{flag}{suffix}.npz")
    md5 = info.get(f"MD5{suffix}")
    if not os.path.exists(path):
        url = info[f"url{suffix}"]
        print(f"downloading {url}\n  -> {path}")
        urllib.request.urlretrieve(url, path + ".part")
        os.replace(path + ".part", path)
    if md5:
        h = hashlib.md5()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() != md5:
            raise SystemExit(f"MD5 mismatch for {path}; delete it and try again")
    return path


def main():
    """Download a MedMNIST dataset, optionally cap train/val, and save
    <out>/<name>/{split}_{images,labels}.npy.
    """
    p = argparse.ArgumentParser(description="download a MedMNIST dataset for the pipeline")
    p.add_argument("name", help="e.g. pathmnist, organamnist, octmnist")
    p.add_argument("--out", default="data", help="folder that will contain <name>/ (default: data)")
    p.add_argument("--cache", default=os.path.expanduser("~/.medmnist"), help="download cache")
    p.add_argument("--size", type=int, default=28, help="image size (28 matches the rest of the project)")
    p.add_argument("--max-train", type=int, default=None, help="stratified cap on training images")
    p.add_argument("--max-val", type=int, default=None, help="stratified cap on validation images")
    p.add_argument("--max-test", type=int, default=None, help="cap on test images (default: keep all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--npz", default=None, help="use an already downloaded MedMNIST .npz file")
    p.add_argument("--force", action="store_true")
    a = p.parse_args()
    ddir = os.path.join(a.out, a.name)
    if os.path.exists(os.path.join(ddir, "test_labels.npy")) and not a.force:
        print(f"{ddir} already prepared (use --force to redo)")
        return
    os.makedirs(ddir, exist_ok=True)
    os.makedirs(a.cache, exist_ok=True)
    caps = {"train": a.max_train, "val": a.max_val, "test": a.max_test}
    meta = {"name": a.name, "size": a.size, "seed": a.seed, "splits": {}}
    info = dataset_info(a.name)
    data = np.load(fetch_npz(a.name, info, a.cache, a.size, a.npz))
    for split in ("train", "val", "test"):
        imgs, labels = data[f"{split}_images"], data[f"{split}_labels"]
        labels = labels.reshape(len(labels), -1)
        if labels.shape[1] != 1:
            raise SystemExit(f"{a.name} labels have shape {labels.shape}; single-label data required")
        keep = stratified_cap(labels[:, 0], caps[split], a.seed)
        np.save(os.path.join(ddir, f"{split}_images.npy"), imgs[keep])
        np.save(os.path.join(ddir, f"{split}_labels.npy"), labels[keep])
        c, n = np.unique(labels[keep, 0], return_counts=True)
        meta["splits"][split] = {"original": int(len(labels)), "kept": int(len(keep)),
                                 "image_shape": list(imgs.shape[1:]),
                                 "class_counts": {int(k): int(v) for k, v in zip(c, n)}}
        print(f"{split:5s}: kept {len(keep)} of {len(labels)} images, shape {imgs.shape[1:]}")
    meta["task"] = info["task"]
    meta["label_names"] = info.get("label")
    with open(os.path.join(ddir, "dataset_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"saved to {ddir}")


if __name__ == "__main__":
    main()
