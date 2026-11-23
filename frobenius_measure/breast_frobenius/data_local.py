"""Offline loader for BreastMNIST stored as .npy files.

Expects in --data-dir:
    train_images.npy / train_labels.npy   (N,28,28) uint8, labels {0,1}
    val_images.npy   / val_labels.npy
    test_images.npy  / test_labels.npy

Preprocessing (identical to the rest of the project):
    channel-select -> resize to size x size -> flatten -> [centre] -> L2-normalise

`merge` is kept only for interface parity with the RetinaMNIST loader --
BreastMNIST is already binary (0=malignant, 1=benign), so "none" is the only
scheme and leaves labels untouched.
"""
from __future__ import annotations

import os

import numpy as np

from config import MERGE_SCHEMES

_EPS = 1e-8


def _apply_merge(y, scheme):
    m = MERGE_SCHEMES.get(scheme)
    if m is None:
        return y
    return np.array([m[int(v)] for v in np.asarray(y).ravel()], dtype=int)


def _to_vec(img, channel, size):
    from PIL import Image
    if img.ndim == 2:
        img = img[..., None]
    nch = img.shape[-1]
    if channel == "green":
        planes = [img[..., 1] if nch >= 3 else img[..., 0]]
    elif channel == "rgb":
        planes = [img[..., i] for i in range(3)] if nch >= 3 else [img[..., 0]]
    else:  # gray
        planes = ([0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]]
                  if nch >= 3 else [img[..., 0]])
    out = []
    for p in planes:
        a = np.asarray(p)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        im = Image.fromarray(a).resize((size, size), Image.LANCZOS)
        out.append(np.asarray(im, dtype=np.float32) / 255.0)
    return np.stack(out, 0).reshape(-1)


def load(data_dir: str, channel: str = "gray", size: int = 16,
         center: bool = True, merge: str = "none"):
    if merge not in MERGE_SCHEMES:
        raise ValueError(f"unknown merge {merge!r}; have {list(MERGE_SCHEMES)}")

    raw = {}
    for split in ("train", "val", "test"):
        ip = os.path.join(data_dir, f"{split}_images.npy")
        lp = os.path.join(data_dir, f"{split}_labels.npy")
        if not os.path.exists(ip):
            raise FileNotFoundError(f"missing {ip}")
        y = _apply_merge(np.load(lp).ravel().astype(int), merge)
        raw[split] = (np.load(ip), y)

    P = {s: np.stack([_to_vec(im, channel, size) for im in raw[s][0]], 0) for s in raw}
    mean = P["train"].mean(0) if center else None

    S = {}
    for s in P:
        X = P[s] - mean if mean is not None else P[s].copy()
        nrm = np.linalg.norm(X, axis=1, keepdims=True)
        S[s] = (X / np.where(nrm < _EPS, 1.0, nrm)).astype(np.float32)

    y_all = np.concatenate([raw[s][1] for s in raw])
    classes = np.unique(y_all)
    remap = {c: i for i, c in enumerate(classes)}
    rl = lambda y: np.array([remap[v] for v in y], dtype=np.int64)   # noqa: E731

    dim = P["train"].shape[1]
    n_qubits = int(np.ceil(np.log2(dim)))
    if (1 << n_qubits) != dim:                       # pad to a power of two
        for s in S:
            pad = np.zeros((S[s].shape[0], (1 << n_qubits) - dim), np.float32)
            S[s] = np.concatenate([S[s], pad], 1)
        dim = 1 << n_qubits

    return {
        "Xtr": S["train"], "ytr": rl(raw["train"][1]),
        "Xval": S["val"], "yval": rl(raw["val"][1]),
        "Xte": S["test"], "yte": rl(raw["test"][1]),
        "n_classes": int(len(classes)), "dim": dim, "n_qubits": n_qubits,
        "chance": 1.0 / len(classes), "side": size, "merge": merge,
    }
