# ----------------------------------------------------------------
# common.py  —  data, features, sampling, metrics, kernels, I/O.
# ----------------------------------------------------------------
import csv
import json
import os

import numpy as np

import config as C


# ----------------------------------------------------------------
# I/O
# ----------------------------------------------------------------

def out_dir(*parts):
    """Create and return config.OUT_ROOT/<parts...>."""
    path = os.path.join(C.OUT_ROOT, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(o):
    """JSON fallback for NumPy scalars/arrays (non-finite floats become null)."""
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o) if np.isfinite(o) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def save_json(obj, path):
    """Write obj as indented JSON (NumPy types converted)."""
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=_json_default)


def load_json(path):
    """Read a JSON file."""
    with open(path) as f:
        return json.load(f)


def write_csv(rows, path):
    """Write a list of dicts as CSV; the columns are the union of all keys, in order of
    appearance.
    """
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in keys})


def get_plt():
    """matplotlib.pyplot with the non-interactive Agg backend (no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------

def load_split(dataset, split):
    """Images and integer labels of one split from data/<dataset>/, or (None, None) if the
    split does not exist. Multi-label datasets are rejected.
    """
    ddir = os.path.join(C.DATA_ROOT, dataset)
    ip, lp = (os.path.join(ddir, f"{split}_{k}.npy") for k in ("images", "labels"))
    if not os.path.exists(ip):
        return None, None
    if not os.path.exists(lp):
        raise FileNotFoundError(f"{lp} missing")
    labels = np.load(lp)
    if labels.ndim == 2 and labels.shape[1] > 1:
        raise ValueError(f"{dataset}: multi-label labels {labels.shape} are not supported")
    return np.load(ip), labels.reshape(-1).astype(int)


def check_datasets(names):
    """Exit with a clear message if any dataset is missing from config.DATA_ROOT."""
    for d in names:
        p = os.path.join(C.DATA_ROOT, d, "train_images.npy")
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found (run from the repository root with the datasets in data/, or fix config.DATA_ROOT)")


def stratified_sample(labels, n, rng):
    """Class-balanced sample of size n; shortfall of small classes filled at random."""
    n = min(n, len(labels))
    classes = np.unique(labels)
    per = n // len(classes)
    chosen = []
    for c in classes:
        idx = np.flatnonzero(labels == c)
        chosen.extend(rng.choice(idx, min(per, len(idx)), replace=False).tolist())
    extra = n - len(chosen)
    if extra > 0:
        rest = np.setdiff1d(np.arange(len(labels)), chosen)
        chosen.extend(rng.choice(rest, extra, replace=False).tolist())
    chosen = np.array(chosen, dtype=int)
    rng.shuffle(chosen)
    return chosen


def class_counts(labels):
    """{class label: number of samples}."""
    c, n = np.unique(labels, return_counts=True)
    return {int(a): int(b) for a, b in zip(c, n)}


# ----------------------------------------------------------------
# Features
# ----------------------------------------------------------------

def _to_float(imgs):
    """Images as float64 in [0, 1] with an explicit channel axis (N, H, W, ch)."""
    x = imgs.astype(np.float64)
    if x.max() > 1.0:
        x = x / 255.0
    if x.ndim == 3:
        x = x[..., None]
    return x                                               # (N, H, W, ch)


def _dct_matrix(n):
    """Orthonormal DCT-II matrix of size n x n (rows = frequencies)."""
    k = np.arange(n)[:, None]
    i = np.arange(n)[None, :]
    M = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * np.sqrt(2.0 / n)
    M[0] /= np.sqrt(2.0)
    return M                                               # orthonormal DCT-II


def raw_candidates(imgs, rgb):
    """All candidate features before ranking. Returns (X, names)."""
    x = _to_float(imgs)
    if not rgb:
        x = x.mean(axis=-1, keepdims=True)
    n, h, w, ch = x.shape
    if C.FEATURE_METHOD == "dct":
        Mh, Mw = _dct_matrix(h), _dct_matrix(w)
        coef = np.einsum("ui,nijc,vj->ncuv", Mh, x, Mw, optimize=True)
        order = C.DCT_MAX_ORDER
        while ch * (order + 1) * (order + 2) // 2 < C.MAX_FEATURES and order < min(h, w) - 1:
            order += 1                     # grayscale needs a higher order for the same pool size
        uv = [(u, v) for u in range(min(h, order + 1))
              for v in range(min(w, order + 1)) if u + v <= order]
        cols, names = [], []
        chn = "RGB" if ch == 3 else ["gray"]
        for c in range(ch):
            for u, v in uv:
                cols.append(coef[:, c, u, v])
                names.append(f"{chn[c]}_dct_{u}_{v}")
        return np.stack(cols, axis=1), names
    if C.FEATURE_METHOD == "pca":
        return x.reshape(n, -1), None
    raise ValueError(f"unknown FEATURE_METHOD {C.FEATURE_METHOD!r}")


def fisher_scores(X, y):
    """Per-feature Fisher score: variance of the class means / mean within-class variance."""
    classes = np.unique(y)
    mu_c = np.stack([X[y == c].mean(0) for c in classes])
    between = ((mu_c - mu_c.mean(0)) ** 2).mean(0)
    within = np.stack([X[y == c].var(0) for c in classes]).mean(0)
    return between / (within + 1e-12)


def build_features(dataset):
    """Fits the feature pipeline on TRAIN and returns angle-scaled features
    for every split plus metadata. Columns are ordered by Fisher score, so a
    genome with d features uses columns [0, d)."""
    tr_x, tr_y = load_split(dataset, "train")
    rgb = C.USE_RGB and tr_x.ndim == 4 and tr_x.shape[-1] == 3
    Xtr_raw, names = raw_candidates(tr_x, rgb)
    meta = {"dataset": dataset, "method": C.FEATURE_METHOD, "rgb": bool(rgb),
            "dct_max_order": C.DCT_MAX_ORDER, "angle_clip": C.ANGLE_CLIP,
            "angle_range": C.ANGLE_RANGE}

    pca = None
    if C.FEATURE_METHOD == "pca":
        from sklearn.decomposition import PCA
        pca = PCA(n_components=min(C.MAX_FEATURES * 2, Xtr_raw.shape[1]), random_state=C.DATA_SEED)
        Xtr_raw = pca.fit_transform(Xtr_raw)
        names = [f"pc{i}" for i in range(Xtr_raw.shape[1])]

    mean = Xtr_raw.mean(0)
    std = Xtr_raw.std(0) + 1e-12
    Z = (Xtr_raw - mean) / std
    fs = fisher_scores(Z, tr_y)
    order = np.argsort(-fs)[:C.MAX_FEATURES]
    if Xtr_raw.shape[1] < C.MAX_FEATURES:
        raise ValueError(f"only {Xtr_raw.shape[1]} candidate features, need MAX_FEATURES={C.MAX_FEATURES}")
    meta.update({"feature_names": [names[i] for i in order],
                 "fisher": fs[order].tolist(), "columns": order.tolist(),
                 "mean": mean[order].tolist(), "std": std[order].tolist()})

    def transform(imgs):
        """Images -> standardised features, clipped and scaled to encoding angles."""
        R, _ = raw_candidates(imgs, rgb)
        if pca is not None:
            R = pca.transform(R)
        z = (R[:, order] - mean[order]) / std[order]
        return np.clip(z, -C.ANGLE_CLIP, C.ANGLE_CLIP) * (C.ANGLE_RANGE / C.ANGLE_CLIP)

    feats = {"train": (transform(tr_x), tr_y)}
    for split in ("val", "test"):
        x, y = load_split(dataset, split)
        if x is not None:
            feats[split] = (transform(x), y)
    return feats, meta


def load_or_build_features(dataset, force=False):
    """Feature matrices for all splits of a dataset, cached in
    <OUT_ROOT>/<dataset>/features/ and rebuilt when the feature settings in config.py
    change (or with force=True).
    """
    fdir = out_dir(dataset, "features")
    mpath = os.path.join(fdir, "feature_meta.json")
    snapshot = {k: getattr(C, k) for k in ("FEATURE_METHOD", "DCT_MAX_ORDER", "USE_RGB",
                                           "MAX_FEATURES", "ANGLE_CLIP", "ANGLE_RANGE", "DATA_SEED")}
    snapshot = json.loads(json.dumps(snapshot))
    if os.path.exists(mpath) and not force:
        meta = load_json(mpath)
        if meta.get("config") == snapshot:
            feats = {}
            for split in ("train", "val", "test"):
                p = os.path.join(fdir, f"{split}_features.npy")
                if os.path.exists(p):
                    feats[split] = (np.load(p), np.load(os.path.join(fdir, f"{split}_labels.npy")))
            return feats, meta
    feats, meta = build_features(dataset)
    meta["config"] = snapshot
    meta["shapes"] = {s: list(v[0].shape) for s, v in feats.items()}
    for split, (X, y) in feats.items():
        np.save(os.path.join(fdir, f"{split}_features.npy"), X)
        np.save(os.path.join(fdir, f"{split}_labels.npy"), y)
    save_json(meta, mpath)
    return feats, meta


# ----------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------

def macro_ovr_auc(y, scores, classes):
    """ROC AUC: binary AUC for two classes, otherwise the macro average of one-vs-rest AUCs
    (NaN if a class is missing from y).
    """
    from sklearn.metrics import roc_auc_score
    classes = list(classes)
    if len(classes) == 2:
        s = scores if scores.ndim == 1 else scores[:, 1]
        return float(roc_auc_score(y == classes[1], s)) if len(np.unique(y)) == 2 else float("nan")
    aucs = []
    for i, c in enumerate(classes):
        pos = y == c
        if pos.any() and not pos.all():
            aucs.append(roc_auc_score(pos, scores[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def summarise(y, pred, scores, classes):
    """Balanced accuracy, AUC, chance level and per-class recall of a set of predictions."""
    from sklearn.metrics import balanced_accuracy_score
    recall = {int(c): (float((pred[y == c] == c).mean()) if (y == c).any() else None)
              for c in classes}
    return {"bal_acc": float(balanced_accuracy_score(y, pred)),
            "auc": macro_ovr_auc(y, scores, classes),
            "chance": 1.0 / len(classes), "recall": recall}


# ----------------------------------------------------------------
# Kernels and models
# ----------------------------------------------------------------

def fidelity_kernel(A, B):
    """Fidelity kernel |<a_i|b_j>|^2 between two sets of pure states (rows)."""
    return np.abs(A.conj() @ B.T) ** 2


def sqdist(A, B):
    """Matrix of squared Euclidean distances between the rows of A and B."""
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def median_gamma(A):
    """RBF bandwidth 1 / median squared distance (median heuristic)."""
    d = sqdist(A, A)
    m = np.median(d[~np.eye(len(A), dtype=bool)])
    return 1.0 / m if m > 0 else 1.0


def rbf_kernel(A, B, gamma):
    """Gaussian kernel exp(-gamma |a - b|^2)."""
    return np.exp(-gamma * sqdist(A, B))


def make_folds(y, k, seed):
    """Stratified k-fold (train, test) index pairs; k is reduced to the smallest class size."""
    from sklearn.model_selection import StratifiedKFold
    k = int(min(k, np.unique(y, return_counts=True)[1].min()))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def cv_kernel_score(K, y, Cval, folds):
    """Mean cross-validated AUC (and its std) and balanced accuracy of a balanced SVM with
    precomputed kernel K and regularisation Cval.
    """
    from sklearn.svm import SVC
    aucs, bals = [], []
    for tr, te in folds:
        clf = SVC(kernel="precomputed", C=Cval, class_weight="balanced")
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        Kt = K[np.ix_(te, tr)]
        s = summarise(y[te], clf.predict(Kt), clf.decision_function(Kt), clf.classes_)
        aucs.append(s["auc"])
        bals.append(s["bal_acc"])
    return {"auc": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
            "bal_acc": float(np.mean(bals))}


def best_cv_kernel_score(K, y, folds, Cs=None):
    """Cross-validated score for every C in the grid; returns the best (highest AUC, then
    balanced accuracy) together with its C.
    """
    Cs = Cs or C.SVM_C_GRID
    scores = {c: cv_kernel_score(K, y, c, folds) for c in Cs}
    best = max(Cs, key=lambda c: (scores[c]["auc"], scores[c]["bal_acc"]))
    return {"C": float(best), **scores[best]}


def fit_kernel_svm(Ktr, ytr, folds=None):
    """C by CV on the training kernel, then refit on all training data."""
    from sklearn.svm import SVC
    folds = folds or make_folds(ytr, 3, C.DATA_SEED)
    best = best_cv_kernel_score(Ktr, ytr, folds)
    clf = SVC(kernel="precomputed", C=best["C"], class_weight="balanced").fit(Ktr, ytr)
    return clf, best["C"]


def eval_kernel_svm(clf, Kte, yte):
    """Test metrics of a fitted precomputed-kernel SVM (Kte = test x train kernel)."""
    return summarise(yte, clf.predict(Kte), clf.decision_function(Kte), clf.classes_)


def classical_models(Xtr, ytr, Xte, yte):
    """Logistic regression and RBF SVM on the same features."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(Xtr)
    A, B = sc.transform(Xtr), sc.transform(Xte)
    out = {}
    lr = LogisticRegression(max_iter=3000, class_weight="balanced").fit(A, ytr)
    out["logreg"] = summarise(yte, lr.predict(B), lr.predict_proba(B), lr.classes_)
    gs = GridSearchCV(SVC(kernel="rbf", gamma="scale", class_weight="balanced"),
                      {"C": list(C.SVM_C_GRID)}, cv=make_folds(ytr, 3, C.DATA_SEED),
                      scoring="balanced_accuracy").fit(A, ytr)
    m = gs.best_estimator_
    out["rbf_svm"] = summarise(yte, m.predict(B), m.decision_function(B), m.classes_)
    out["rbf_svm"]["C"] = float(gs.best_params_["C"])
    return out
