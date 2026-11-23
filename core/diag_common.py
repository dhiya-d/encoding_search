# ----------------------------------------------------------------
# diag_common.py  —  Shared helpers for the three cheap diagnostics.
#
#   diag_a_phase_check.py   (a) global sign / phase artefact in the probe
#   diag_b_ceiling.py       (b) phase-invariant ceiling of exact amplitude
#                               encoding (what ANY AAE circuit can reach)
#   diag_c_spectral.py      (c) spatial Fourier / Walsh map of where the
#                               class information lives
#
# Preprocessing is kept identical to the benchmark repo's data_prep.py:
#   grayscale (channel mean, /255) -> separable linear resize to 16x16
#   -> flatten row-major -> [train-mean centering] -> [train-std
#   whitening] -> L2 normalise to a unit 256-amplitude vector.
# The resize here is a vectorised but numerically identical version of
# the repo's per-row / per-column np.interp loop.
#
# Row-major flattening + PennyLane wire order (wire 0 = most significant
# bit) means wires 0-3 index the image ROW and wires 4-7 the COLUMN, so
# the amplitude vector is a function on the group Z_16 x Z_16.
#
# Output layout (everything for a dataset stays under its own folder):
#   experiments/<dataset>/<diagnostic>/<mode>/...      per-mode results
#   experiments/<dataset>/summary_<diagnostic>.csv     per-dataset table
#   experiments/summary_<diagnostic>.csv               cross-dataset table
# ----------------------------------------------------------------

import csv
import json
import os

import numpy as np

IMG_SIDE = 16
N_QUBITS = 8
STATE_DIM = IMG_SIDE * IMG_SIDE          # 256 = 2**8
MODES = ("uncentered", "centered", "whitened")


# ----------------------------------------------------------------
# CLI helpers
# ----------------------------------------------------------------

def add_common_args(parser):
    """Command-line options shared by the three diagnostics."""
    parser.add_argument("--data-root", default="data",
                        help="folder containing <dataset>/train_images.npy etc. "
                             "(default: data)")
    parser.add_argument("--out", default="experiments",
                        help="root output folder (default: experiments)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="datasets to run (default: every subfolder of "
                             "--data-root that contains train_images.npy)")
    parser.add_argument("--modes", nargs="*", default=list(MODES),
                        choices=list(MODES), help="preprocessing modes")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true",
                        help="recompute combos that already have results.json")
    return parser


def find_selected_genome_file(root, dataset, mode, kta=False):
    """Locate a selected genome for (dataset, mode). Two layouts:
      new  (encoding_search.py / select_encoder.py):
           <root>/<dataset>/search/<mode>/selected_genome.json
      old  (benchmark repo; whitened runs live in the centered folder):
           <root>/<dataset>_<uncentered|centered>/selected_genome[_kta].json
    Returns (path or None, list of paths tried)."""
    tried = []
    if not kta:
        tried.append(os.path.join(root, dataset, "search", mode, "selected_genome.json"))
    old_mode = "uncentered" if mode == "uncentered" else "centered"
    fname = "selected_genome_kta.json" if kta else "selected_genome.json"
    tried.append(os.path.join(root, f"{dataset}_{old_mode}", fname))
    for p in tried:
        if os.path.exists(p):
            return p, tried
    return None, tried


def check_genome_root(genome_root):
    """Fail loudly on a wrong --genome-root instead of silently falling back
    to DEFAULT_GENOME. Lists what it found so the right folder is obvious."""
    if genome_root is None:
        return
    if not os.path.isdir(genome_root):
        raise SystemExit(f"--genome-root {genome_root!r} does not exist. Use the search output "
                         f"folder (e.g. experiments) or the repo's experiments/encoding_benchmark.")
    import glob
    found = sorted(glob.glob(os.path.join(genome_root, "*", "search", "*", "selected_genome.json"))
                   + glob.glob(os.path.join(genome_root, "*_*", "selected_genome*.json")))
    if not found:
        raise SystemExit(f"--genome-root {genome_root!r} contains no selected genomes "
                         f"(<dataset>/search/<mode>/selected_genome.json or "
                         f"<dataset>_<mode>/selected_genome*.json)")
    rel = [os.path.relpath(f, genome_root) for f in found]
    print(f"genome root: {os.path.abspath(genome_root)}  ({len(found)} genomes: {rel})")


def resolve_datasets(data_root, requested):
    """Requested datasets (checked to exist), or every dataset found in data_root."""
    if requested:
        for d in requested:
            p = os.path.join(data_root, d, "train_images.npy")
            if not os.path.exists(p):
                raise FileNotFoundError(f"{p} not found")
        return list(requested)
    found = []
    for d in sorted(os.listdir(data_root)):
        if os.path.exists(os.path.join(data_root, d, "train_images.npy")):
            found.append(d)
    if not found:
        raise FileNotFoundError(
            f"No <dataset>/train_images.npy folders found under {os.path.abspath(data_root)}")
    return found


def out_dir(root, dataset, diagnostic, mode=None):
    """Create and return <root>/<dataset>/<diagnostic>[/<mode>]."""
    parts = [root, dataset, diagnostic] + ([mode] if mode else [])
    path = os.path.join(*parts)
    os.makedirs(path, exist_ok=True)
    return path


def _json_default(o):
    """JSON fallback for NumPy scalars/arrays (non-finite floats become null)."""
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
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


def collect_summaries(out_root, diagnostic):
    """Rebuild per-dataset and cross-dataset CSVs from every saved
    summary_row.json under out_root, so partial / resumed runs and runs
    on different dataset subsets always produce complete tables."""
    all_rows = []
    if not os.path.isdir(out_root):
        return all_rows
    for ds in sorted(os.listdir(out_root)):
        rows = []
        for mode in MODES:
            p = os.path.join(out_root, ds, diagnostic, mode, "summary_row.json")
            if os.path.exists(p):
                rows.append(load_json(p))
        if rows:
            write_csv(rows, os.path.join(out_root, ds, f"summary_{diagnostic}.csv"))
            all_rows.extend(rows)
    if all_rows:
        write_csv(all_rows, os.path.join(out_root, f"summary_{diagnostic}.csv"))
    return all_rows


# ----------------------------------------------------------------
# Data
# ----------------------------------------------------------------

def load_split(data_root, dataset, split):
    """Images and integer labels of one split, or (None, None) if the split does not exist.
    Multi-label datasets are rejected.
    """
    ddir = os.path.join(data_root, dataset)
    ip = os.path.join(ddir, f"{split}_images.npy")
    lp = os.path.join(ddir, f"{split}_labels.npy")
    if not os.path.exists(ip):
        return None, None
    imgs = np.load(ip)
    if not os.path.exists(lp):
        raise FileNotFoundError(f"{lp} missing — labels are required")
    labels = np.load(lp)
    if labels.ndim == 2 and labels.shape[1] > 1:
        raise ValueError(
            f"{dataset} labels have shape {labels.shape} (multi-label); these "
            f"diagnostics expect single-label multiclass data")
    return imgs, labels.reshape(-1).astype(int)


def load_train_eval(data_root, dataset):
    """Train split for fitting; test split for evaluation (val if no test)."""
    tr_x, tr_y = load_split(data_root, dataset, "train")
    ev_x, ev_y = load_split(data_root, dataset, "test")
    ev_name = "test"
    if ev_x is None:
        ev_x, ev_y = load_split(data_root, dataset, "val")
        ev_name = "val"
    if ev_x is None:
        raise FileNotFoundError(f"{dataset}: no test_ or val_ split found")
    return tr_x, tr_y, ev_x, ev_y, ev_name


def to_grayscale(imgs):
    """(N,H,W,C) or (N,H,W) -> (N,H,W) float64 in [0,1]; same as repo."""
    imgs = imgs.astype(np.float64)
    if imgs.ndim == 4:
        imgs = imgs.mean(axis=-1)
    elif imgs.ndim != 3:
        raise ValueError(f"Expected (N,H,W) or (N,H,W,C), got {imgs.shape}")
    if imgs.max() > 1.0:
        imgs = imgs / 255.0
    return imgs


def _interp_matrix(n_in, side):
    """Linear map equal to np.interp(linspace(0,1,side), linspace(0,1,n_in), .)"""
    x_old = np.linspace(0.0, 1.0, n_in)
    x_new = np.linspace(0.0, 1.0, side)
    eye = np.eye(n_in)
    return np.stack([np.interp(x_new, x_old, eye[j]) for j in range(n_in)], axis=1)


def images_to_vectors(imgs, side=IMG_SIDE):
    """(N,H,W[,C]) -> (N, side*side) raw (unnormalised) pixel vectors."""
    g = to_grayscale(imgs)
    _, h, w = g.shape
    Mh, Mw = _interp_matrix(h, side), _interp_matrix(w, side)
    out = np.einsum("ri,nij,cj->nrc", Mh, g, Mw, optimize=True)
    return out.reshape(len(g), side * side)


def is_rgb(imgs):
    """True for colour image arrays (N, H, W, 3)."""
    return imgs.ndim == 4 and imgs.shape[-1] == 3


def to_rgb_float(imgs):
    """(N,H,W,3) -> float64 in [0,1], same scaling rule as to_grayscale."""
    x = imgs.astype(np.float64)
    if x.max() > 1.0:
        x = x / 255.0
    return x


def images_to_rgb_vectors(imgs, side=IMG_SIDE):
    """(N,H,W,3) -> (N, 3*side*side): each channel resized exactly like the
    grayscale pipeline (same separable linear interpolation), then
    concatenated R, G, B. Classical reference only — an 8-qubit amplitude
    encoding cannot hold 768 values."""
    x = to_rgb_float(imgs)
    _, h, w, _ = x.shape
    Mh, Mw = _interp_matrix(h, side), _interp_matrix(w, side)
    out = np.einsum("ri,nijk,cj->nkrc", Mh, x, Mw, optimize=True)
    return out.reshape(len(x), -1)


def normalise(v):
    """Scale every row to unit Euclidean norm (zero rows are left unchanged)."""
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return v / n


def compute_stats(train_vecs, mode):
    """Statistics always from the FULL train split (as in the repo)."""
    if mode == "uncentered":
        return None, None
    mean = train_vecs.mean(axis=0)
    std = train_vecs.std(axis=0) + 1e-8 if mode == "whitened" else None
    return mean, std


def apply_stats(vecs, mean, std):
    """Subtract the mean and divide by the std (each if given), then normalise every row ->
    amplitude-encoding targets.
    """
    if mean is not None:
        vecs = vecs - mean[None, :]
    if std is not None:
        vecs = vecs / std[None, :]
    return normalise(vecs)


def stratified_sample(labels, n, rng):
    """Class-balanced sample of size n (capped by availability; any
    shortfall from small classes is filled at random from the rest)."""
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
    chosen = np.array(chosen)
    rng.shuffle(chosen)
    return chosen


def class_counts(labels):
    """{class label: number of samples}."""
    c, n = np.unique(labels, return_counts=True)
    return {int(a): int(b) for a, b in zip(c, n)}


# ----------------------------------------------------------------
# Metrics and models
# ----------------------------------------------------------------

def macro_ovr_auc(y, scores, classes):
    """ROC AUC: binary AUC for two classes, otherwise the macro average of one-vs-rest AUCs
    (NaN if a class is missing from y).
    """
    from sklearn.metrics import roc_auc_score
    classes = list(classes)
    if len(classes) == 2:
        s = scores if scores.ndim == 1 else scores[:, 1]
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y == classes[1], s))
    aucs = []
    for i, c in enumerate(classes):
        pos = y == c
        if pos.all() or (~pos).all():
            continue
        aucs.append(roc_auc_score(pos, scores[:, i]))
    return float(np.mean(aucs)) if aucs else float("nan")


def summarise_predictions(y, pred, scores, classes):
    """Balanced accuracy, AUC, chance level and per-class recall."""
    from sklearn.metrics import balanced_accuracy_score
    recall = {}
    for c in classes:
        m = y == c
        recall[int(c)] = float((pred[m] == c).mean()) if m.any() else None
    return {
        "bal_acc": float(balanced_accuracy_score(y, pred)),
        "auc": macro_ovr_auc(y, scores, classes),
        "chance": 1.0 / len(classes),
        "recall": recall,
    }


def linear_probe(Xtr, ytr, Xte, yte, C=1.0):
    """Standardised multinomial logistic regression (the repo's probe)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=3000, C=C, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    Xt = sc.transform(Xte)
    return summarise_predictions(yte, clf.predict(Xt), clf.predict_proba(Xt), clf.classes_)


def _cv_splits(y):
    """Stratified 3-fold splitter for the linear probe (None if a class has < 2 samples)."""
    from sklearn.model_selection import StratifiedKFold
    k = int(min(3, np.unique(y, return_counts=True)[1].min()))
    return StratifiedKFold(k, shuffle=True, random_state=0) if k >= 2 else None


def kernel_svm(Ktr, ytr, Kte, yte, Cs=(0.1, 1.0, 10.0, 100.0)):
    """SVM on a precomputed kernel, C chosen by stratified CV on train."""
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVC
    base = SVC(kernel="precomputed", class_weight="balanced")
    cv = _cv_splits(ytr)
    if cv is None:
        clf = base.set_params(C=1.0).fit(Ktr, ytr)
        best_C = 1.0
    else:
        gs = GridSearchCV(base, {"C": list(Cs)}, cv=cv, scoring="balanced_accuracy")
        gs.fit(Ktr, ytr)
        clf, best_C = gs.best_estimator_, gs.best_params_["C"]
    out = summarise_predictions(yte, clf.predict(Kte), clf.decision_function(Kte), clf.classes_)
    out["C"] = float(best_C)
    return out


def make_folds(y, k, seed):
    """Stratified k-fold (train, test) index pairs; k is reduced to the smallest class size."""
    from sklearn.model_selection import StratifiedKFold
    k = int(min(k, np.unique(y, return_counts=True)[1].min()))
    return list(StratifiedKFold(k, shuffle=True, random_state=seed).split(np.zeros(len(y)), y))


def cv_kernel_score(K, y, C, folds):
    """Stratified-CV score of an SVM on a precomputed kernel. Label-free
    encodings can be scored this way without leakage: labels only enter
    through the SVM, which is refit inside every fold."""
    from sklearn.svm import SVC
    aucs, bals = [], []
    for tr, te in folds:
        clf = SVC(kernel="precomputed", C=C, class_weight="balanced")
        clf.fit(K[np.ix_(tr, tr)], y[tr])
        Kte = K[np.ix_(te, tr)]
        s = summarise_predictions(y[te], clf.predict(Kte), clf.decision_function(Kte), clf.classes_)
        aucs.append(s["auc"])
        bals.append(s["bal_acc"])
    return {"auc": float(np.nanmean(aucs)), "auc_std": float(np.nanstd(aucs)),
            "bal_acc": float(np.mean(bals)), "bal_acc_std": float(np.std(bals))}


def choose_C(K, y, Cs, folds):
    """Cross-validated score for every C; returns the best C (highest AUC, then balanced
    accuracy) and all scores.
    """
    scores = {C: cv_kernel_score(K, y, C, folds) for C in Cs}
    best = max(Cs, key=lambda C: (scores[C]["auc"], scores[C]["bal_acc"]))
    return best, scores


def rbf_svm(Xtr, ytr, Xte, yte, Cs=(0.1, 1.0, 10.0, 100.0)):
    """Classical baseline: standardised features, RBF SVM with C chosen by grid search;
    returns test metrics.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    sc = StandardScaler().fit(Xtr)
    Xtr, Xte = sc.transform(Xtr), sc.transform(Xte)
    base = SVC(kernel="rbf", gamma="scale", class_weight="balanced")
    cv = _cv_splits(ytr)
    if cv is None:
        clf, best_C = base.fit(Xtr, ytr), 1.0
    else:
        gs = GridSearchCV(base, {"C": list(Cs)}, cv=cv, scoring="balanced_accuracy")
        gs.fit(Xtr, ytr)
        clf, best_C = gs.best_estimator_, gs.best_params_["C"]
    out = summarise_predictions(yte, clf.predict(Xte), clf.decision_function(Xte), clf.classes_)
    out["C"] = float(best_C)
    return out


def amp_features(S):
    """Amplitudes as real features (real/imag concatenated if complex)."""
    if np.iscomplexobj(S):
        return np.concatenate([S.real, S.imag], axis=1)
    return S


def fidelity_kernel(A, B):
    """K_ij = |<a_i|b_j>|^2 — the kernel any expectation-value model on
    these states is built from; invariant to per-state global phase."""
    return np.abs(A.conj() @ B.T) ** 2


def kernel_stats(K):
    """Mean, std, min and max of the off-diagonal kernel entries (concentration check)."""
    off = K[~np.eye(len(K), dtype=bool)]
    return {"offdiag_mean": float(off.mean()), "offdiag_std": float(off.std()),
            "offdiag_min": float(off.min()), "offdiag_max": float(off.max())}


def kta(K, y):
    """Kernel-target alignment, same definition as the repo:
    <K,T>_F / (||K||_F ||T||_F), T_ij = +1 same class, -1 otherwise.
    Uses <K,T> = 2 tr(Y^T K Y) - sum(K) and ||T||_F = N."""
    _, yi = np.unique(y, return_inverse=True)
    Y = np.eye(yi.max() + 1)[yi]
    same = np.trace(Y.T @ K @ Y)
    return float((2 * same - K.sum()) / (np.linalg.norm(K) * len(y)))


def kta_with_null(K, y, n_perm, rng):
    """KTA minus the label-permutation null mean (repo convention)."""
    true = kta(K, y)
    null = np.array([kta(K, rng.permutation(y)) for _ in range(n_perm)])
    sd = null.std()
    return {"kta": true, "null_mean": float(null.mean()), "null_std": float(sd),
            "kta_above_null": float(true - null.mean()),
            "z_score": float((true - null.mean()) / sd) if sd > 0 else float("nan")}


def get_plt():
    """matplotlib.pyplot with the non-interactive Agg backend (no display needed)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt
