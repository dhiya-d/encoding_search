# ----------------------------------------------------------------
# classifiers.py  —  measurement-based quantum classifiers + training.
#
#   hea   : L layers of [RY, RZ on every qubit, CNOT ring]; readout
#           <Z_q>, <Z_q Z_q+1> -> linear layer (a trainable observable)
#   qcnn  : hierarqcal-style QCNN: convolution blocks on Qcycle pairs of
#           the active qubits, pooling halves the active qubits, until
#           `final` qubits remain; readout = their basis probabilities
#           -> linear layer
#   reupload (Fourier encodings only, see encoding_genome.classifier_circuit)
#
# All circuits can start from |0> (then prepend the encoding ops) or from
# given input states (amplitude-encoded images).
# ----------------------------------------------------------------
import numpy as np

import common as U
import config as C
from encoding_genome import edges
from qsim import Circuit, basis_projectors


def hea_ops(p0=0, layers=4, n=C.N_QUBITS):
    """Hardware-efficient classifier: `layers` x (RY RZ on every qubit + CNOT ring).
    Returns (ops, next free parameter index).
    """
    ops, p = [], p0
    for _ in range(layers):
        for q in range(n):
            ops += [("R", "Y", q, p), ("R", "Z", q, p + 1)]
            p += 2
        ops += [("CNOT", a, b) for a, b in edges(1, 1, 0, 2, n)]
    for q in range(n):
        ops.append(("R", "Y", q, p))
        p += 1
    return ops, p


def _cycle(active, stride=1):
    """Nearest-neighbour ring (given stride) on the currently active qubits, via
    hierarqcal.
    """
    from hierarqcal import Qcycle, Qunitary
    m = Qcycle(stride=stride, step=1, offset=0,
               mapping=Qunitary(lambda bits, symbols=None: None, n_symbols=0, arity=2))
    m(list(active))
    return [tuple(int(q) for q in e) for e in m.E]


def qcnn_ops(p0=0, final=2, n=C.N_QUBITS):
    """Returns (ops, next param index, final active qubits)."""
    ops, p, active = [], p0, list(range(n))

    def conv(pairs):
        """Two-qubit convolution block (RY RY CNOT RY RY CNOT) on every pair."""
        nonlocal p
        for a, b in pairs:
            ops.extend([("R", "Y", a, p), ("R", "Y", b, p + 1), ("CNOT", a, b),
                        ("R", "Y", a, p + 2), ("R", "Y", b, p + 3), ("CNOT", b, a)])
            p += 4

    while len(active) > final:
        conv(_cycle(active))
        keep, drop = active[0::2], active[1::2]
        for k, d in zip(keep, drop):                  # pooling: fold dropped qubit into kept one
            ops.extend([("R", "Y", d, p), ("CNOT", d, k), ("R", "Y", k, p + 1)])
            p += 2
        active = keep
    if len(active) >= 2:
        conv(_cycle(active))
    for q in active:
        ops.append(("R", "Y", q, p))
        p += 1
    return ops, p, active


def qcnn_final_qubits(n_classes):
    """Number of qubits left after pooling: 4 for more than four classes, else 2."""
    return 4 if n_classes > 4 else 2


def reupload_ops(encoding_ops, reuploads=None, n=C.N_QUBITS):
    """Schuld et al. layout: W0, then [S(x), W_r] for r = 1..reuploads."""
    reuploads = C.VQC_REUPLOADS if reuploads is None else reuploads

    def block(p):
        """One trainable block: RY RZ on every qubit followed by a CNOT ring."""
        ops = []
        for q in range(n):
            ops += [("R", "Y", q, p), ("R", "Z", q, p + 1)]
            p += 2
        return ops + [("CNOT", a, b) for a, b in edges(1, 1, 0, 2, n)], p

    ops, p = block(0)
    for _ in range(reuploads):
        blk, p = block(p)
        ops += list(encoding_ops) + blk
    return ops


def spec_version(kind):
    """Bumped whenever a classifier's definition changes; cached results with
    another version are recomputed. v2: scaled probability readout."""
    return 2 if kind.endswith("_probs") else 1


def training_signature():
    """Training settings that must match for a cached result to be reused."""
    return {"epochs": C.VQC_EPOCHS, "patience": C.VQC_PATIENCE, "batch": C.VQC_BATCH,
            "lr": C.VQC_LR, "seeds": list(C.CLF_SEEDS), "weight_decay": C.CLF_WEIGHT_DECAY}


def result_is_current(res, kind):
    """Cached classifier result usable? Needs the same definition, the same
    training settings and saved test predictions (for paired statistics)."""
    return (res is not None
            and res.get("spec_version", 1) == spec_version(kind)
            and res.get("training") == training_signature()
            and all("test_proba" in r for r in res.get("runs", [])))


def build(kind, n_classes, prefix_ops=()):
    """Classifier circuit. prefix_ops = encoding ops (run once in front, or
    re-uploaded for kind='reupload').
    HEA variants: 'hea' (CLF_HEA_LAYERS layers) or 'hea<L>' (L layers), with
    suffix '_probs' for a readout of all 2^n basis probabilities (one
    measurement setting) instead of <Z_q>, <Z_q Z_q+1>."""
    if kind == "reupload":
        return Circuit(reupload_ops(prefix_ops))
    if kind.startswith("hea"):
        base, _, suffix = kind.partition("_")
        layers = int(base[3:]) if base[3:] else C.CLF_HEA_LAYERS
        ops, _ = hea_ops(0, layers)
        # scaled by 2^n so features average 1 (unscaled 1/256-sized features
        # plus weight decay kept the readout too small to learn)
        readout = (basis_projectors(list(range(C.N_QUBITS))) * 2 ** C.N_QUBITS
                   if suffix == "probs" else None)
        if suffix not in ("", "probs"):
            raise ValueError(kind)
        return Circuit(list(prefix_ops) + ops, readout=readout)
    if kind == "qcnn":
        ops, _, active = qcnn_ops(0, qcnn_final_qubits(n_classes))
        return Circuit(list(prefix_ops) + ops, readout=basis_projectors(active))
    raise ValueError(kind)


# ----------------------------------------------------------------
# Training
# ----------------------------------------------------------------

class Adam:
    """Minimal Adam optimiser for a list of NumPy parameter arrays (updated in place)."""
    def __init__(self, shapes, lr):
        """Zero moment estimates for arrays of the given shapes."""
        self.m = [np.zeros(s) for s in shapes]
        self.v = [np.zeros(s) for s in shapes]
        self.lr, self.t = lr, 0

    def step(self, params, grads):
        """One bias-corrected Adam update of every array in `params`."""
        self.t += 1
        for i, (x, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mh = self.m[i] / (1 - 0.9 ** self.t)
            vh = self.v[i] / (1 - 0.999 ** self.t)
            x -= self.lr * mh / (np.sqrt(vh) + 1e-8)


def _softmax(z):
    """Row-wise numerically stable softmax."""
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _rows(idx, n_traj):
    """Row indices of samples idx when each sample has n_traj consecutive rows."""
    idx = np.asarray(idx)
    return idx if n_traj == 1 else (idx[:, None] * n_traj + np.arange(n_traj)).ravel()


def evaluate(circ, model, X, y, n_classes, init=None, return_proba=False, n_traj=1):
    """Metrics (balanced accuracy, AUC, ...) of a trained model on (X, y); optionally also
    the class probabilities. n_traj > 1 averages noisy encoding trajectories.
    """
    logits = circ.predict_logits(X, model["params"], model["W"], model["b"], init=init, n_traj=n_traj)
    P = _softmax(logits)
    s = U.summarise(y, P.argmax(1), P, np.arange(n_classes))
    return (s, P) if return_proba else s


def evaluate_shots(circ, model, X, y, n_classes, shots, seed, init=None, n_traj=1):
    """Test metrics when the readout is estimated from `shots` measurements in
    the computational basis (one measurement setting): outcome counts are
    sampled from the exact outcome distribution (averaged over noise
    trajectories), and every Z-type readout is computed from those counts."""
    psi = circ.states(X, model["params"], init)
    P = np.abs(psi) ** 2
    if n_traj > 1:
        P = P.reshape(-1, n_traj, P.shape[1]).mean(axis=1)
    P = P / P.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(shots, P)
    F = (counts / shots) @ circ.readout.T
    Pr = _softmax(F @ np.asarray(model["W"]).T + np.asarray(model["b"]))
    return U.summarise(y, Pr.argmax(1), Pr, np.arange(n_classes)), Pr


def train(circ, n_classes, Xtr, ytr, Xva, yva, seed, init_tr=None, init_va=None, verbose=False,
          n_traj=1):
    """n_traj > 1: Xtr/init_tr (and Xva/init_va) hold n_traj rows per label."""
    rng = np.random.default_rng(seed)
    params = rng.normal(0.0, 0.3, circ.n_params)
    W = rng.normal(0.0, 0.1 / np.sqrt(max(1, circ.readout.shape[0] / 16)),
                   (n_classes, circ.readout.shape[0]))
    b = np.zeros(n_classes)
    counts = np.bincount(ytr, minlength=n_classes).astype(float)
    cw = np.where(counts > 0, len(ytr) / (n_classes * np.maximum(counts, 1)), 0.0)
    opt = Adam([params.shape, W.shape, b.shape], C.VQC_LR)
    best, hist, bad = None, [], 0
    for ep in range(1, C.VQC_EPOCHS + 1):
        perm = rng.permutation(len(ytr))
        losses = []
        for s in range(0, len(perm), C.VQC_BATCH):
            idx = perm[s:s + C.VQC_BATCH]
            rows = _rows(idx, n_traj)
            loss, gp, gW, gb = circ.loss_and_grad(
                Xtr[rows], ytr[idx], params, W, b, cw,
                init=None if init_tr is None else init_tr[rows], n_traj=n_traj)
            if C.CLF_WEIGHT_DECAY and W.shape[1] > 16:      # only the large readouts
                gW = gW + C.CLF_WEIGHT_DECAY * W
            opt.step([params, W, b], [gp, gW, gb])
            losses.append(loss)
        model = {"params": params, "W": W, "b": b}
        va = evaluate(circ, model, Xva, yva, n_classes, init_va, n_traj=n_traj)
        hist.append({"epoch": ep, "train_loss": float(np.mean(losses)),
                     "val_bal_acc": va["bal_acc"], "val_auc": va["auc"]})
        if verbose:
            print(f"        epoch {ep:2d} loss {np.mean(losses):.3f} val bal_acc {va['bal_acc']:.3f}")
        if best is None or va["bal_acc"] > best["val_bal_acc"] + 1e-9:
            best = {"params": params.copy(), "W": W.copy(), "b": b.copy(),
                    "epoch": ep, "val_bal_acc": va["bal_acc"], "val_auc": va["auc"]}
            bad = 0
        else:
            bad += 1
            if bad >= C.VQC_PATIENCE:
                break
    return best, hist


def train_seeds(circ, n_classes, Xtr, ytr, Xva, yva, Xte, yte, seeds,
                init_tr=None, init_va=None, init_te=None, n_traj=1):
    """Train with several seeds; returns per-seed test metrics and their mean/std."""
    runs = []
    for sd in seeds:
        model, hist = train(circ, n_classes, Xtr, ytr, Xva, yva, sd, init_tr, init_va, n_traj=n_traj)
        te, P = evaluate(circ, model, Xte, yte, n_classes, init_te, return_proba=True, n_traj=n_traj)
        runs.append({"seed": sd, "best_epoch": model["epoch"], "val_bal_acc": model["val_bal_acc"],
                     "test": te, "history": hist, "model": model,
                     "test_proba": np.round(P, 6).tolist()})
    ba = [r["test"]["bal_acc"] for r in runs]
    au = [r["test"]["auc"] for r in runs]
    return {"training": training_signature(),
            "test_labels": np.asarray(yte).tolist(),
            "test_bal_acc_mean": float(np.mean(ba)), "test_bal_acc_std": float(np.std(ba)),
            "test_auc_mean": float(np.nanmean(au)), "test_auc_std": float(np.nanstd(au)),
            "runs": runs}
