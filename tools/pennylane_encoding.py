#!/usr/bin/env python3
# ----------------------------------------------------------------
# pennylane_encoding.py  —  use a searched encoding in PennyLane code
# (e.g. your hierarqcal QCNN).
#
#   import pennylane as qml
#   from pennylane_encoding import load_encoding, apply_encoding, load_features
#
#   enc = load_encoding("experiments_fourier/bloodmnist/downstream/encodings/search_best.json")
#   X, y = load_features("bloodmnist", "train", enc)      # angles, first n_features columns
#   dev = qml.device("default.qubit", wires=8)
#
#   @qml.qnode(dev)
#   def circuit(x, weights):
#       apply_encoding(enc, x)            # S(x) from the search
#       my_qcnn(weights)                  # e.g. a hierarqcal motif's circuit
#       return qml.probs(wires=[0, 4])
#
# classifier_qnode(enc) rebuilds the trained re-uploading classifier.
# Self-check against the NumPy simulator used for the search:
#   python tools/pennylane_encoding.py experiments_fourier/<dataset>/downstream/encodings/<name>.json
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import json
import os
import sys

import numpy as np

import config as C
import encoding_genome as G


def load_encoding(path):
    """Exported encoding JSON (or a bare genome file)."""
    with open(path) as f:
        enc = json.load(f)
    if "genome" not in enc:                       # bare genome file
        enc = {"genome": enc, "n_features": enc["n_features"]}
    return enc


def load_features(dataset, split, enc=None):
    """Feature matrix and labels of one split, truncated to the encoding's n_features."""
    fdir = os.path.join(C.OUT_ROOT, dataset, "features")
    X = np.load(os.path.join(fdir, f"{split}_features.npy"))
    y = np.load(os.path.join(fdir, f"{split}_labels.npy"))
    if enc is not None:
        X = X[:, :enc["genome"]["n_features"]]
    return X, y


def _apply_ops(ops, x, params=None, wires=None):
    """Apply a list of qsim operations as PennyLane gates (inside a QNode)."""
    import pennylane as qml
    w = wires if wires is not None else list(range(C.N_QUBITS))
    for op in ops:
        k = op[0]
        if k == "H":
            qml.Hadamard(wires=w[op[1]])
        elif k == "E":
            _, axis, q, f, s = op
            getattr(qml, f"R{axis}")(s * x[f], wires=w[q])
        elif k == "ZZ":
            _, a, b, f, g, s = op
            qml.IsingZZ(s * x[f] * x[g], wires=[w[a], w[b]])
        elif k == "R":
            _, axis, q, p = op
            getattr(qml, f"R{axis}")(params[p], wires=w[q])
        elif k == "CNOT":
            qml.CNOT(wires=[w[op[1]], w[op[2]]])
        elif k == "CZ":
            qml.CZ(wires=[w[op[1]], w[op[2]]])
        else:
            raise ValueError(k)


def apply_encoding(enc, x, wires=None):
    """Apply the data-encoding circuit S(x) inside a QNode.
    x: one feature vector (at least n_features entries). wires: 8 wire labels."""
    _apply_ops(G.encoding_ops(enc["genome"]), x, wires=wires)


def classifier_qnode(enc, device=None):
    """Trained re-uploading classifier -> function x -> class probabilities."""
    import pennylane as qml
    import classifiers as CL
    m = enc["reupload_classifier"]
    ops = CL.reupload_ops(G.encoding_ops(enc["genome"]), m["reuploads"])
    params, W, b = np.array(m["params"]), np.array(m["W"]), np.array(m["b"])
    n = C.N_QUBITS
    dev = device or qml.device("default.qubit", wires=n)
    obs = [qml.PauliZ(q) for q in range(n)] + [qml.PauliZ(q) @ qml.PauliZ((q + 1) % n) for q in range(n)]

    @qml.qnode(dev)
    def readout(x):
        """Expectation values of the readout observables."""
        _apply_ops(ops, x, params)
        return [qml.expval(o) for o in obs]

    def predict_proba(x):
        """Class probabilities: softmax of the linear layer on the readout."""
        z = W @ np.array(readout(x), dtype=float) + b
        z = np.exp(z - z.max())
        return z / z.sum()

    return predict_proba


def self_check(path, n=3):
    """Check that the PennyLane circuit reproduces the NumPy simulator's states on random
    inputs.
    """
    import classifiers as CL
    from qsim import Circuit
    import pennylane as qml
    enc = load_encoding(path)
    d = enc["genome"]["n_features"]
    X = np.random.default_rng(0).uniform(-C.ANGLE_RANGE, C.ANGLE_RANGE, (n, d))
    dev = qml.device("default.qubit", wires=C.N_QUBITS)

    @qml.qnode(dev)
    def state(x):
        """Encoded state of one input (PennyLane)."""
        apply_encoding(enc, x)
        return qml.state()

    S = Circuit(G.encoding_ops(enc["genome"])).states(X)
    err = max(np.abs(np.array(state(X[i])) - S[i]).max() for i in range(n))
    print(f"encoding state: max |PennyLane - simulator| = {err:.2e}")
    if "reupload_classifier" in enc:
        m = enc["reupload_classifier"]
        circ = Circuit(CL.reupload_ops(G.encoding_ops(enc["genome"]), m["reuploads"]))
        logits = circ.predict_logits(X, np.array(m["params"]), np.array(m["W"]), np.array(m["b"]))
        P = np.exp(logits - logits.max(1, keepdims=True))
        P /= P.sum(1, keepdims=True)
        f = classifier_qnode(enc, dev)
        err = max(np.abs(f(X[i]) - P[i]).max() for i in range(n))
        print(f"trained classifier probabilities: max difference = {err:.2e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("usage: python pennylane_encoding.py <exported encoding json>")
    self_check(sys.argv[1])
