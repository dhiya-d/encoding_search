#!/usr/bin/env python3
# ----------------------------------------------------------------
# product_kernel_check.py  —  exact classical evaluation of the fidelity
# kernel of entanglement-free (product-state) Fourier encodings.
#
# If an encoding contains only single-qubit gates (no CNOT/CZ/ZZ layers),
# its state is a product state |psi(x)> = (x)_q |phi_q(x)>, so
#     k(x, x') = |<psi(x)|psi(x')>|^2 = prod_q |<phi_q(x)|phi_q(x')>|^2,
# which a classical computer evaluates with n two-dimensional vectors per
# sample instead of a 2^n-dimensional state. This script
#   1. finds every entanglement-free encoding among the Fourier picks and
#      references (seed 0 and the extra search seeds),
#   2. checks the product formula against the state-vector kernel,
#   3. retrains the kernel SVM with the CLASSICAL kernel on exactly the same
#      data and C grid, and compares it with the saved quantum-kernel result,
#   4. times the classical kernel for growing qubit numbers.
# A match shows that the quantum kernel of these encodings is an exactly
# classical model.
#
# Usage (from the repository root):
#   python studies/product_kernel_check.py --dataset bloodmnist
# Output: reports/product_kernel_<dataset>.md/.json
# ----------------------------------------------------------------
# Make the shared modules importable when this script is run from the repository
# root (see core/repo.py).
import os as _os, sys as _sys  # noqa: E401
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir, "core"))
import repo  # noqa: E402,F401

import argparse
import os
import time

import numpy as np

import common as U
import config as C
import encoding_genome as G
import make_report as R

_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)


def is_product(genome):
    """True if the encoding has no entangling or ZZ layers (product state for every input)."""
    return all(L["kind"] in ("enc", "H") for L in genome["layers"])


def _rot_batch(axis, theta):
    """(B, 2, 2) matrices exp(-i theta P / 2) for P = X, Y, Z (qsim convention)."""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    M = np.zeros((len(theta), 2, 2), dtype=complex)
    if axis == "X":
        M[:, 0, 0] = M[:, 1, 1] = c
        M[:, 0, 1] = M[:, 1, 0] = -1j * s
    elif axis == "Y":
        M[:, 0, 0] = M[:, 1, 1] = c
        M[:, 0, 1], M[:, 1, 0] = -s, s
    else:
        M[:, 0, 0], M[:, 1, 1] = np.exp(-0.5j * theta), np.exp(0.5j * theta)
    return M


def qubit_states(genome, X, n=C.N_QUBITS):
    """(n, B, 2) single-qubit states phi_q(x), same gate list as qsim.Circuit."""
    phi = np.zeros((n, len(X), 2), dtype=complex)
    phi[:, :, 0] = 1.0
    for op in G.encoding_ops(genome, n):
        if op[0] == "E":
            _, axis, q, f, s = op
            phi[q] = np.einsum("bij,bj->bi", _rot_batch(axis, s * X[:, f]), phi[q])
        elif op[0] == "H":
            phi[op[1]] = phi[op[1]] @ _H.T
        else:
            raise ValueError(f"not a product encoding: gate {op[0]}")
    return phi


def product_kernel(phiA, phiB):
    """Exact fidelity kernel of product states: product over qubits of the single-qubit
    overlaps |<phi_q(a)|phi_q(b)>|^2.
    """
    K = np.ones((phiA.shape[1], phiB.shape[1]))
    for q in range(phiA.shape[0]):
        K *= np.abs(phiA[q].conj() @ phiB[q].T) ** 2
    return K


def candidates(a):
    """Entanglement-free Fourier encodings found by the searches (all seeds)."""
    out = {}
    for s in a.seeds:
        root = a.fourier_root if s == 0 else f"{a.fourier_root}_seed{s}"
        four, _ = R.load_fourier(root, a.dataset)
        for name, r in four.items():
            if s != 0 and not name.startswith("search_"):
                continue
            if is_product(r["genome"]):
                out[f"seed{s}:{name}"] = r
    return out


def scaling(n_list, n_samples=200, seed=0):
    """Seconds for an n_samples x n_samples product kernel, one RY layer."""
    rng = np.random.default_rng(seed)
    out = []
    for n in n_list:
        X = rng.uniform(-1, 1, (n_samples, n))
        g = {"n_features": n, "layers": [{"kind": "enc", "axis": "Y", "scale": 1.0, "start": 0}]}
        t = time.time()
        try:
            ph = qubit_states(g, X, n)
            product_kernel(ph, ph)
            sec = time.time() - t
        except Exception as e:                       # genome grammar may differ
            sec = float("nan")
            print("scaling test skipped:", e)
            break
        out.append({"n_qubits": n, "seconds": sec,
                    "statevector_log10_GB_per_state": float(n * np.log10(2) + np.log10(16) - 9)})
    return out


def main():
    """For every product encoding: check the classical kernel against the state-vector
    kernel, then retrain the kernel SVM with it and compare test metrics and run time.
    """
    p = argparse.ArgumentParser(description="exact classical kernel for product-state encodings")
    p.add_argument("--dataset", default="bloodmnist")
    p.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    p.add_argument("--fourier-root", default="experiments_fourier")
    p.add_argument("--reports", default="reports")
    p.add_argument("--check-n", type=int, default=300, help="samples for the state-vector comparison")
    p.add_argument("--quick", action="store_true", help="match train_downstream.py --quick")
    a = p.parse_args()
    import train_downstream as TD
    if a.quick:
        from search_encoding import apply_quick
        apply_quick()
    C.OUT_ROOT = a.fourier_root
    feats, _ = U.load_or_build_features(a.dataset)
    D = TD.splits(feats)
    Xk, yk = D["kernel"]
    Xt, yt = D["test"]
    cands = candidates(a)
    if not cands:
        print("no entanglement-free encodings found")
        return
    rows, js = [], {"dataset": a.dataset, "encodings": {}}
    for key, saved in cands.items():
        g = saved["genome"]
        m = min(a.check_n, len(Xk))
        t0 = time.time()
        S = G.encoding_circuit(g).states(Xk[:m])
        K_sv = U.fidelity_kernel(S, S)
        t_sv = time.time() - t0
        t0 = time.time()
        ph = qubit_states(g, Xk[:m])
        K_cl = product_kernel(ph, ph)
        t_cl = time.time() - t0
        err = float(np.abs(K_sv - K_cl).max())
        pk, pt = qubit_states(g, Xk), qubit_states(g, Xt)
        clf, Cv = U.fit_kernel_svm(product_kernel(pk, pk), yk)
        res = {**U.eval_kernel_svm(clf, product_kernel(pt, pk), yt), "C": Cv}
        q = saved.get("qk_fidelity", {})
        # Same model <=> same test metrics. The cross-validated C is NOT compared:
        # kernels that agree to ~1e-15 can tie differently in the C grid search
        # and still give identical predictions.
        same = (bool(q) and abs(q["bal_acc"] - res["bal_acc"]) < 1e-9
                and abs(q["auc"] - res["auc"]) < 1e-9)
        js["encodings"][key] = {"description": saved.get("description"), "n_features": g["n_features"],
                                "max_abs_kernel_difference": err, "classical": res, "quantum_saved": q,
                                "identical_result": same, "seconds_statevector": t_sv,
                                "seconds_classical": t_cl, "check_n": m}
        rows.append([key, saved.get("description", ""), f"{err:.1e}",
                     f"{q.get('bal_acc', float('nan')):.3f} / {q.get('auc', float('nan')):.3f}",
                     f"{res['bal_acc']:.3f} / {res['auc']:.3f}", "yes" if same else "NO",
                     f"{t_sv * 1000:.0f} ms / {t_cl * 1000:.0f} ms"])
        print(f"{key}: max |K_sv - K_cl| = {err:.1e}; quantum {q.get('bal_acc')} vs classical {res['bal_acc']:.3f}")
    js["scaling"] = scaling([8, 32, 128, 512, 1024])
    md = f"# Exact classical kernel for entanglement-free encodings ({a.dataset})\n\n"
    md += ("Encodings with only single-qubit gates produce product states, so their fidelity kernel is a "
           "product of single-qubit overlaps, k(x,x') = Π_q |⟨φ_q(x)|φ_q(x')⟩|², computable classically "
           f"in O(n) per entry. Kernel check on {min(a.check_n, len(Xk))} training images; the SVM is retrained "
           "with the classical kernel on exactly the same data and C grid as the quantum kernel.\n\n")
    md += R.table(["encoding", "circuit", "max |K_quantum − K_classical|", "quantum kernel bal. acc. / AUC (saved)",
                   "classical kernel bal. acc. / AUC", "identical metrics", "time: state vector / classical"], rows)
    ok = all(e["identical_result"] for e in js["encodings"].values())
    js["all_identical"] = ok
    md += ("\n**All entanglement-free encodings are reproduced exactly by the classical product kernel.**\n\n"
           if ok else "\n**Some results differ; see the table.**\n\n")
    if js["scaling"]:
        md += "Classical product kernel, 200 × 200 entries, one RY layer:\n\n"
        md += R.table(["qubits", "seconds", "memory for ONE state vector"],
                      [[s["n_qubits"], f"{s['seconds']:.3f}", f"10^{s['statevector_log10_GB_per_state']:.0f} GB"]
                       for s in js["scaling"]])
    os.makedirs(a.reports, exist_ok=True)
    with open(os.path.join(a.reports, f"product_kernel_{a.dataset}.md"), "w") as fh:
        fh.write(md)
    U.save_json(js, os.path.join(a.reports, f"product_kernel_{a.dataset}.json"))
    print(md)


if __name__ == "__main__":
    main()
