# ----------------------------------------------------------------
# noise.py  —  gate noise for the encoding circuits (trajectory method).
#
# Noise model: after every two-qubit gate (CNOT, CZ, ZZ rotation), a
# two-qubit depolarizing channel with probability p2: one of the 15
# non-identity two-qubit Paulis is applied uniformly at random. After every
# single-qubit gate, a single-qubit depolarizing channel with probability
# p1 (one of X, Y, Z). Averaged over trajectories this is
#     rho -> (1 - p) rho + p/15 sum_P P rho P      (two-qubit case).
#
# Each image is simulated n_traj times with independently sampled errors;
# rows i*n_traj ... i*n_traj + n_traj - 1 belong to image i. Quantities
# that are linear in rho (measurement expectation values, fidelities,
# Tr(rho rho')) are estimated by averaging over those rows; the
# classifiers in classifiers.py accept this layout via n_traj.
#
# Only the ENCODING is noisy; the classifier circuit is kept noise-free so
# that differences between encoders are not confounded by the classifier.
# ----------------------------------------------------------------
import numpy as np

TWO_QUBIT = ("CNOT", "CZ", "ZZ")


def _apply_pauli(psi, q, kind, rows):
    """Apply X (1), Y (2) or Z (3) on qubit q to the rows selected by mask `rows`."""
    if not rows.any():
        return
    B = psi.shape[0]
    v = psi.reshape(B, 2 ** q, 2, -1)
    sub = v[rows]
    a0, a1 = sub[:, :, 0, :].copy(), sub[:, :, 1, :].copy()
    if kind == 1:
        sub[:, :, 0, :], sub[:, :, 1, :] = a1, a0
    elif kind == 2:
        sub[:, :, 0, :], sub[:, :, 1, :] = -1j * a1, 1j * a0
    else:
        sub[:, :, 1, :] = -a1
    v[rows] = sub


def depolarize_1q(psi, q, p, rng):
    """Single-qubit depolarizing channel on qubit q: with probability p a uniformly random
    X, Y or Z is applied (independently per row, in place).
    """
    if p <= 0:
        return
    hit = rng.random(psi.shape[0]) < p
    kind = rng.integers(1, 4, psi.shape[0])
    for k in (1, 2, 3):
        _apply_pauli(psi, q, k, hit & (kind == k))


def depolarize_2q(psi, a, b, p, rng):
    """Two-qubit depolarizing channel on (a, b): with probability p one of the 15 non-
    identity Pauli pairs is applied (independently per row, in place).
    """
    if p <= 0:
        return
    n = psi.shape[0]
    hit = rng.random(n) < p
    k = rng.integers(1, 16, n)                 # 1..15: non-identity pairs (pa, pb)
    pa, pb = k // 4, k % 4
    for kind in (1, 2, 3):
        _apply_pauli(psi, a, kind, hit & (pa == kind))
        _apply_pauli(psi, b, kind, hit & (pb == kind))


def noisy_fourier_states(circ, X, p2, p1, n_traj, seed, chunk=2048):
    """qsim.Circuit encoding (data-dependent ops) with gate noise.
    Returns (len(X) * n_traj, dim) complex states."""
    rng = np.random.default_rng(seed)
    Xr = np.repeat(X, n_traj, axis=0)
    out = []
    for s in range(0, len(Xr), chunk):
        Xb = Xr[s:s + chunk]
        psi = np.zeros((len(Xb), circ.dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        for op in circ.ops:
            psi = circ._apply(psi, op, Xb, None)
            if op[0] in TWO_QUBIT:
                depolarize_2q(psi, op[1], op[2], p2, rng)
            elif op[0] in ("E", "R", "H"):
                depolarize_1q(psi, op[2] if op[0] in ("E", "R") else op[1], p1, rng)
        out.append(psi)
    return np.concatenate(out)


def noisy_aae_states(aae_circ, params, p2, p1, n_traj, seed, chunk=2048):
    """aae_sim.AAECircuit with per-image fitted parameters, with gate noise.
    Returns (len(params) * n_traj, dim) complex states."""
    rng = np.random.default_rng(seed)
    Pr = np.repeat(np.asarray(params, dtype=float), n_traj, axis=0)
    out = []
    for s in range(0, len(Pr), chunk):
        Pb = Pr[s:s + chunk]
        psi = np.zeros((len(Pb), aae_circ.dim), dtype=np.complex128)
        psi[:, 0] = 1.0
        for op in aae_circ.ops:
            if op[0] == "R":
                _, axis, q, pi = op
                psi = aae_circ._rot(psi, axis, q, Pb[:, pi])
                depolarize_1q(psi, q, p1, rng)
            else:
                psi = aae_circ._fixed(psi, op)
                depolarize_2q(psi, op[1], op[2], p2, rng)
        out.append(psi)
    return np.concatenate(out)


def mixed_kernel(SA, SB, n_traj, chunk=256):
    """K_ij = Tr(rho_i rho_j) estimated from trajectories: mean over a, b of
    |<psi_i^a|psi_j^b>|^2. SA, SB have n_traj rows per sample."""
    na, nb = len(SA) // n_traj, len(SB) // n_traj
    K = np.empty((na, nb))
    for i in range(0, na, chunk):
        j = min(na, i + chunk)
        O = np.abs(SA[i * n_traj:j * n_traj].conj() @ SB.T) ** 2
        K[i:j] = O.reshape(j - i, n_traj, nb, n_traj).mean(axis=(1, 3))
    return K


def mixed_fidelity(S, targets, n_traj):
    """<t|rho|t> per sample, from trajectories."""
    T = np.repeat(np.asarray(targets), n_traj, axis=0)
    f = np.abs(np.sum(T.conj() * S, axis=1)) ** 2
    return f.reshape(-1, n_traj).mean(axis=1)


def rows_of(idx, n_traj):
    """Row indices of all trajectories of the given images (image i -> rows i*n_traj ...)."""
    idx = np.asarray(idx)
    return (idx[:, None] * n_traj + np.arange(n_traj)).ravel()


def swap_test_kernel(K, shots, rng, symmetric=False):
    """Kernel entries estimated with the swap test: the ancilla reads 0 with
    probability (1 + Tr(rho rho'))/2, so K_hat = 2 n0/shots - 1."""
    P0 = np.clip((1.0 + K) / 2.0, 0.0, 1.0)
    if symmetric:
        iu = np.triu_indices(len(K))
        est = 2.0 * rng.binomial(shots, P0[iu]) / shots - 1.0
        Kh = np.zeros_like(K)
        Kh[iu] = est
        Kh = Kh + Kh.T - np.diag(np.diag(Kh))
    else:
        Kh = 2.0 * rng.binomial(shots, P0) / shots - 1.0
    return np.clip(Kh, 0.0, 1.0)
