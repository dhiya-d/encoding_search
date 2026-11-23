# ----------------------------------------------------------------
# qsim.py  —  batched statevector simulator (NumPy).
#
# Operations (wire 0 = most significant bit, PennyLane conventions):
#   ("H", q)                       Hadamard
#   ("E", axis, q, f, s)           data encoding  exp(-i s x_f P/2),  P in {X,Y,Z}
#   ("ZZ", a, b, f, g, s)          data encoding  exp(-i s x_f x_g Z_a Z_b / 2)
#   ("R", axis, q, p)              trainable      exp(-i theta_p P/2)
#   ("CNOT", a, b), ("CZ", a, b)   fixed entanglers
#
# Every sample has its own data angles; trainable angles are shared.
# Gradients use the adjoint method for a diagonal (Z-type) observable,
# which is all the variational classifier's readout needs.
# ----------------------------------------------------------------
import numpy as np


class Circuit:
    """Batched statevector simulator for encoding + trainable circuits with a diagonal
    (Z-type) readout and adjoint gradients (see the operation list in the header).
    """
    def __init__(self, ops, n_qubits=8, readout=None):
        """Store the gate list, count trainable parameters and precompute bit tables, CNOT
        permutations, CZ signs and ZZ parities used by the gate kernels.
        """
        self.ops = list(ops)
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        self.n_params = 1 + max([o[3] for o in self.ops if o[0] == "R"], default=-1)
        k = np.arange(self.dim)
        self.bits = np.stack([(k >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)])  # (n, dim)
        self.z = 1.0 - 2.0 * self.bits                                                   # Z eigenvalues
        self._perm, self._sign, self._par = {}, {}, {}
        for o in self.ops:
            if o[0] in ("CNOT", "CZ", "ZZ"):
                a, b = o[1], o[2]
                ba, bb = self.bits[a], self.bits[b]
                self._perm[(a, b)] = np.where(ba == 1, k ^ (1 << (n_qubits - 1 - b)), k)
                self._sign[(a, b)] = np.where((ba & bb) == 1, -1.0, 1.0)
                self._par[(a, b)] = self.z[a] * self.z[b]
        # readout: rows are diagonal observables; features = <psi|row|psi>.
        # default: Z_q and Z_q Z_{q+1} (ring) on all qubits
        self.readout = np.asarray(readout, float) if readout is not None else np.concatenate(
            [self.z, np.stack([self.z[q] * self.z[(q + 1) % n_qubits] for q in range(n_qubits)])])

    # -- kernels ---------------------------------------------------------
    @staticmethod
    def _rot(psi, axis, q, theta):
        """Apply a rotation about `axis` on qubit q with one angle per row."""
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** q, 2, -1)
        c = np.cos(theta / 2)[:, None, None]
        s = np.sin(theta / 2)[:, None, None]
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        out = np.empty_like(v)
        if axis == "Y":
            out[:, :, 0, :] = c * a0 - s * a1
            out[:, :, 1, :] = s * a0 + c * a1
        elif axis == "X":
            out[:, :, 0, :] = c * a0 - 1j * s * a1
            out[:, :, 1, :] = -1j * s * a0 + c * a1
        else:
            out[:, :, 0, :] = (c - 1j * s) * a0
            out[:, :, 1, :] = (c + 1j * s) * a1
        return out.reshape(B, -1)

    @staticmethod
    def _gen(psi, axis, q):
        """-(i/2) P_q psi, so that dR/dtheta = gen o R."""
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** q, 2, -1)
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        out = np.empty_like(v)
        if axis == "Y":
            out[:, :, 0, :], out[:, :, 1, :] = -0.5 * a1, 0.5 * a0
        elif axis == "X":
            out[:, :, 0, :], out[:, :, 1, :] = -0.5j * a1, -0.5j * a0
        else:
            out[:, :, 0, :], out[:, :, 1, :] = -0.5j * a0, 0.5j * a1
        return out.reshape(B, -1)

    @staticmethod
    def _had(psi, q):
        """Apply a Hadamard gate on qubit q to every row."""
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** q, 2, -1)
        out = np.empty_like(v)
        out[:, :, 0, :] = (v[:, :, 0, :] + v[:, :, 1, :]) / np.sqrt(2)
        out[:, :, 1, :] = (v[:, :, 0, :] - v[:, :, 1, :]) / np.sqrt(2)
        return out.reshape(B, -1)

    def _apply(self, psi, op, X, params, inverse=False):
        """Apply one operation (or its inverse) to the batch; data-encoding angles come
        from X, trainable angles from params.
        """
        kind = op[0]
        sgn = -1.0 if inverse else 1.0
        B = psi.shape[0]
        if kind == "H":
            return self._had(psi, op[1])
        if kind == "E":
            _, axis, q, f, s = op
            return self._rot(psi, axis, q, sgn * s * X[:, f])
        if kind == "R":
            _, axis, q, p = op
            return self._rot(psi, axis, q, np.full(B, sgn * params[p]))
        if kind == "ZZ":
            _, a, b, f, g, s = op
            th = sgn * s * X[:, f] * X[:, g]
            return psi * np.exp(-0.5j * th[:, None] * self._par[(a, b)][None, :])
        if kind == "CNOT":
            return psi[:, self._perm[(op[1], op[2])]]
        if kind == "CZ":
            return psi * self._sign[(op[1], op[2])]
        raise ValueError(kind)

    # -- forward ---------------------------------------------------------
    def states(self, X, params=None, init=None, chunk=2048):
        """init: optional (B, dim) starting states (e.g. amplitude-encoded
        images); default |0...0>."""
        out = []
        for s in range(0, len(X), chunk):
            Xb = X[s:s + chunk]
            if init is None:
                psi = np.zeros((len(Xb), self.dim), dtype=np.complex128)
                psi[:, 0] = 1.0
            else:
                psi = np.array(init[s:s + chunk], dtype=np.complex128)
            for op in self.ops:
                psi = self._apply(psi, op, Xb, params)
            out.append(psi)
        return np.concatenate(out) if out else np.zeros((0, self.dim), complex)

    def readout_features(self, psi, n_traj=1):
        """Readout expectation values. With n_traj > 1, psi holds n_traj noise
        trajectories per sample (consecutive rows); averaging them gives the
        exact expectation value for the mixed state."""
        F = (np.abs(psi) ** 2) @ self.readout.T                       # (rows, n_readout)
        if n_traj > 1:
            F = F.reshape(-1, n_traj, F.shape[1]).mean(axis=1)
        return F

    def pauli_features(self, psi):
        """<X_q>, <Y_q>, <Z_q> for every qubit: (B, 3n). Local measurements only."""
        B = psi.shape[0]
        out = []
        for q in range(self.n):
            v = psi.reshape(B, 2 ** q, 2, -1)
            a = (v[:, :, 0, :].conj() * v[:, :, 1, :]).sum(axis=(1, 2))
            p = np.abs(v) ** 2
            out += [2 * a.real, 2 * a.imag, (p[:, :, 0, :] - p[:, :, 1, :]).sum(axis=(1, 2))]
        return np.stack(out, axis=1)

    # -- training --------------------------------------------------------
    def loss_and_grad(self, X, y, params, W, b, class_w, init=None, n_traj=1):
        """Weighted cross-entropy of logits = W @ readout + b.
        Returns loss, dparams, dW, db. With n_traj > 1, X and init have
        n_traj rows per label in y (noise trajectories of one mixed state)."""
        psi = self.states(X, params, init)
        F = self.readout_features(psi, n_traj)
        logits = F @ W.T + b
        logits -= logits.max(1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(1, keepdims=True)
        wts = class_w[y]
        norm = wts.sum()
        loss = float(-(wts * np.log(P[np.arange(len(y)), y] + 1e-12)).sum() / norm)
        dlog = P.copy()
        dlog[np.arange(len(y)), y] -= 1.0
        dlog *= (wts / norm)[:, None]
        dW = dlog.T @ F
        db = dlog.sum(0)
        dF = dlog @ W                                                  # (B, n_readout)
        if n_traj > 1:
            dF = np.repeat(dF, n_traj, axis=0) / n_traj
        lam = (dF @ self.readout) * psi                                # O_eff |psi>
        phi = psi
        grad = np.zeros(self.n_params)
        for op in reversed(self.ops):
            if op[0] == "R":
                _, axis, q, p = op
                g = np.sum(lam.conj() * self._gen(phi, axis, q), axis=1)
                grad[p] += 2.0 * np.real(g).sum()
            phi = self._apply(phi, op, X, params, inverse=True)
            lam = self._apply(lam, op, X, params, inverse=True)
        return loss, grad, dW, db

    def predict_logits(self, X, params, W, b, init=None, n_traj=1):
        """Class logits: linear layer (W, b) on the readout features of the final states."""
        return self.readout_features(self.states(X, params, init), n_traj) @ W.T + b


def basis_projectors(qubits, n_qubits=8):
    """Diagonal projectors onto every basis state of `qubits` (probability readout)."""
    k = np.arange(2 ** n_qubits)
    bits = np.stack([(k >> (n_qubits - 1 - q)) & 1 for q in qubits])
    idx = np.zeros_like(k)
    for i in range(len(qubits)):
        idx = (idx << 1) | bits[i]
    return np.stack([(idx == j).astype(float) for j in range(2 ** len(qubits))])
