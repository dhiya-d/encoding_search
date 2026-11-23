# ----------------------------------------------------------------
# aae_sim.py  —  Batched approximate-amplitude-encoding simulator.
#
# Takes a genome in the SAME format as the benchmark repo
#   [{"kind": "rot", "gate": "RY"},
#    {"kind": "ent", "gate": "CNOT_ring", "stride": 1, "step": 1, "offset": 0}, ...]
# Entangling layers may also set "boundary": "open" (no wrap-around pair,
# hierarqcal Qcycle boundary) and "reverse": true (apply the pairs in
# reverse order); both default to the original behaviour.
# builds its connectivity with hierarqcal Qcycle motifs (exactly as the
# repo's circuit.py does), and simulates it in pure NumPy.
#
# Every image gets its own parameter vector; all images in a chunk are
# simulated and trained at once. Gradients are exact (adjoint method),
# and Adam is element-wise, so each image is optimised independently —
# equivalent to the repo's per-target inner loop, just much faster.
#
# Conventions match PennyLane: wire 0 is the most significant bit,
# RY(t) = exp(-i t Y / 2), etc. Parameters are initialised uniformly in
# [0, 2*pi), so — like the repo — nothing pins the global sign/phase of
# the result: fidelity |<t|psi>|^2 is identical for psi and -psi.
# ----------------------------------------------------------------

import json

import numpy as np
from hierarqcal import Qcycle, Qunitary

ROT_AXES = {"RY": ("Y",), "RYRZ": ("Y", "Z"), "RXRYRZ": ("X", "Y", "Z")}
ENT_GATES = {"CNOT_ring": "CNOT", "CZ_ring": "CZ", "CNOT_pairs": "CNOT"}

# The HAND-DESIGNED 31-CNOT REFERENCE circuit of the paper (Sec. IV A).
# Used by the diagnostics and as the baseline of the amplitude search
# whenever no searched genome is given. A plain hardware-efficient RY stack
# with four entangling rings (strides 1, 2, 3 and 1 with offset 1) and 31
# two-qubit gates, i.e. in the range of the searched circuits. Its outputs
# live in experiments_handref/ ("default" in older outputs).
DEFAULT_GENOME = [
    {"kind": "rot", "gate": "RY"},
    {"kind": "ent", "gate": "CNOT_ring", "stride": 1, "step": 1, "offset": 0},
    {"kind": "rot", "gate": "RY"},
    {"kind": "ent", "gate": "CNOT_ring", "stride": 2, "step": 1, "offset": 0},
    {"kind": "rot", "gate": "RY"},
    {"kind": "ent", "gate": "CZ_ring", "stride": 3, "step": 1, "offset": 0},
    {"kind": "rot", "gate": "RY"},
    {"kind": "ent", "gate": "CNOT_ring", "stride": 1, "step": 1, "offset": 1},
    {"kind": "rot", "gate": "RY"},
]


def load_genome_file(path):
    """Returns (genome, meta). Accepts a bare list or a dict holding the
    list under a common key (selected_genome.json from the repo)."""
    with open(path) as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj, {}
    for key in ("genome", "architecture", "layers", "best_genome"):
        if isinstance(obj.get(key), list):
            return obj[key], obj
    raise KeyError(f"{path}: no genome list found; top-level keys are {list(obj)}")


def _noop(bits, symbols=None):
    """Placeholder gate for hierarqcal motifs; only the connectivity is used."""
    pass


def _edges(stride, step, offset, arity, n, boundary="periodic"):
    """Qubit tuples of a hierarqcal Qcycle motif (stride/step/offset/boundary) on n qubits."""
    m = Qcycle(stride=stride, step=step, offset=offset, boundary=boundary,
               mapping=Qunitary(_noop, n_symbols=0, arity=arity))
    m(list(range(n)))
    return [tuple(int(q) for q in e) for e in m.E]


class AAECircuit:
    """Batched simulator of one amplitude-encoding genome: each row (image) has its own
    parameter vector, and fidelity gradients are computed with the adjoint method.
    """
    def __init__(self, genome, n_qubits=8):
        """Translate the genome into a flat gate list ('R', axis, qubit, param) /
        ('CNOT'|'CZ', a, b), count parameters and two-qubit gates, and precompute index
        permutations (CNOT) and sign vectors (CZ) for the entanglers.
        """
        self.n = n_qubits
        self.dim = 2 ** n_qubits
        ops, p, n2q = [], 0, 0
        for layer in genome:
            kind = layer["kind"]
            if kind == "rot":
                if layer["gate"] not in ROT_AXES:
                    raise KeyError(f"unknown rotation gate {layer['gate']!r}; "
                                   f"add it to ROT_AXES in aae_sim.py")
                for (q,) in _edges(1, 1, 0, 1, n_qubits):
                    for ax in ROT_AXES[layer["gate"]]:
                        ops.append(("R", ax, q, p))
                        p += 1
            elif kind == "ent":
                if layer["gate"] not in ENT_GATES:
                    raise KeyError(f"unknown entangler {layer['gate']!r}; "
                                   f"add it to ENT_GATES in aae_sim.py")
                g = ENT_GATES[layer["gate"]]
                pairs = _edges(layer.get("stride", 1), layer.get("step", 1),
                               layer.get("offset", 0), 2, n_qubits,
                               layer.get("boundary", "periodic"))
                if layer.get("reverse", False):      # e.g. Qiskit's reverse-linear order
                    pairs = pairs[::-1]
                for a, b in pairs:
                    ops.append((g, a, b))
                    n2q += 1
            else:
                raise ValueError(f"unknown layer kind {kind!r}")
        self.ops = ops
        self.n_params = p
        self.n_two_qubit = n2q
        self.is_complex = any(o[0] == "R" and o[1] != "Y" for o in ops)
        self.dtype = np.complex128 if self.is_complex else np.float64

        k = np.arange(self.dim)
        self._perm, self._sign = {}, {}
        for o in ops:
            if o[0] in ("CNOT", "CZ") and (o[1], o[2]) not in self._perm:
                a, b = o[1], o[2]
                ba = (k >> (self.n - 1 - a)) & 1
                bb = (k >> (self.n - 1 - b)) & 1
                self._perm[(a, b)] = np.where(ba == 1, k ^ (1 << (self.n - 1 - b)), k)
                self._sign[(a, b)] = np.where((ba & bb) == 1, -1.0, 1.0)

    def edges_description(self):
        """List of the two-qubit gates as [gate, control, target]."""
        return [list(o) for o in self.ops if o[0] != "R"]

    # -- gate kernels --------------------------------------------------
    def _rot(self, psi, axis, q, theta):
        """Apply a rotation about `axis` on qubit q with one angle per row of psi."""
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
        else:  # Z
            out[:, :, 0, :] = (c - 1j * s) * a0
            out[:, :, 1, :] = (c + 1j * s) * a1
        return out.reshape(B, -1)

    def _gen(self, psi, axis, q):
        """-(i/2) sigma_axis acting on qubit q  (= dR/dtheta * R^-1)."""
        B = psi.shape[0]
        v = psi.reshape(B, 2 ** q, 2, -1)
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        out = np.empty_like(v)
        if axis == "Y":
            out[:, :, 0, :] = -0.5 * a1
            out[:, :, 1, :] = 0.5 * a0
        elif axis == "X":
            out[:, :, 0, :] = -0.5j * a1
            out[:, :, 1, :] = -0.5j * a0
        else:
            out[:, :, 0, :] = -0.5j * a0
            out[:, :, 1, :] = 0.5j * a1
        return out.reshape(B, -1)

    def _fixed(self, psi, op):
        """Apply a fixed entangler: CNOT as an index permutation, CZ as a sign flip."""
        key = (op[1], op[2])
        if op[0] == "CNOT":
            return psi[:, self._perm[key]]
        return psi * self._sign[key]

    # -- simulation ----------------------------------------------------
    def forward(self, params):
        """Final states (rows) for a batch of parameter vectors, starting from |0...0>."""
        B = params.shape[0]
        psi = np.zeros((B, self.dim), dtype=self.dtype)
        psi[:, 0] = 1.0
        for op in self.ops:
            psi = (self._rot(psi, op[1], op[2], params[:, op[3]])
                   if op[0] == "R" else self._fixed(psi, op))
        return psi

    def fidelity_and_grad(self, params, targets):
        """Returns (fid, d(1-fid)/dparams, psi) for each row."""
        psi = self.forward(params)
        ov = np.sum(targets.conj() * psi, axis=1)            # <t|psi>
        fid = np.abs(ov) ** 2
        grad = np.zeros_like(params)
        phi, lam = psi, targets.astype(self.dtype)
        for op in reversed(self.ops):
            if op[0] == "R":
                _, ax, q, pi = op
                g = np.sum(lam.conj() * self._gen(phi, ax, q), axis=1)
                grad[:, pi] = -2.0 * np.real(np.conj(ov) * g)
                phi = self._rot(phi, ax, q, -params[:, pi])
                lam = self._rot(lam, ax, q, -params[:, pi])
            else:
                phi = self._fixed(phi, op)
                lam = self._fixed(lam, op)
        return fid, grad, psi


def fit_states(circ, targets, steps=250, lr=0.15, restarts=2, seed=0,
               chunk=1024, verbose=True):
    """Per-image Adam fit, best-of-`restarts`. Returns (fid, params, states)."""
    N, P = len(targets), circ.n_params
    best_fid = np.full(N, -np.inf)
    best_params = np.zeros((N, P))
    best_states = np.zeros((N, circ.dim), dtype=circ.dtype)
    rng = np.random.default_rng(seed)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for r in range(restarts):
        for s in range(0, N, chunk):
            T = targets[s:s + chunk]
            B = len(T)
            th = rng.uniform(0.0, 2 * np.pi, (B, P))
            m, v = np.zeros_like(th), np.zeros_like(th)
            for t in range(1, steps + 1 if P > 0 else 1):
                _, g, _ = circ.fidelity_and_grad(th, T)
                m = b1 * m + (1 - b1) * g
                v = b2 * v + (1 - b2) * g * g
                th -= lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
            psi = circ.forward(th)
            fid = np.abs(np.sum(T.conj() * psi, axis=1)) ** 2
            sl = slice(s, s + B)
            better = fid > best_fid[sl]
            best_fid[sl] = np.where(better, fid, best_fid[sl])
            best_params[sl][better] = th[better]
            best_states[sl][better] = psi[better]
            if verbose:
                print(f"      restart {r + 1}/{restarts}  images {s + B}/{N}  "
                      f"mean fid so far {best_fid[:s + B].mean():.4f}", flush=True)
    return best_fid, best_params, best_states
