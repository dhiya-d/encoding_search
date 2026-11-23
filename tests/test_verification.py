# ----------------------------------------------------------------
# test_verification.py  —  checks that the simulators compute what the
# paper says they compute (paper appendix, "Verification").
#
#   1. both simulators == PennyLane statevector
#        - Fourier encodings (qsim), amplitude-encoding circuits (aae_sim)
#   2. adjoint gradients == finite differences
#        - amplitude-encoding fidelity (aae_sim)
#        - classifier cross-entropy (qsim), also trajectory-averaged (noisy states)
#   3. noise trajectories reproduce the exact depolarizing channel
#      (density-matrix simulation of the same circuit)
#   4. mixed-state kernel from trajectories == Tr(rho rho') exactly
#   5. swap-test kernel estimate is unbiased
#   6. classical product kernel == state-vector fidelity kernel for
#      entanglement-free encodings
#   7. RealAmplitudes genome == Qiskit RealAmplitudes (if qiskit is installed)
#
# Usage (from the repository root):
#   python tests/test_verification.py        # plain Python, prints PASS/SKIP
#   python -m pytest tests                   # or with pytest
# Takes about a minute. Tests whose optional package (pennylane, qiskit)
# is missing are skipped.
# ----------------------------------------------------------------
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "core"))
import repo  # noqa: E402,F401  (puts all code folders on sys.path)

import config as C  # noqa: E402
import encoding_genome as G  # noqa: E402
import noise as NZ  # noqa: E402
from aae_sim import AAECircuit, DEFAULT_GENOME  # noqa: E402
from qsim import Circuit  # noqa: E402


class Skip(Exception):
    """Raised when an optional dependency is missing (plain-Python runner)."""


def skip(reason):
    """Skip the current test (pytest's skip when run under pytest)."""
    if "pytest" in sys.modules:
        import pytest
        pytest.skip(reason)
    raise Skip(reason)


def random_fourier_genome(seed):
    """A random valid genome from the Fourier search space (with entanglers and ZZ terms)."""
    rng = np.random.default_rng(seed)
    while True:
        g = G.random_genome(rng)
        kinds = {L["kind"] for L in g["layers"]}
        if {"enc", "ent"} <= kinds or "zz" in kinds:
            return g


# ── 1. qsim vs PennyLane ─────────────────────────────────────────────
def test_qsim_matches_pennylane():
    """Encoded states of random genomes agree with PennyLane's default.qubit."""
    try:
        import pennylane as qml
    except ImportError:
        skip("pennylane not installed")
    import pennylane_encoding as PE
    dev = qml.device("default.qubit", wires=C.N_QUBITS)
    for seed in range(5):
        g = random_fourier_genome(seed)
        enc = {"genome": g}
        X = np.random.default_rng(seed).uniform(-np.pi, np.pi, (3, g["n_features"]))

        @qml.qnode(dev)
        def state(x):
            PE.apply_encoding(enc, x)
            return qml.state()

        S = G.encoding_circuit(g).states(X)
        err = max(np.abs(np.asarray(state(X[i])) - S[i]).max() for i in range(len(X)))
        assert err < 1e-10, f"genome {G.describe(g)}: max deviation {err:.2e}"


def test_aae_matches_pennylane():
    """Amplitude-encoding circuit (reference genome) agrees with PennyLane's default.qubit."""
    try:
        import pennylane as qml
    except ImportError:
        skip("pennylane not installed")
    circ = AAECircuit(DEFAULT_GENOME)
    dev = qml.device("default.qubit", wires=circ.n)
    th = np.random.default_rng(7).uniform(0, 2 * np.pi, circ.n_params)
    gates = {"X": qml.RX, "Y": qml.RY, "Z": qml.RZ}

    @qml.qnode(dev)
    def state():
        for op in circ.ops:
            if op[0] == "R":
                gates[op[1]](th[op[3]], wires=op[2])
            elif op[0] == "CNOT":
                qml.CNOT(wires=[op[1], op[2]])
            else:
                qml.CZ(wires=[op[1], op[2]])
        return qml.state()

    err = np.abs(np.asarray(state()) - circ.forward(th[None])[0]).max()
    assert err < 1e-12, err


# ── 2. adjoint gradients vs finite differences ──────────────────────
def test_aae_fidelity_gradient():
    """d(1 - fidelity)/dtheta of the amplitude-encoding circuit vs central differences."""
    circ = AAECircuit(DEFAULT_GENOME)
    rng = np.random.default_rng(0)
    T = rng.normal(size=(2, circ.dim))
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    th = rng.uniform(0, 2 * np.pi, (2, circ.n_params))
    _, grad, _ = circ.fidelity_and_grad(th, T)
    h = 1e-6
    for p in rng.choice(circ.n_params, 10, replace=False):
        e = np.zeros_like(th)
        e[:, p] = h
        fp, _, _ = circ.fidelity_and_grad(th + e, T)
        fm, _, _ = circ.fidelity_and_grad(th - e, T)
        fd = -(fp - fm) / (2 * h)                       # gradient of 1 - fidelity
        assert np.allclose(grad[:, p], fd, atol=1e-6), (p, grad[:, p], fd)


def test_classifier_gradient():
    """Cross-entropy gradient of an encoding + trainable circuit vs central differences."""
    import classifiers as CL
    g = random_fourier_genome(3)
    circ = CL.build("reupload", 3, prefix_ops=G.encoding_ops(g))
    rng = np.random.default_rng(1)
    X = rng.uniform(-np.pi, np.pi, (6, g["n_features"]))
    y = np.array([0, 1, 2, 0, 1, 2])
    params = rng.normal(size=circ.n_params)
    W = rng.normal(size=(3, circ.readout.shape[0])) * 0.3
    b = rng.normal(size=3) * 0.1
    cw = np.ones(3)
    _, grad, dW, db = circ.loss_and_grad(X, y, params, W, b, cw)
    h = 1e-6

    def loss(pp, WW, bb):
        return circ.loss_and_grad(X, y, pp, WW, bb, cw)[0]

    for p in rng.choice(circ.n_params, 8, replace=False):
        e = np.zeros_like(params)
        e[p] = h
        fd = (loss(params + e, W, b) - loss(params - e, W, b)) / (2 * h)
        assert abs(grad[p] - fd) < 1e-6, (p, grad[p], fd)
    E = np.zeros_like(W)
    E[1, 2] = h
    assert abs(dW[1, 2] - (loss(params, W + E, b) - loss(params, W - E, b)) / (2 * h)) < 1e-6
    e = np.zeros(3)
    e[0] = h
    assert abs(db[0] - (loss(params, W, b + e) - loss(params, W, b - e)) / (2 * h)) < 1e-6


def test_trajectory_averaged_gradient():
    """Classifier gradient when each sample is a mixed state given by n_traj trajectories."""
    import classifiers as CL
    circ = CL.build("hea", 2)
    rng = np.random.default_rng(8)
    n, T = 4, 3
    S = rng.normal(size=(n * T, circ.dim)) + 1j * rng.normal(size=(n * T, circ.dim))
    S /= np.linalg.norm(S, axis=1, keepdims=True)          # trajectory states as initial states
    X = np.zeros((n * T, 1))
    y = np.array([0, 1, 0, 1])
    params = rng.normal(size=circ.n_params)
    W, b, cw = rng.normal(size=(2, circ.readout.shape[0])) * 0.3, np.zeros(2), np.ones(2)
    _, grad, _, _ = circ.loss_and_grad(X, y, params, W, b, cw, init=S, n_traj=T)
    h = 1e-6
    for p in rng.choice(circ.n_params, 6, replace=False):
        e = np.zeros_like(params)
        e[p] = h
        lp = circ.loss_and_grad(X, y, params + e, W, b, cw, init=S, n_traj=T)[0]
        lm = circ.loss_and_grad(X, y, params - e, W, b, cw, init=S, n_traj=T)[0]
        assert abs(grad[p] - (lp - lm) / (2 * h)) < 1e-6


# ── 3. noise trajectories vs exact channel ──────────────────────────
_PAULI = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1.0, -1.0])]


def _op_on(mats, n):
    """Kronecker product placing 2x2 matrices on given qubits (wire 0 = most significant)."""
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, mats.get(q, np.eye(2)))
    return out


def _exact_noisy_rho(circ, params, p2, p1):
    """Density matrix of the AAE circuit with depolarizing noise after every gate."""
    n, dim = circ.n, circ.dim
    rho = np.zeros((dim, dim), complex)
    rho[0, 0] = 1.0
    for op in circ.ops:
        if op[0] == "R":
            _, axis, q, pi = op
            # gate as a matrix: apply the simulator's kernel to the identity
            U = circ._rot(np.eye(dim, dtype=complex), axis, q, np.full(dim, params[pi])).T
            rho = U @ rho @ U.conj().T
            out = (1 - p1) * rho
            for k in (1, 2, 3):
                P = _op_on({q: _PAULI[k]}, n)
                out += p1 / 3 * P @ rho @ P.conj().T
            rho = out
        else:
            U = circ._fixed(np.eye(dim, dtype=complex), op).T
            rho = U @ rho @ U.conj().T
            out = (1 - p2) * rho
            for ka in range(4):
                for kb in range(4):
                    if ka == kb == 0:
                        continue
                    P = _op_on({op[1]: _PAULI[ka], op[2]: _PAULI[kb]}, n)
                    out += p2 / 15 * P @ rho @ P.conj().T
            rho = out
    return rho


def test_noise_trajectories_match_channel():
    """<t|rho|t> from 4000 trajectories agrees with the density-matrix result (5 sigma)."""
    genome = [{"kind": "rot", "gate": "RY"},
              {"kind": "ent", "gate": "CNOT_ring", "stride": 1, "step": 1, "offset": 0},
              {"kind": "rot", "gate": "RY"}]
    circ = AAECircuit(genome, n_qubits=4)
    rng = np.random.default_rng(2)
    params = rng.uniform(0, 2 * np.pi, (1, circ.n_params))
    t = circ.forward(params)[0].astype(complex)           # noiseless output as the target
    p2, p1 = 0.05, 0.01
    rho = _exact_noisy_rho(circ, params[0], p2, p1)
    exact = float(np.real(t.conj() @ rho @ t))
    n_traj = 4000
    S = NZ.noisy_aae_states(circ, params, p2, p1, n_traj, seed=0)
    f = np.abs(S @ t.conj()) ** 2
    est, se = f.mean(), f.std() / np.sqrt(n_traj)
    assert abs(est - exact) < 5 * se + 1e-12, (est, exact, se)
    assert exact < 0.99, "noise had no effect; test is not informative"


# ── 4. mixed kernel ─────────────────────────────────────────────────
def test_mixed_kernel_is_trace_overlap():
    """Average over trajectory pairs of |<a|b>|^2 equals Tr(rho_a rho_b)."""
    rng = np.random.default_rng(3)
    n_traj, dim = 5, 16
    SA = rng.normal(size=(2 * n_traj, dim)) + 1j * rng.normal(size=(2 * n_traj, dim))
    SB = rng.normal(size=(3 * n_traj, dim)) + 1j * rng.normal(size=(3 * n_traj, dim))
    SA /= np.linalg.norm(SA, axis=1, keepdims=True)
    SB /= np.linalg.norm(SB, axis=1, keepdims=True)
    K = NZ.mixed_kernel(SA, SB, n_traj)

    def rho(S, i):
        V = S[i * n_traj:(i + 1) * n_traj]
        return V.T @ V.conj() / n_traj

    for i in range(2):
        for j in range(3):
            ref = np.real(np.trace(rho(SA, i) @ rho(SB, j)))
            assert abs(K[i, j] - ref) < 1e-12


# ── 5. swap test ────────────────────────────────────────────────────
def test_swap_test_unbiased():
    """Mean of many swap-test estimates converges to the true kernel entries."""
    K = np.array([[1.0, 0.3], [0.3, 1.0]])
    rng = np.random.default_rng(4)
    est = np.mean([NZ.swap_test_kernel(K, 1000, rng) for _ in range(2000)], axis=0)
    assert np.allclose(est[0, 1], 0.3, atol=0.002), est


# ── 6. product kernel ───────────────────────────────────────────────
def test_product_kernel_exact():
    """Classical product kernel == fidelity kernel of the full 8-qubit states."""
    import product_kernel_check as PK
    rng = np.random.default_rng(5)
    for name, g in G.references().items():
        if not PK.is_product(g):
            continue
        X = rng.uniform(-np.pi, np.pi, (20, g["n_features"]))
        S = G.encoding_circuit(g).states(X)
        Kq = np.abs(S.conj() @ S.T) ** 2
        phi = PK.qubit_states(g, X)
        Kc = PK.product_kernel(phi, phi)
        assert np.abs(Kq - Kc).max() < 1e-12, name


# ── 7. RealAmplitudes vs Qiskit ─────────────────────────────────────
def test_realamplitudes_matches_qiskit():
    """lit_realamp genome == qiskit.circuit.library.RealAmplitudes (reverse_linear)."""
    try:
        from qiskit.quantum_info import Statevector
        try:                                             # qiskit >= 2.1
            from qiskit.circuit.library import real_amplitudes as RealAmplitudes
        except ImportError:
            from qiskit.circuit.library import RealAmplitudes
    except ImportError:
        skip("qiskit not installed")
    import literature_ansatze as LA
    reps = 2
    circ = AAECircuit(LA.realamp(reps))
    qc = RealAmplitudes(C.N_QUBITS, reps=reps, entanglement="reverse_linear")
    assert qc.num_parameters == circ.n_params
    th = np.random.default_rng(6).uniform(0, 2 * np.pi, circ.n_params)
    ours = circ.forward(th[None])[0]
    # Qiskit's parameter order is layer by layer, qubit 0..7, as in the genome.
    sv = Statevector(qc.assign_parameters(th)).data
    # Qiskit is little-endian (qubit 0 = least significant bit): reverse the bit order.
    n = C.N_QUBITS
    idx = [int(format(k, f"0{n}b")[::-1], 2) for k in range(2 ** n)]
    assert np.allclose(ours, sv[idx].real, atol=1e-10) and np.allclose(sv.imag, 0, atol=1e-10)


# ── plain-Python runner ─────────────────────────────────────────────
if __name__ == "__main__":
    tests = [(k, v) for k, v in dict(globals()).items() if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Skip as e:
            print(f"SKIP  {name}  ({e})")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}  {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed or skipped")
    sys.exit(1 if failed else 0)
