"""Two-qubit convolution (Qcycle) and pooling (Qmask) ansatze.

Parameter counts match the gate table used in the earlier QCNN repo:

  conv : U_TTN 2 | U_5 10 | U_6 10 | U_9 2 | U_13 6 | U_14 6 | U_15 4
         U_SO4 6 | U_SU4 15 | U_gate 9
  pool : pool_U 6 | pool_ansatz 2

Every function has the signature f(state, p, a, b, n) -> state, where `p` is a
1-D tensor slice of exactly N_PARAMS[name] angles, `a` and `b` are wire indices
and `n` is the register size. Pooling functions treat `a` as the qubit being
discarded (the control) and `b` as the survivor.
"""
from __future__ import annotations

from qsim_c import (apply_1q, apply_cnot, apply_ctrl_1q, apply_cz,
                    m_h, m_rx, m_ry, m_rz, m_u3, m_x)

# ---------------------------------------------------------------------------
CONV_PARAMS = {"U_TTN": 2, "U_5": 10, "U_6": 10, "U_9": 2, "U_13": 6,
               "U_14": 6, "U_15": 4, "U_SO4": 6, "U_SU4": 15, "U_gate": 9}
POOL_PARAMS = {"pool_U": 6, "pool_ansatz": 2}

CONV_TYPES = tuple(CONV_PARAMS)
POOL_TYPES = tuple(POOL_PARAMS)


# ---------------------------------------------------------------------------
# convolution ansatze
# ---------------------------------------------------------------------------
def U_TTN(s, p, a, b, n):
    s = apply_1q(s, m_ry(p[0]), a, n)
    s = apply_1q(s, m_ry(p[1]), b, n)
    return apply_cnot(s, a, b, n)


def _u5_u6(s, p, a, b, n, ctrl):
    s = apply_1q(s, m_rx(p[0]), a, n)
    s = apply_1q(s, m_rx(p[1]), b, n)
    s = apply_1q(s, m_rz(p[2]), a, n)
    s = apply_1q(s, m_rz(p[3]), b, n)
    s = apply_ctrl_1q(s, ctrl(p[4]), b, a, n)
    s = apply_ctrl_1q(s, ctrl(p[5]), a, b, n)
    s = apply_1q(s, m_rx(p[6]), a, n)
    s = apply_1q(s, m_rx(p[7]), b, n)
    s = apply_1q(s, m_rz(p[8]), a, n)
    return apply_1q(s, m_rz(p[9]), b, n)


def U_5(s, p, a, b, n):
    return _u5_u6(s, p, a, b, n, m_rz)


def U_6(s, p, a, b, n):
    return _u5_u6(s, p, a, b, n, m_rx)


def U_9(s, p, a, b, n):
    s = apply_1q(s, m_h(s.device), a, n)
    s = apply_1q(s, m_h(s.device), b, n)
    s = apply_cz(s, a, b, n)
    s = apply_1q(s, m_rx(p[0]), a, n)
    return apply_1q(s, m_rx(p[1]), b, n)


def _u13_u14(s, p, a, b, n, ctrl):
    s = apply_1q(s, m_ry(p[0]), a, n)
    s = apply_1q(s, m_ry(p[1]), b, n)
    s = apply_ctrl_1q(s, ctrl(p[2]), b, a, n)
    s = apply_1q(s, m_ry(p[3]), a, n)
    s = apply_1q(s, m_ry(p[4]), b, n)
    return apply_ctrl_1q(s, ctrl(p[5]), a, b, n)


def U_13(s, p, a, b, n):
    return _u13_u14(s, p, a, b, n, m_rz)


def U_14(s, p, a, b, n):
    return _u13_u14(s, p, a, b, n, m_rx)


def U_15(s, p, a, b, n):
    s = apply_1q(s, m_ry(p[0]), a, n)
    s = apply_1q(s, m_ry(p[1]), b, n)
    s = apply_cnot(s, b, a, n)
    s = apply_1q(s, m_ry(p[2]), a, n)
    s = apply_1q(s, m_ry(p[3]), b, n)
    return apply_cnot(s, a, b, n)


def U_SO4(s, p, a, b, n):
    """A 6-parameter SO(4) block: local RYs, CNOT, local RYs, CNOT, local RYs."""
    s = apply_1q(s, m_ry(p[0]), a, n)
    s = apply_1q(s, m_ry(p[1]), b, n)
    s = apply_cnot(s, a, b, n)
    s = apply_1q(s, m_ry(p[2]), a, n)
    s = apply_1q(s, m_ry(p[3]), b, n)
    s = apply_cnot(s, a, b, n)
    s = apply_1q(s, m_ry(p[4]), a, n)
    return apply_1q(s, m_ry(p[5]), b, n)


def U_SU4(s, p, a, b, n):
    """15-parameter universal two-qubit gate: U3 x2, the canonical 3-CNOT core
    with RZ/RY/RZ interleaved, then U3 x2."""
    s = apply_1q(s, m_u3(p[0], p[1], p[2]), a, n)
    s = apply_1q(s, m_u3(p[3], p[4], p[5]), b, n)
    s = apply_cnot(s, b, a, n)
    s = apply_1q(s, m_rz(p[6]), a, n)
    s = apply_1q(s, m_ry(p[7]), b, n)
    s = apply_cnot(s, a, b, n)
    s = apply_1q(s, m_ry(p[8]), b, n)
    s = apply_cnot(s, b, a, n)
    s = apply_1q(s, m_u3(p[9], p[10], p[11]), a, n)
    return apply_1q(s, m_u3(p[12], p[13], p[14]), b, n)


def U_gate(s, p, a, b, n):
    """U3 on each wire plus a controlled-U3 (9 parameters)."""
    s = apply_1q(s, m_u3(p[0], p[1], p[2]), a, n)
    s = apply_1q(s, m_u3(p[3], p[4], p[5]), b, n)
    return apply_ctrl_1q(s, m_u3(p[6], p[7], p[8]), a, b, n)


CONV_FN = {"U_TTN": U_TTN, "U_5": U_5, "U_6": U_6, "U_9": U_9, "U_13": U_13,
           "U_14": U_14, "U_15": U_15, "U_SO4": U_SO4, "U_SU4": U_SU4,
           "U_gate": U_gate}


# ---------------------------------------------------------------------------
# pooling ansatze  (a = discarded / control, b = survivor)
# ---------------------------------------------------------------------------
def pool_U(s, p, a, b, n):
    """Controlled-U3 in both branches: one rotation applied when the discarded
    qubit is |1>, a different one when it is |0>."""
    s = apply_ctrl_1q(s, m_u3(p[0], p[1], p[2]), a, b, n)
    return apply_ctrl_1q(s, m_u3(p[3], p[4], p[5]), a, b, n, anti=True)


def pool_ansatz(s, p, a, b, n):
    """CRZ on the |1> branch, CRX on the |0> branch (2 parameters)."""
    s = apply_ctrl_1q(s, m_rz(p[0]), a, b, n)
    return apply_ctrl_1q(s, m_rx(p[1]), a, b, n, anti=True)


POOL_FN = {"pool_U": pool_U, "pool_ansatz": pool_ansatz}


# ---------------------------------------------------------------------------
def conv_pairs(active, pattern: str):
    """Wire pairs a convolution touches, given the currently active qubits."""
    k = len(active)
    if k < 2:
        return []
    if pattern == "linear":                       # (0,1) (2,3) ...
        return [(active[i], active[i + 1]) for i in range(0, k - 1, 2)]
    if pattern == "ring":                         # (0,1) (1,2) ... (k-1,0)
        return [(active[i], active[(i + 1) % k]) for i in range(k)] if k > 2 \
            else [(active[0], active[1])]
    if pattern == "brick":                        # even offsets then odd
        even = [(active[i], active[i + 1]) for i in range(0, k - 1, 2)]
        odd = [(active[i], active[i + 1]) for i in range(1, k - 1, 2)]
        return even + odd
    raise ValueError(pattern)


PATTERNS = ("linear", "ring", "brick")


def pool_pairs(active):
    """Halving pool: pair adjacent active qubits, discard the first of each."""
    return [(active[i], active[i + 1]) for i in range(0, len(active) - 1, 2)]
