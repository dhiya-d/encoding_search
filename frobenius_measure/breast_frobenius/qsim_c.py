"""Complex batched statevector simulator for the QCNN stage.

Difference from the encoder's simulator: QCNN parameters are SHARED across all
samples (one circuit classifies every image), whereas the encoder fits angles
per image. That makes this simpler and faster -- gate matrices are built once
per call and broadcast over the batch.

Complex arithmetic is required here: the QCNN ansatze use RZ/U3/CRZ, which act
trivially on real amplitudes' magnitudes if restricted to reals but do change the
measured distribution once phases are involved. Restricting to a real simulator
would silently remove most of the expressivity of these ansatze.

Conventions
-----------
* state: complex tensor (B, 2**n)
* qubit 0 is the MOST significant bit: bit_q(i) = (i >> (n-1-q)) & 1
"""
from __future__ import annotations

import torch

_cache: dict = {}
CDTYPE = torch.complex64


# ---------------------------------------------------------------------------
# index helpers
# ---------------------------------------------------------------------------
def _ctrl_pairs(n: int, c: int, t: int, device):
    """(i0, i1): basis indices with control=1 and target 0 / 1."""
    key = ("cp", n, c, t, str(device))
    v = _cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cb = (idx >> (n - 1 - c)) & 1
        tb = (idx >> (n - 1 - t)) & 1
        i0 = idx[(cb == 1) & (tb == 0)]
        i1 = i0 ^ (1 << (n - 1 - t))
        v = (i0, i1)
        _cache[key] = v
    return v


def _anti_ctrl_pairs(n: int, c: int, t: int, device):
    """(i0, i1): basis indices with control=0 and target 0 / 1."""
    key = ("ap", n, c, t, str(device))
    v = _cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cb = (idx >> (n - 1 - c)) & 1
        tb = (idx >> (n - 1 - t)) & 1
        i0 = idx[(cb == 0) & (tb == 0)]
        i1 = i0 ^ (1 << (n - 1 - t))
        v = (i0, i1)
        _cache[key] = v
    return v


def _cnot_perm(n: int, c: int, t: int, device):
    key = ("cn", n, c, t, str(device))
    v = _cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cb = (idx >> (n - 1 - c)) & 1
        v = torch.where(cb.bool(), idx ^ (1 << (n - 1 - t)), idx)
        _cache[key] = v
    return v


def _cz_phase(n: int, c: int, t: int, device):
    key = ("czp", n, c, t, str(device))
    v = _cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cb = (idx >> (n - 1 - c)) & 1
        tb = (idx >> (n - 1 - t)) & 1
        v = torch.where((cb & tb).bool(),
                        torch.tensor(-1.0 + 0j, dtype=CDTYPE, device=device),
                        torch.tensor(1.0 + 0j, dtype=CDTYPE, device=device))
        _cache[key] = v
    return v


# ---------------------------------------------------------------------------
# single-qubit matrices (all differentiable in their angles)
# ---------------------------------------------------------------------------
def _c(x):
    return torch.complex(x, torch.zeros_like(x))


def m_rx(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    return torch.stack([torch.stack([_c(c), -1j * _c(s)]),
                        torch.stack([-1j * _c(s), _c(c)])])


def m_ry(t):
    c, s = torch.cos(t / 2), torch.sin(t / 2)
    z = torch.zeros_like(c)
    return torch.stack([torch.stack([_c(c), _c(-s)]),
                        torch.stack([_c(s), _c(c)])]) + 0j * _c(z)


def m_rz(t):
    e0 = torch.exp(-0.5j * t.to(CDTYPE))
    e1 = torch.exp(0.5j * t.to(CDTYPE))
    z = torch.zeros((), dtype=CDTYPE, device=t.device)
    return torch.stack([torch.stack([e0, z]), torch.stack([z, e1])])


def m_u3(theta, phi, lam):
    c, s = torch.cos(theta / 2), torch.sin(theta / 2)
    a = _c(c)
    b = -torch.exp(1j * lam.to(CDTYPE)) * _c(s)
    cc = torch.exp(1j * phi.to(CDTYPE)) * _c(s)
    d = torch.exp(1j * (phi + lam).to(CDTYPE)) * _c(c)
    return torch.stack([torch.stack([a, b]), torch.stack([cc, d])])


def m_h(device):
    r = 1.0 / (2 ** 0.5)
    return torch.tensor([[r, r], [r, -r]], dtype=CDTYPE, device=device)


def m_x(device):
    return torch.tensor([[0, 1], [1, 0]], dtype=CDTYPE, device=device)


# ---------------------------------------------------------------------------
# gate application
# ---------------------------------------------------------------------------
def apply_1q(state, mat, q, n):
    B = state.shape[0]
    left, right = 1 << q, 1 << (n - 1 - q)
    v = state.view(B, left, 2, right)
    out = torch.einsum("kj,bljr->blkr", mat, v)
    return out.reshape(B, 1 << n)


def apply_ctrl_1q(state, mat, c, t, n, anti=False):
    """Apply a 2x2 matrix to target t on the subspace where control c is 1
    (or 0 if anti=True)."""
    i0, i1 = (_anti_ctrl_pairs if anti else _ctrl_pairs)(n, c, t, state.device)
    a, b = state[:, i0], state[:, i1]
    out = state.clone()
    out[:, i0] = mat[0, 0] * a + mat[0, 1] * b
    out[:, i1] = mat[1, 0] * a + mat[1, 1] * b
    return out


def apply_cnot(state, c, t, n):
    return state[:, _cnot_perm(n, c, t, state.device)]


def apply_cz(state, c, t, n):
    return state * _cz_phase(n, c, t, state.device)


# ---------------------------------------------------------------------------
# readout
# ---------------------------------------------------------------------------
def marginal_probs(state, keep, n):
    """Probability distribution over the `keep` qubits, tracing out the rest.

    Returns (B, 2**len(keep)), rows summing to 1.
    """
    B = state.shape[0]
    p = (state.real ** 2 + state.imag ** 2).view((B,) + (2,) * n)
    trace_axes = [1 + q for q in range(n) if q not in keep]
    if trace_axes:
        p = p.sum(dim=trace_axes)
    # reorder remaining axes to the order given in `keep`
    order = sorted(keep)
    if list(keep) != order:
        perm = [0] + [1 + order.index(q) for q in keep]
        p = p.permute(*perm)
    return p.reshape(B, 1 << len(keep))


def to_complex_state(x):
    """Real amplitude vectors (B, 2**n) -> normalised complex state."""
    s = x.to(CDTYPE)
    nrm = torch.sqrt((s.real ** 2 + s.imag ** 2).sum(1, keepdim=True)).clamp(min=1e-9)
    return s / nrm.to(CDTYPE)
