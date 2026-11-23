"""Batched real-valued statevector simulator with PER-SAMPLE angles.

Every gate used here (RY, CRY, CNOT, CZ) has a real matrix, so a real initial
state stays real throughout: we run in float32 and still backprop through the
angles with autograd.

The key feature for approximate amplitude encoding (AAE) is that each sample in
the batch carries its OWN angle vector. Fitting B images means one batched
optimisation of shape (B, n_params) rather than B sequential fits -- which is
what makes this fast on a single GPU.

Conventions
-----------
* state: real tensor (B, 2**n).
* qubit 0 is the MOST significant bit: bit_q(i) = (i >> (n-1-q)) & 1.
* all gates are norm preserving.
"""
from __future__ import annotations

import torch

_idx_cache: dict = {}


# ---------------------------------------------------------------------------
# index helpers (cached per (n, qubits, device))
# ---------------------------------------------------------------------------
def _cnot_perm(n: int, c: int, t: int, device) -> torch.Tensor:
    key = ("cnot", n, c, t, str(device))
    v = _idx_cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cbit = (idx >> (n - 1 - c)) & 1
        v = torch.where(cbit.bool(), idx ^ (1 << (n - 1 - t)), idx)
        _idx_cache[key] = v
    return v


def _cz_phase(n: int, c: int, t: int, device) -> torch.Tensor:
    key = ("cz", n, c, t, str(device))
    v = _idx_cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cbit = (idx >> (n - 1 - c)) & 1
        tbit = (idx >> (n - 1 - t)) & 1
        v = torch.where((cbit & tbit).bool(),
                        torch.tensor(-1.0, device=device),
                        torch.tensor(1.0, device=device))
        _idx_cache[key] = v
    return v


def _cry_pairs(n: int, c: int, t: int, device):
    """Indices (i0, i1) of basis states with control=1, target 0 vs 1."""
    key = ("cry", n, c, t, str(device))
    v = _idx_cache.get(key)
    if v is None:
        idx = torch.arange(1 << n, device=device)
        cbit = (idx >> (n - 1 - c)) & 1
        tbit = (idx >> (n - 1 - t)) & 1
        i0 = idx[(cbit == 1) & (tbit == 0)]
        i1 = i0 ^ (1 << (n - 1 - t))
        v = (i0, i1)
        _idx_cache[key] = v
    return v


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------
def apply_ry(state: torch.Tensor, theta: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """RY(theta) on qubit q. theta is (B,) -- one angle per sample."""
    B = state.shape[0]
    c = torch.cos(theta / 2).view(B, 1, 1, 1)
    s = torch.sin(theta / 2).view(B, 1, 1, 1)
    left, right = 1 << q, 1 << (n - 1 - q)
    v = state.view(B, left, 2, right)
    a, b = v[:, :, 0:1, :], v[:, :, 1:2, :]
    out = torch.cat([c * a - s * b, s * a + c * b], dim=2)
    return out.reshape(B, 1 << n)


def apply_cry(state: torch.Tensor, theta: torch.Tensor, c_q: int, t_q: int, n: int) -> torch.Tensor:
    """Controlled-RY: RY(theta) on target, only where control qubit is |1>."""
    i0, i1 = _cry_pairs(n, c_q, t_q, state.device)
    B = state.shape[0]
    co = torch.cos(theta / 2).view(B, 1)
    si = torch.sin(theta / 2).view(B, 1)
    a = state[:, i0]
    b = state[:, i1]
    out = state.clone()
    out[:, i0] = co * a - si * b
    out[:, i1] = si * a + co * b
    return out


def apply_cnot(state: torch.Tensor, c: int, t: int, n: int) -> torch.Tensor:
    return state[:, _cnot_perm(n, c, t, state.device)]


def apply_cz(state: torch.Tensor, c: int, t: int, n: int) -> torch.Tensor:
    return state * _cz_phase(n, c, t, state.device)


def zero_state(B: int, n: int, device) -> torch.Tensor:
    """|0...0> replicated across the batch."""
    s = torch.zeros(B, 1 << n, device=device)
    s[:, 0] = 1.0
    return s


def frobenius(target: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """||t - s||_F per sample -> (B,). Identical to the flattened matrix norm."""
    return torch.linalg.norm(target - state, dim=1)
