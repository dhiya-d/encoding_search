"""Approximate amplitude encoding (AAE) with the Frobenius objective.

For each image target t (a 256-dim L2-normalised vector, i.e. the flattened
16x16 matrix) we fit the circuit's angles so that

        s = U(theta) |0...0>        minimises    ||t - s||_F

Every image gets its OWN angle vector -- that is what AAE means -- but all
images in the batch are fitted SIMULTANEOUSLY as one (B, n_params) tensor, so
the whole thing is a single batched optimisation on the GPU.

Note the identity: for unit vectors, ||t - s||^2 = 2 - 2<t,s>. So minimising the
Frobenius distance is equivalent to maximising the SIGNED overlap <t,s>. Unlike
fidelity |<t|s>|^2 it is sign sensitive: s = -t scores 2 (worst), not 0 (best).
"""
from __future__ import annotations

import torch

from genome import Genome
from qsim import apply_cnot, apply_cry, apply_cz, apply_ry, frobenius, zero_state


def run_circuit(genome: Genome, params: torch.Tensor, B: int, device) -> torch.Tensor:
    """Apply U(theta) to |0...0>. params is (B, n_params) -- per-sample angles."""
    n = genome.n_qubits
    state = zero_state(B, n, device)
    pi = 0
    for layer in genome.layers:
        if layer["type"] == "rot":
            for q in range(n):
                state = apply_ry(state, params[:, pi], q, n)
                pi += 1
        else:
            for (c, t, gate) in layer["pairs"]:
                c, t = int(c), int(t)
                if gate == "cnot":
                    state = apply_cnot(state, c, t, n)
                elif gate == "cz":
                    state = apply_cz(state, c, t, n)
                else:                                   # cry -- parameterised
                    state = apply_cry(state, params[:, pi], c, t, n)
                    pi += 1
    return state


def fit_batch(genome: Genome, targets: torch.Tensor, steps: int = 150,
              lr: float = 0.08, restarts: int = 1, seed: int = 0,
              device: str = "cpu", return_states: bool = False):
    """Fit per-image angles to minimise ||t - U(theta)|0>||_F.

    Returns dict with per-image frobenius distances (B,), their mean, and
    optionally the encoded states and fitted angles.
    """
    targets = targets.to(device)
    B = targets.shape[0]
    n_p = genome.n_params()

    best_d = None
    best_state = None
    best_params = None

    for r in range(max(1, restarts)):
        g = torch.Generator(device="cpu").manual_seed(seed + 1000 * r)
        theta = (torch.rand(B, max(n_p, 1), generator=g) * 2 * torch.pi).to(device)
        theta.requires_grad_(n_p > 0)

        if n_p > 0:
            opt = torch.optim.Adam([theta], lr=lr)
            for _ in range(steps):
                opt.zero_grad()
                s = run_circuit(genome, theta, B, device)
                loss = frobenius(targets, s).mean()
                loss.backward()
                opt.step()

        with torch.no_grad():
            s = run_circuit(genome, theta, B, device)
            d = frobenius(targets, s)

        if best_d is None:
            best_d, best_state, best_params = d, s, theta.detach()
        else:
            better = d < best_d
            best_d = torch.where(better, d, best_d)
            best_state = torch.where(better.unsqueeze(1), s, best_state)
            best_params = torch.where(better.unsqueeze(1), theta.detach(), best_params)

    out = {
        "frobenius": best_d.detach().cpu(),
        "mean_frobenius": float(best_d.mean().detach().cpu()),
        "max_frobenius": float(best_d.max().detach().cpu()),
        # overlap recovered from the distance: <t,s> = 1 - d^2/2
        "mean_overlap": float((1 - best_d.detach() ** 2 / 2).mean().cpu()),
    }
    if return_states:
        out["states"] = best_state.detach().cpu()
        out["params"] = best_params.cpu()
    return out
