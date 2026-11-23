"""Build, run and train a QCNN from a genome.

Readout: the surviving qubit(s) (--target-qubits) give 2**target_qubits
measurement outcomes, mapped directly onto that many classes -- for
BreastMNIST's binary task, target_qubits=1 gives exactly 2 outcomes. Because
the outcome distribution is already a normalised probability vector, training
minimises the (class
weighted) negative log-likelihood of that distribution -- no classical head is
needed. A small linear head is available via `use_head=True` for comparison, but
the direct mapping is the default because it keeps the model genuinely quantum
end-to-end and has no extra parameters to confound the architecture comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score

from ansatz import CONV_FN, CONV_PARAMS, POOL_FN, POOL_PARAMS
from qcnn_genome import QCNNGenome
from qsim_c import marginal_probs, to_complex_state

EPS = 1e-9


@dataclass
class TrainConfig:
    epochs: int = 40
    lr: float = 0.003
    batch_size: int = 32
    seed: int = 0
    use_head: bool = False
    l1_gate: float = 5e-4        # complexity penalty on entangling-gate count
    l2_param: float = 1e-3       # complexity penalty on parameter count


# ---------------------------------------------------------------------------
def forward(genome: QCNNGenome, params: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """x: (B, 2**n) real amplitudes -> (B, 4) outcome probabilities."""
    n = genome.n_qubits
    s = to_complex_state(x)
    sched, _ = genome.schedule()
    final = genome.readout_wires()
    i = 0
    for layer, st in zip(genome.layers, sched):
        cfn = CONV_FN[layer["conv"]]
        cnp = CONV_PARAMS[layer["conv"]]
        for (a, b) in st["conv_pairs"]:
            s = cfn(s, params[i:i + cnp], a, b, n)
            i += cnp
        pfn = POOL_FN[layer["pool_type"]]
        pnp = POOL_PARAMS[layer["pool_type"]]
        for stage in st["pool_stages"]:
            for (a, b) in stage:
                s = pfn(s, params[i:i + pnp], a, b, n)
                i += pnp
    return marginal_probs(s, final, n)


def init_params(genome: QCNNGenome, seed: int, device) -> torch.Tensor:
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_p = max(genome.n_params(), 1)
    return (torch.rand(n_p, generator=g) * 2 * np.pi).to(device).requires_grad_(True)


def class_weights(y: torch.Tensor, n_classes: int, device,
                  n_outcomes: int | None = None) -> torch.Tensor:
    """Inverse-frequency weight per class, optionally padded out to
    n_outcomes (2**target_qubits) when that's wider than n_classes -- the
    direct (no-head) readout treats every outcome as a class index, so
    nll_loss's weight tensor must match that width even though the extra
    outcomes never appear in `y` (their padding value is never read)."""
    cnt = torch.bincount(y, minlength=n_classes).float().clamp(min=1.0)
    w = cnt.sum() / (n_classes * cnt)
    if n_outcomes is not None and n_outcomes > n_classes:
        w = torch.cat([w, torch.ones(n_outcomes - n_classes)])
    return w.to(device)


# ---------------------------------------------------------------------------
def train_and_score(genome: QCNNGenome, data: dict, cfg: TrainConfig,
                    device: str = "cpu", n_classes: int = 4,
                    return_params: bool = False) -> dict:
    """Train on data['Xtr']/['ytr'], score balanced accuracy on data['Xval'].

    Returns bal_acc plus the complexity terms and the penalised fitness used by
    the single-objective search.
    """
    torch.manual_seed(cfg.seed)
    Xtr = torch.as_tensor(data["Xtr"], dtype=torch.float32, device=device)
    ytr = torch.as_tensor(data["ytr"], dtype=torch.long, device=device)
    Xva = torch.as_tensor(data["Xval"], dtype=torch.float32, device=device)
    yva = np.asarray(data["yval"])

    n_outcomes = 1 << len(genome.readout_wires())
    params = init_params(genome, cfg.seed, device)
    head = torch.nn.Linear(n_outcomes, n_classes).to(device) if cfg.use_head else None
    plist = [params] + ([head.weight, head.bias] if head else [])
    opt = torch.optim.Adam(plist, lr=cfg.lr)
    # no-head path maps outcomes directly onto class indices via nll_loss, so
    # the weight vector must span all n_outcomes; the head path's loss is
    # over the head's own n_classes-wide output, so no padding needed there.
    cw = class_weights(ytr, n_classes, device, n_outcomes=None if head else n_outcomes)

    N = Xtr.shape[0]
    bs = min(cfg.batch_size, N)
    losses = []
    for ep in range(cfg.epochs):
        perm = torch.randperm(N, device=device)
        ep_loss = 0.0
        for k in range(0, N, bs):
            idx = perm[k:k + bs]
            opt.zero_grad()
            probs = forward(genome, params, Xtr[idx])
            if head is not None:
                loss = F.cross_entropy(head(probs), ytr[idx], weight=cw)
            else:
                loss = F.nll_loss(torch.log(probs + EPS), ytr[idx], weight=cw)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu()) * len(idx)
        losses.append(ep_loss / N)

    with torch.no_grad():
        pv = forward(genome, params, Xva)
        logits = head(pv) if head is not None else torch.log(pv + EPS)
        pred = logits.argmax(1).cpu().numpy()
    bal = float(balanced_accuracy_score(yva, pred))

    gates, n_p = genome.two_qubit_gates(), genome.n_params()
    penalty = cfg.l1_gate * gates + cfg.l2_param * n_p
    out = {"bal_acc": bal, "gates": gates, "params": n_p,
           "penalty": penalty, "fitness": bal - penalty,
           "final_loss": losses[-1] if losses else None,
           "final_wires": genome.final_wires(),
           "readout_wires": genome.readout_wires()}
    if return_params:
        out["theta"] = params.detach().cpu().numpy()
        if head is not None:
            out["head"] = (head.weight.detach().cpu().numpy(),
                           head.bias.detach().cpu().numpy())
    return out


def predict(genome: QCNNGenome, params: torch.Tensor, X, device="cpu",
            head=None, batch=512) -> np.ndarray:
    preds = []
    with torch.no_grad():
        for k in range(0, len(X), batch):
            xb = torch.as_tensor(X[k:k + batch], dtype=torch.float32, device=device)
            p = forward(genome, params, xb)
            lg = head(p) if head is not None else torch.log(p + EPS)
            preds.append(lg.argmax(1).cpu().numpy())
    return np.concatenate(preds)