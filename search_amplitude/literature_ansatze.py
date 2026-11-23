# ----------------------------------------------------------------
# literature_ansatze.py  —  standard circuit ansätze from the literature,
# expressed in the project's genome format, used as matched-cost
# baselines for the amplitude-encoder search (objective_study.py).
#
# All use RY rotations and CNOTs only (real amplitudes), like the searched
# circuits, so they are fitted per image with the same optimiser.
#
#   hea_ry   Hardware-efficient ansatz as used for approximate amplitude
#            encoding by Nakaji et al., Phys. Rev. Research 4, 023136 (2022),
#            Sec. II A / Fig. 1 (after Kandala et al., Nature 549, 242 (2017)):
#            each layer = RY on every qubit, then CNOTs on adjacent qubits
#            (open chain 0-1, 1-2, ..., 6-7; 7 CNOTs). A final RY layer is
#            added. The CNOT order within a layer is taken as 0-1, 1-2, ...
#   realamp  Qiskit RealAmplitudes with its default "reverse_linear"
#            entanglement (checked against qiskit 2.5): RY layers and CNOTs
#            (6,7), (5,6), ..., (0,1) per repetition, final RY layer
#            (7 CNOTs per repetition).
#   alt      Alternating layered ansatz (Cerezo et al., Nat. Commun. 12, 1791
#            (2021); Nakaji & Yamamoto, Quantum 5, 434 (2021)) with blocks of
#            m = 2 qubits and block depth 2, real-valued (RY + CNOT) version:
#            odd layers act on pairs (0,1),(2,3),(4,5),(6,7); even layers are
#            shifted by one qubit, (1,2),(3,4),(5,6), with qubits 0 and 7 as
#            single-qubit end blocks. Each layer = 2 x [RY on every qubit,
#            CNOT inside each block] (8 CNOTs in odd, 6 in even layers).
#            A final RY layer is added.
#
# Depths are chosen to span the CNOT counts of the searched circuits
# (about 15-20), plus one deep hea_ry circuit at the depth Nakaji et al.
# used for 5 qubits (8 layers), as a high-budget reference.
# ----------------------------------------------------------------

RY = {"kind": "rot", "gate": "RY"}


def _ent(offset, step, reverse=False, stride=1):
    """Open-boundary CNOT chain layer (reverse=True applies the gates in reverse order)."""
    return {"kind": "ent", "gate": "CNOT_ring", "stride": stride, "step": step,
            "offset": offset, "boundary": "open", "reverse": reverse}


def hea_ry(layers):
    """Hardware-efficient RY/CNOT-chain ansatz with `layers` entangling layers (Nakaji et
    al. 2022).
    """
    g = []
    for _ in range(layers):
        g += [dict(RY), _ent(0, 1)]
    return g + [dict(RY)]


def realamp(reps):
    """Qiskit RealAmplitudes with reverse_linear entanglement and `reps` repetitions."""
    g = []
    for _ in range(reps):
        g += [dict(RY), _ent(0, 1, reverse=True)]
    return g + [dict(RY)]


def alt(layers, block_depth=2):
    """Alternating layered ansatz (Cerezo et al. 2021; Nakaji & Yamamoto 2021): blocks of
    depth `block_depth` on qubit pairs, shifted by one qubit in every other layer.
    """
    g = []
    for layer in range(layers):
        offset = 0 if layer % 2 == 0 else 1
        for _ in range(block_depth):
            g += [dict(RY), _ent(offset, 2)]
    return g + [dict(RY)]


FAMILIES = {
    "hea_ry": {"label": "Hardware-efficient RY/CNOT chain (Nakaji et al. 2022)", "build": hea_ry},
    "realamp": {"label": "Qiskit RealAmplitudes, reverse-linear", "build": realamp},
    "alt": {"label": "Alternating layered ansatz, m=2 (Cerezo et al. 2021)", "build": alt},
}

# (family, depth) pairs used by default
DEFAULT_SET = [("hea_ry", 2), ("hea_ry", 3), ("hea_ry", 4), ("hea_ry", 8),
               ("realamp", 2), ("realamp", 3),
               ("alt", 2), ("alt", 3), ("alt", 4)]


def literature_set(pairs=None):
    """{name: {"family", "depth", "label", "genome"}} for the chosen (family, depth) pairs."""
    out = {}
    for fam, d in (pairs or DEFAULT_SET):
        out[f"lit_{fam}_{d}"] = {"family": fam, "depth": d, "label": FAMILIES[fam]["label"],
                                 "genome": FAMILIES[fam]["build"](d)}
    return out
