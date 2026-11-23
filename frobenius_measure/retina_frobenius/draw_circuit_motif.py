"""Draw a Stage 1 encoding genome as a motif/architecture diagram: a sequence
of labeled layer-blocks (rotation / entangle), the way a classical network's
layer stack gets sketched -- rather than a literal gate-by-gate circuit
(that's draw_circuit.py).

Unlike Stage 2's QCNN (which pools qubits away layer by layer), Stage 1's
genome keeps all n_qubits active the whole way through -- there is no
shrinking register to draw here, just an alternating sequence of rotation
layers and entangling layers of varying width. This diagram makes that
sequence-and-cost structure legible at a glance on genomes too deep to read
gate-by-gate (r_seed3 here is 29 layers / 78 drawing columns of literal
gates).

Examples
--------
    python draw_circuit_motif.py --genome experiments/r_seed3/best_genome.json \
        --out figures/r_seed3_motif.png

    python draw_circuit_motif.py --result experiments/r_seed3/search_result.json \
        --out figures/r_seed3_motif.png
"""
from __future__ import annotations

import argparse
import json

from draw_circuit import load_genome, stats

C_INK = "#17212B"
C_ROT = "#5A4BD6"
C_ROT_FILL = "#EDEAFB"
C_ENT = "#0E9384"
C_ENT_FILL = "#E3F5F2"
C_ARROW = "#94A3B8"


def layer_blocks(g: dict):
    """One block per genome layer: {"kind": "rot"|"ent", "label": ..., "sub": ...}."""
    n = g["n_qubits"]
    blocks = []
    for l in g["layers"]:
        if l["type"] == "rot":
            blocks.append({"kind": "rot", "label": "R", "sub": f"{n}×RY"})
        else:
            pairs = l["pairs"]
            cnot = sum(1 for p in pairs if p[2] == "cnot")
            cz = sum(1 for p in pairs if p[2] == "cz")
            cry = sum(1 for p in pairs if p[2] == "cry")
            parts = []
            if cnot:
                parts.append(f"{cnot}×CNOT")
            if cz:
                parts.append(f"{cz}×CZ")
            if cry:
                parts.append(f"{cry}×CRY")
            blocks.append({"kind": "ent", "label": "E", "sub": "\n".join(parts)})
    return blocks


def draw(g: dict, blocks, out, per_row=12, title=None, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrow, FancyBboxPatch

    rows = [blocks[i:i + per_row] for i in range(0, len(blocks), per_row)] or [[]]
    nrows = len(rows)

    box_w, box_h, gap = 1.05, 0.95, 0.55
    fig_w = max(6.0, 1.0 + (box_w + gap) * min(len(blocks), per_row))
    fig_h = nrows * (box_h + 0.9) + 0.7
    fig, axes = plt.subplots(nrows, 1, figsize=(fig_w, fig_h), squeeze=False)

    for r, row in enumerate(rows):
        ax = axes[r][0]
        ncol = max(len(rr) for rr in rows)
        ax.set_xlim(-0.4, (box_w + gap) * ncol + 0.2)
        ax.set_ylim(-0.7, box_h + 0.3)
        ax.axis("off")

        for i, blk in enumerate(row):
            x = i * (box_w + gap)
            fc = C_ROT_FILL if blk["kind"] == "rot" else C_ENT_FILL
            ec = C_ROT if blk["kind"] == "rot" else C_ENT
            ax.add_patch(FancyBboxPatch(
                (x, 0), box_w, box_h,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                fc=fc, ec=ec, lw=1.3, zorder=2))
            ax.text(x + box_w / 2, box_h * 0.68, blk["label"], ha="center", va="center",
                    fontsize=11, color=ec, fontweight="bold", zorder=3)
            ax.text(x + box_w / 2, box_h * 0.28, blk["sub"], ha="center", va="center",
                    fontsize=6, color=C_INK, zorder=3)
            layer_idx = r * per_row + i
            ax.text(x + box_w / 2, box_h + 0.14, str(layer_idx), ha="center", va="bottom",
                    fontsize=6.5, color=C_ARROW)
            if i < len(row) - 1:
                ax.add_patch(FancyArrow(
                    x + box_w + 0.08, box_h / 2, gap - 0.16, 0,
                    width=0.02, head_width=0.14, head_length=0.12,
                    length_includes_head=True, fc=C_ARROW, ec=C_ARROW, zorder=1))

        if nrows > 1:
            ax.text((box_w + gap) * ncol, -0.55, f"({r + 1}/{nrows})", fontsize=7,
                    color=C_ARROW, ha="right", va="center")

    s = stats(g)
    ttl = title or (f"{s['n']} qubits (constant -- Stage 1 has no pooling)   |   "
                    f"{s['depth']} layers ({s['rot_layers']}R + "
                    f"{s['depth'] - s['rot_layers']}E)   |   {s['gates']} two-qubit gates   |   "
                    f"{s['params']} parameters")
    fig.suptitle(ttl, fontsize=9, color=C_INK, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"        {ttl}")


def main():
    ap = argparse.ArgumentParser(description="Draw a Stage 1 genome as a layer-block architecture diagram.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--genome", help="path to a genome JSON (e.g. best_genome.json)")
    src.add_argument("--result", help="path to a search_result.json")
    ap.add_argument("--gates", type=int, default=None,
                    help="with --result: pick the candidate nearest this gate count")
    ap.add_argument("--out", default="circuit_motif.png")
    ap.add_argument("--per-row", type=int, default=12, help="layer blocks per row before wrapping")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    g = load_genome(a)
    blocks = layer_blocks(g)
    s = stats(g)
    print(f"[genome] {s['n']} qubits, {s['depth']} layers "
          f"({s['rot_layers']} rot, {s['depth'] - s['rot_layers']} ent), "
          f"{s['gates']} 2q gates")
    draw(g, blocks, a.out, per_row=a.per_row, title=a.title, dpi=a.dpi)


if __name__ == "__main__":
    main()
