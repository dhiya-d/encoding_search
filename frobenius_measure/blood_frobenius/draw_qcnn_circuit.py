"""Draw a QCNN genome (Stage 2) as a circuit diagram.

Works on any genome JSON written by qcnn_search.py (`best_qcnn.json`), or on
any candidate inside a `qcnn_search_result.json` selected by gate count.

Unlike Stage 1's encoding circuit (draw_circuit.py), a QCNN genome reduces the
register as it goes: each pooling stage discards one qubit per pair. Discarded
wires are drawn as grayed-out, terminated lines with an "x" where they leave
the circuit; the wires still active at the end get a measurement box.

Examples
--------
List what is available in a search result:
    python draw_qcnn_circuit.py --result experiments/qcnn_blood/qcnn_search_result.json --list

Draw the selected best genome:
    python draw_qcnn_circuit.py --genome experiments/qcnn_blood/best_qcnn.json \
        --out figures/best_qcnn_circuit.png

Draw a specific candidate from a saved run's finalists (nearest gate count is used):
    python draw_qcnn_circuit.py --result experiments/qcnn_blood/qcnn_search_result.json \
        --gates 20 --out figures/qcnn_20gates.png

Wide circuits wrap automatically; control with --max-cols.
"""
from __future__ import annotations

import argparse
import json
import os

from qcnn_genome import QCNNGenome, configure

# ---------------------------------------------------------------------------
# palette -- conv reuses Stage 1's rotation violet, pool gets its own warm
# color so the two kinds of layer are visually distinct at a glance.
# ---------------------------------------------------------------------------
C_WIRE = "#41505E"
C_WIRE_DEAD = "#C7CDD4"     # discarded wire tail, after its pool column
C_CONV = "#5A4BD6"
C_CONV_FILL = "#EDEAFB"
C_POOL = "#B45309"
C_POOL_FILL = "#FBEBD9"
C_MEASURE = "#0E9384"
C_INK = "#17212B"

CONV_ABBR = {"U_TTN": "TTN", "U_5": "5", "U_6": "6", "U_9": "9", "U_13": "13",
            "U_14": "14", "U_15": "15", "U_SO4": "SO4", "U_SU4": "SU4",
            "U_gate": "G"}
POOL_ABBR = {"pool_U": "pU", "pool_ansatz": "pA"}


# ---------------------------------------------------------------------------
def load_genome(args) -> dict:
    if args.genome:
        return json.load(open(args.genome))
    res = json.load(open(args.result))
    cands = res.get("finalists") or res.get("candidates") or []
    if not cands:
        raise SystemExit("no candidates found in result file")
    if args.gates is None:
        w = res.get("winner")
        if isinstance(w, dict) and "genome" in w:
            return w["genome"]
        if isinstance(w, dict) and "layers" in w:
            return w
        return cands[-1]["genome"]
    best = min(cands, key=lambda c: abs(c["gates"] - args.gates))
    print(f"[pick] nearest to {args.gates} gates -> {best['gates']} gates, "
          f"acc={best.get('bal_acc', float('nan')):.4f}")
    return best["genome"]


def list_candidates(path):
    res = json.load(open(path))
    cands = res.get("finalists") or res.get("candidates") or []
    print(f"{len(cands)} candidates in {path}\n")
    print(f"  {'gates':>6} {'bal_acc':>8} {'params':>7}   architecture")
    for c in sorted(cands, key=lambda x: x["gates"]):
        print(f"  {c['gates']:>6} {c.get('bal_acc', float('nan')):>8.4f} "
              f"{c.get('params', '-'):>7}   {c.get('summary', '')}")


# ---------------------------------------------------------------------------
# genome -> drawing columns
# ---------------------------------------------------------------------------
def build_qcnn_columns(gd: dict):
    """Walk the genome's actual schedule() (same code path used at train/eval
    time) and flatten it into drawing columns.

    Returns (cols, discarded_at, final_active, n_qubits). `discarded_at` maps
    a wire index to the column it was pooled away in -- used to stop drawing
    that wire's line past that point. `final_active` is the surviving
    (measured) register at the end.
    """
    g = QCNNGenome.from_dict(gd)
    # target_qubits=1 is permissive: schedule() only stops pooling early once
    # active <= target_qubits, and a validly-repaired genome's own pool counts
    # already sum to exactly its true reduction, so this just lets the genome
    # run its own specified schedule through to the end without truncation.
    configure(g.n_qubits, target_qubits=1)
    sched, final_active = g.schedule()
    # re-configure to the genome's ACTUAL survivor count so downstream calls
    # that read the module-level TARGET_QUBITS (e.g. g.summary()) see a
    # consistent value instead of the permissive probe above. Best-effort:
    # an unusual (non-power-of-two) survivor count just falls back to the
    # permissive setting rather than failing the whole render.
    try:
        configure(g.n_qubits, target_qubits=max(len(final_active), 1))
    except ValueError:
        pass

    cols = []
    discarded_at = {}
    for layer, s in zip(g.layers, sched):
        packed = []   # list of (occupied_span, [(a, b), ...])
        for (a, b) in s["conv_pairs"]:
            span = set(range(min(a, b), max(a, b) + 1))
            placed = False
            for occ, items in packed:
                if not (occ & span):
                    occ |= span
                    items.append((a, b))
                    placed = True
                    break
            if not placed:
                packed.append((set(span), [(a, b)]))
        for _, items in packed:
            cols.append({"kind": "conv", "conv": layer["conv"], "gates": items})

        for stage in s["pool_stages"]:
            cols.append({"kind": "pool", "pool_type": layer["pool_type"], "gates": stage})
            ci = len(cols) - 1
            for (a, _b) in stage:
                discarded_at[a] = ci

    return cols, discarded_at, final_active, g.n_qubits


# ---------------------------------------------------------------------------
# matplotlib rendering
# ---------------------------------------------------------------------------
def draw(gd, cols, discarded_at, final_active, n, out, max_cols=30, title=None, dpi=200):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    chunks = [cols[i:i + max_cols] for i in range(0, len(cols), max_cols)] or [[]]
    rows = len(chunks)
    total_cols = len(cols)

    col_w, wire_h = 0.72, 0.5
    fig_w = max(6.4, 1.7 + col_w * min(total_cols, max_cols))
    fig_h = rows * (n * wire_h + 1.0) + 0.6
    fig, axes = plt.subplots(rows, 1, figsize=(fig_w, fig_h), squeeze=False)

    for r, chunk in enumerate(chunks):
        ax = axes[r][0]
        base = r * max_cols
        ncol = max(len(c) for c in chunks)
        ax.set_xlim(-1.35, ncol + 0.85)
        ax.set_ylim(-0.7, n - 0.3)
        ax.invert_yaxis()
        ax.axis("off")

        for q in range(n):
            end_ci = discarded_at.get(q, total_cols - 1)
            # portion of this wire's line that falls within THIS chunk
            local_end = min(len(chunk) - 1, end_ci - base) if end_ci >= base else -1
            wire_end = (local_end + 1.1) if local_end >= 0 else 0.0
            alive_past_chunk = end_ci >= base + len(chunk)
            color = C_WIRE if (alive_past_chunk or q not in discarded_at
                               or end_ci >= base) else C_WIRE_DEAD
            if wire_end > 0 or alive_past_chunk:
                ax.plot([-0.35, max(wire_end, 0.0) if not alive_past_chunk else len(chunk) + 0.1],
                        [q, q], color=color, lw=0.9, zorder=1)
            ax.text(-0.55, q, f"$q_{{{q}}}$", ha="right", va="center",
                    fontsize=8.5, color=C_INK)
            if q in discarded_at and base <= discarded_at[q] < base + len(chunk):
                x_end = (discarded_at[q] - base) + 0.35
                ax.plot([x_end - 0.1, x_end + 0.1], [q - 0.13, q + 0.13], color=C_POOL, lw=1.3, zorder=6)
                ax.plot([x_end - 0.1, x_end + 0.1], [q + 0.13, q - 0.13], color=C_POOL, lw=1.3, zorder=6)

        for ci_local, col in enumerate(chunk):
            x = ci_local + 0.35
            if col["kind"] == "conv":
                label = CONV_ABBR.get(col["conv"], col["conv"][:3])
                for (a, b) in col["gates"]:
                    ax.plot([x, x], [a, b], color=C_CONV, lw=1.2, zorder=2)
                    ax.add_patch(Circle((x, a), 0.06, fc=C_CONV, ec=C_CONV, zorder=4))
                    ax.add_patch(Circle((x, b), 0.06, fc=C_CONV, ec=C_CONV, zorder=4))
                    ymid = (a + b) / 2
                    ax.add_patch(FancyBboxPatch(
                        (x - 0.21, ymid - 0.16), 0.42, 0.32,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc=C_CONV_FILL, ec=C_CONV, lw=0.9, zorder=3))
                    ax.text(x, ymid, label, ha="center", va="center",
                            fontsize=5.6, color=C_CONV, zorder=5, fontweight="bold")
            else:  # pool
                label = POOL_ABBR.get(col["pool_type"], col["pool_type"][:2])
                for (a, b) in col["gates"]:
                    ax.plot([x, x], [a, b], color=C_POOL, lw=1.2, zorder=2)
                    ax.add_patch(Circle((x, b), 0.075, fc=C_POOL, ec=C_POOL, zorder=4))
                    ymid = (a + b) / 2
                    ax.add_patch(FancyBboxPatch(
                        (x - 0.18, ymid - 0.15), 0.36, 0.3,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc=C_POOL_FILL, ec=C_POOL, lw=0.9, zorder=3))
                    ax.text(x, ymid, label, ha="center", va="center",
                            fontsize=5.4, color=C_POOL, zorder=5, fontweight="bold")

        if r == rows - 1:
            for q in final_active:
                x = len(chunk) + 0.45
                ax.add_patch(FancyBboxPatch(
                    (x - 0.19, q - 0.17), 0.38, 0.34,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    fc="white", ec=C_MEASURE, lw=1.1, zorder=4))
                ax.text(x, q, "M", ha="center", va="center",
                        fontsize=7.5, color=C_MEASURE, zorder=5, fontweight="bold")

        if rows > 1:
            ax.text(len(chunk) + 0.1, -0.55, f"({r + 1}/{rows})", fontsize=7,
                    color=C_WIRE, ha="right", va="center")

    g = QCNNGenome.from_dict(gd)
    ttl = title or g.summary()
    fig.suptitle(ttl, fontsize=9.5, color=C_INK, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"        {ttl}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Draw a QCNN genome (Stage 2) as a circuit diagram.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--genome", help="path to a genome JSON (e.g. best_qcnn.json)")
    src.add_argument("--result", help="path to a qcnn_search_result.json")
    ap.add_argument("--gates", type=int, default=None,
                    help="with --result: pick the candidate nearest this gate count")
    ap.add_argument("--list", action="store_true", help="list candidates and exit")
    ap.add_argument("--out", default="qcnn_circuit.png", help=".png or .pdf")
    ap.add_argument("--max-cols", type=int, default=30, help="columns per row before wrapping")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    a = ap.parse_args()

    if a.list:
        if not a.result:
            raise SystemExit("--list requires --result")
        list_candidates(a.result)
        return

    gd = load_genome(a)
    cols, discarded_at, final_active, n = build_qcnn_columns(gd)
    print(f"[genome] {n} qubits -> {len(final_active)} measured, "
          f"{len(cols)} drawing columns")
    draw(gd, cols, discarded_at, final_active, n, a.out,
         max_cols=a.max_cols, title=a.title, dpi=a.dpi)


if __name__ == "__main__":
    main()
