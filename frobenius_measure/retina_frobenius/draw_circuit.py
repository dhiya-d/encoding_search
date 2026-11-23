"""Draw an encoding-circuit genome as a circuit diagram.

Works on any genome JSON written by the search (`best_genome.json`), or on any
candidate inside a `search_result.json` selected by gate count.

Examples
--------
List what is available in a front:
    python draw_circuit.py --result experiments/r_seed0/search_result.json --list

Draw the selected best genome:
    python draw_circuit.py --genome experiments/r_seed0/best_genome.json \
        --out figures/best_circuit.pdf

Draw a specific cheap candidate from the front (nearest gate count is used):
    python draw_circuit.py --result experiments/r_seed0/search_result.json \
        --gates 19 --out figures/circuit_19gates.pdf

Emit quantikz LaTeX for a thesis figure instead of an image:
    python draw_circuit.py --genome experiments/r_seed0/best_genome.json \
        --quantikz --out figures/circuit.tex

Wide circuits wrap automatically; control with --max-cols.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# palette
# ---------------------------------------------------------------------------
C_WIRE = "#41505E"
C_ROT = "#5A4BD6"      # RY
C_ROT_FILL = "#EDEAFB"
C_CNOT = "#0E9384"
C_CZ = "#B45309"
C_CRY = "#C0392B"
C_CRY_FILL = "#F7E4E1"
C_INK = "#17212B"

GATE_COLOR = {"cnot": C_CNOT, "cz": C_CZ, "cry": C_CRY}


# ---------------------------------------------------------------------------
# genome loading + column packing
# ---------------------------------------------------------------------------
def load_genome(args) -> dict:
    if args.genome:
        return json.load(open(args.genome))
    res = json.load(open(args.result))
    cands = res.get("candidates") or res.get("pareto") or []
    if not cands:
        raise SystemExit("no candidates found in result file")
    if args.gates is None:
        w = res.get("winner") or res.get("knee_genome")
        if isinstance(w, dict) and "genome" in w:
            return w["genome"]
        if isinstance(w, dict) and "layers" in w:
            return w
        return cands[-1]["genome"]
    best = min(cands, key=lambda c: abs(c["gates"] - args.gates))
    print(f"[pick] nearest to {args.gates} gates -> {best['gates']} gates, "
          f"frob={best.get('frob', float('nan')):.4f}")
    return best["genome"]


def list_candidates(path):
    res = json.load(open(path))
    cands = res.get("candidates") or res.get("pareto") or []
    print(f"{len(cands)} candidates in {path}\n")
    print(f"  {'gates':>6} {'frob':>8} {'cry':>5} {'params':>7} {'depth':>6}")
    for c in sorted(cands, key=lambda x: x["gates"]):
        i = c.get("info", {})
        print(f"  {c['gates']:>6} {c.get('frob', float('nan')):>8.4f} "
              f"{i.get('cry', '-'):>5} {i.get('params', '-'):>7} {i.get('depth', '-'):>6}")


def stats(g):
    n = g["n_qubits"]
    rot = sum(1 for l in g["layers"] if l["type"] == "rot")
    pairs = [tuple(p) for l in g["layers"] if l["type"] == "ent" for p in l["pairs"]]
    cry = sum(1 for p in pairs if p[2] == "cry")
    return {"n": n, "rot_layers": rot, "gates": len(pairs), "cry": cry,
            "cnot": sum(1 for p in pairs if p[2] == "cnot"),
            "cz": sum(1 for p in pairs if p[2] == "cz"),
            "params": n * rot + cry, "depth": len(g["layers"])}


def build_columns(g):
    """Flatten the genome into drawing columns.

    A rotation layer is one column of RY boxes on every wire. Gates inside an
    entangling layer are packed greedily into columns so that no two gates in a
    column share any wire in their span -- this keeps wide circuits readable
    without changing the circuit's meaning (gates on disjoint wires commute).
    """
    n = g["n_qubits"]
    cols = []
    for layer in g["layers"]:
        if layer["type"] == "rot":
            cols.append({"kind": "rot", "wires": list(range(n))})
            continue
        packed = []          # list of (occupied_set, [gates])
        for p in layer["pairs"]:
            c, t, gate = int(p[0]), int(p[1]), p[2]
            span = set(range(min(c, t), max(c, t) + 1))
            placed = False
            for occ, items in packed:
                if not (occ & span):
                    occ |= span
                    items.append((c, t, gate))
                    placed = True
                    break
            if not placed:
                packed.append((set(span), [(c, t, gate)]))
        for _, items in packed:
            cols.append({"kind": "ent", "gates": items})
    return cols


# ---------------------------------------------------------------------------
# matplotlib rendering
# ---------------------------------------------------------------------------
def draw(g, cols, out, max_cols=40, title=None, dpi=200, style="default"):
    """style="default": this project's colour-coded convention (violet RY,
    per-gate-type colour). style="standard": plain textbook/Qiskit-style
    black-and-white gate-circuit convention -- no colour coding, filled dots
    for controls, a circle-plus for CNOT targets, boxes for CZ/CRY targets."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, FancyBboxPatch

    standard = (style == "standard")
    ink = "#000000" if standard else C_INK
    wire = "#000000" if standard else C_WIRE
    rot_c, rot_fill = ("#000000", "#FFFFFF") if standard else (C_ROT, C_ROT_FILL)
    cry_fill = "#FFFFFF" if standard else C_CRY_FILL

    def gate_color(gate):
        return "#000000" if standard else GATE_COLOR[gate]

    n = g["n_qubits"]
    chunks = [cols[i:i + max_cols] for i in range(0, len(cols), max_cols)] or [[]]
    rows = len(chunks)

    col_w, wire_h = 0.62, 0.52
    fig_w = max(6.0, 1.6 + col_w * min(len(cols), max_cols))
    fig_h = rows * (n * wire_h + 1.0) + 0.6
    fig, axes = plt.subplots(rows, 1, figsize=(fig_w, fig_h), squeeze=False)

    for r, chunk in enumerate(chunks):
        ax = axes[r][0]
        # use a COMMON x-extent across all rows, otherwise a short final chunk
        # gets stretched and the control dots render as ellipses
        ncol = max(len(c) for c in chunks)
        ax.set_xlim(-1.25, ncol + 0.35)
        ax.set_ylim(-0.7, n - 0.3)
        ax.invert_yaxis()
        ax.axis("off")

        wire_end = len(chunk) + 0.1
        for q in range(n):
            ax.plot([-0.35, wire_end], [q, q], color=wire, lw=0.9, zorder=1)
            ax.text(-0.55, q, f"$q_{{{q}}}$", ha="right", va="center",
                    fontsize=8.5, color=ink)

        for ci, col in enumerate(chunk):
            x = ci + 0.35
            if col["kind"] == "rot":
                for q in col["wires"]:
                    ax.add_patch(FancyBboxPatch(
                        (x - 0.19, q - 0.17), 0.38, 0.34,
                        boxstyle="round,pad=0.02,rounding_size=0.05",
                        fc=rot_fill, ec=rot_c, lw=0.9, zorder=3))
                    ax.text(x, q, "RY", ha="center", va="center",
                            fontsize=6.2, color=rot_c, zorder=4, fontweight="bold")
            else:
                for (c, t, gate) in col["gates"]:
                    col_c = gate_color(gate)
                    ax.plot([x, x], [c, t], color=col_c, lw=1.2, zorder=2)
                    ax.add_patch(Circle((x, c), 0.075, fc=col_c, ec=col_c, zorder=4))
                    if gate == "cnot":
                        ax.add_patch(Circle((x, t), 0.15, fc="white", ec=col_c,
                                            lw=1.2, zorder=4))
                        ax.plot([x - 0.15, x + 0.15], [t, t], color=col_c, lw=1.1, zorder=5)
                        ax.plot([x, x], [t - 0.15, t + 0.15], color=col_c, lw=1.1, zorder=5)
                    elif gate == "cz":
                        ax.add_patch(Circle((x, t), 0.075, fc=col_c, ec=col_c, zorder=4))
                    else:  # cry
                        ax.add_patch(FancyBboxPatch(
                            (x - 0.19, t - 0.17), 0.38, 0.34,
                            boxstyle="round,pad=0.02,rounding_size=0.05",
                            fc=cry_fill, ec=col_c, lw=0.9, zorder=4))
                        ax.text(x, t, "RY", ha="center", va="center",
                                fontsize=6.2, color=col_c, zorder=5, fontweight="bold")

        if rows > 1:
            ax.text(len(chunk) + 0.1, -0.55, f"({r + 1}/{rows})", fontsize=7,
                    color=wire, ha="right", va="center")

    s = stats(g)
    ttl = title or (f"{s['n']} qubits   |   {s['gates']} two-qubit gates "
                    f"(CNOT {s['cnot']}, CZ {s['cz']}, CRY {s['cry']})   |   "
                    f"{s['params']} parameters   |   depth {s['depth']}")
    fig.suptitle(ttl, fontsize=9.5, color=ink, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")
    print(f"        {ttl}")


# ---------------------------------------------------------------------------
# quantikz export
# ---------------------------------------------------------------------------
def to_quantikz(g, cols, out):
    """Emit a quantikz tikzcd body. Requires \\usepackage{tikz,quantikz}."""
    n = g["n_qubits"]
    grid = [["\\qw" for _ in cols] for _ in range(n)]
    for ci, col in enumerate(cols):
        if col["kind"] == "rot":
            for q in range(n):
                grid[q][ci] = "\\gate{R_Y}"
        else:
            for (c, t, gate) in col["gates"]:
                if gate == "cnot":
                    grid[c][ci] = f"\\ctrl{{{t - c}}}"
                    grid[t][ci] = "\\targ{}"
                elif gate == "cz":
                    grid[c][ci] = f"\\ctrl{{{t - c}}}"
                    grid[t][ci] = "\\control{}"
                else:
                    grid[c][ci] = f"\\ctrl{{{t - c}}}"
                    grid[t][ci] = "\\gate{R_Y}"
    lines = ["% requires: \\usepackage{tikz} \\usetikzlibrary{quantikz}",
             "\\begin{quantikz}[column sep=4pt, row sep=10pt]"]
    for q in range(n):
        row = f"  \\lstick{{$\\ket{{0}}_{{{q}}}$}} & " + " & ".join(grid[q]) + " & \\qw"
        row += " \\\\" if q < n - 1 else ""
        lines.append(row)
    lines.append("\\end{quantikz}")
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"saved -> {out}  ({len(cols)} columns; consider splitting if too wide)")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Draw an encoding-circuit genome.")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--genome", help="path to a genome JSON (e.g. best_genome.json)")
    src.add_argument("--result", help="path to a search_result.json")
    ap.add_argument("--gates", type=int, default=None,
                    help="with --result: pick the candidate nearest this gate count")
    ap.add_argument("--list", action="store_true", help="list candidates and exit")
    ap.add_argument("--out", default="circuit.png", help=".png/.pdf, or .tex with --quantikz")
    ap.add_argument("--max-cols", type=int, default=40, help="columns per row before wrapping")
    ap.add_argument("--title", default=None)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--quantikz", action="store_true", help="emit LaTeX instead of an image")
    ap.add_argument("--style", default="default", choices=("default", "standard"),
                    help="default: this project's colour-coded convention. "
                         "standard: plain black/white textbook/Qiskit-style diagram")
    a = ap.parse_args()

    if a.list:
        if not a.result:
            raise SystemExit("--list requires --result")
        list_candidates(a.result)
        return

    g = load_genome(a)
    cols = build_columns(g)
    s = stats(g)
    print(f"[genome] {s['n']} qubits, depth {s['depth']}, {s['gates']} 2q gates "
          f"(cnot {s['cnot']}, cz {s['cz']}, cry {s['cry']}), {s['params']} params "
          f"-> {len(cols)} drawing columns")

    if a.quantikz:
        to_quantikz(g, cols, a.out)
    else:
        draw(g, cols, a.out, max_cols=a.max_cols, title=a.title, dpi=a.dpi, style=a.style)


if __name__ == "__main__":
    main()