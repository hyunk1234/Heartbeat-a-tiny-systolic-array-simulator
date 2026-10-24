"""ASCII rendering of the array, one frame per clock cycle."""
from __future__ import annotations

from typing import Optional

from .core import Snapshot

CELL = 10  # inner width of a PE box

LABELS = {
    "output-stationary": ("A→", "B↓", "Σ "),
    "weight-stationary": ("x→", "w ", "Σ↓"),
}


def _fmt(v: Optional[float]) -> str:
    if v is None:
        return "·"
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    s = f"{v:g}" if isinstance(v, float) else str(v)
    return s if len(s) <= CELL - 4 else f"{v:.2e}"[: CELL - 4]


def render(snap: Snapshot, dataflow: str) -> str:
    """Draw every PE as a box. Boxes that did a MAC this cycle get a '=' border."""
    la, lb, lc = LABELS[dataflow]
    rows, cols = len(snap.x), len(snap.x[0])
    out = [f"cycle {snap.cycle:>3}   tile @ {snap.tile}   "
           f"busy PEs: {sum(map(sum, snap.active))}/{rows * cols}"]
    for i in range(rows):
        edge = "+" + "+".join(("=" if snap.active[i][j] else "-") * CELL for j in range(cols)) + "+"
        out.append(edge)
        for lab, g in ((la, snap.x), (lb, snap.y), (lc, snap.acc)):
            out.append("|" + "|".join(f"{lab} {_fmt(g[i][j]):>{CELL - 3}}" for j in range(cols)) + "|")
    out.append("+" + "+".join("-" * CELL for _ in range(cols)) + "+")
    return "\n".join(out)


def render_matrix(M, name: str = "") -> str:
    w = max(len(_fmt(v)) for r in M for v in r)
    lines = [f"{name} ({len(M)}x{len(M[0])})"] if name else []
    lines += ["  [" + " ".join(f"{_fmt(v):>{w}}" for v in r) + "]" for r in M]
    return "\n".join(lines)
