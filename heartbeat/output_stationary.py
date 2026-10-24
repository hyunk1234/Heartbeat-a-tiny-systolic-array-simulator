"""Output-stationary systolic array.

Each PE(i, j) owns one output element C[i][j]. Row i of A enters from the
left edge delayed by i cycles; column j of B enters from the top edge
delayed by j cycles. Every cycle each PE multiplies the A value arriving
from its left with the B value arriving from above, adds the product to its
accumulator, and passes A right and B down.

For a tile of r x c PEs and inner dimension k, the last MAC happens at
cycle (k - 1) + (r - 1) + (c - 1), so a tile takes k + r + c - 2 cycles.
Outputs larger than the array are split into tiles run one after another.
"""
from __future__ import annotations

from typing import Optional

from .core import Matrix, Result, Snapshot, check_matmul, copy_grid, grid


def _run_tile(A, B, i0, j0, r, c, k, trace, cycle0, tile):
    x = grid(r, c)          # A value held in each PE's right-going register
    y = grid(r, c)          # B value held in each PE's down-going register
    acc = grid(r, c, 0.0)
    macs = 0
    snaps = []
    T = k + r + c - 2
    for t in range(T):
        nx, ny = grid(r, c), grid(r, c)
        active = grid(r, c, False)
        for i in range(r):
            for j in range(c):
                if j == 0:
                    s = t - i
                    a: Optional[float] = A[i0 + i][s] if 0 <= s < k else None
                else:
                    a = x[i][j - 1]
                if i == 0:
                    s = t - j
                    b: Optional[float] = B[s][j0 + j] if 0 <= s < k else None
                else:
                    b = y[i - 1][j]
                if a is not None and b is not None:
                    acc[i][j] += a * b
                    macs += 1
                    active[i][j] = True
                nx[i][j], ny[i][j] = a, b
        x, y = nx, ny       # all registers latch on the same clock edge
        if trace:
            snaps.append(Snapshot(cycle0 + t, copy_grid(x), copy_grid(y), copy_grid(acc), active, tile))
    return acc, T, macs, snaps


def simulate(A: Matrix, B: Matrix, rows: Optional[int] = None, cols: Optional[int] = None,
             trace: bool = False) -> Result:
    """Multiply A (m x k) by B (k x n) on a rows x cols output-stationary array.

    rows/cols default to m/n (one tile, no tiling).
    """
    m, k, n = check_matmul(A, B)
    rows = rows or m
    cols = cols or n
    if rows < 1 or cols < 1:
        raise ValueError("array dimensions must be positive")
    C = grid(m, n, 0.0)
    cycles = macs = tiles = 0
    snaps = []
    for i0 in range(0, m, rows):
        for j0 in range(0, n, cols):
            r, c = min(rows, m - i0), min(cols, n - j0)
            acc, T, mc, sn = _run_tile(A, B, i0, j0, r, c, k, trace, cycles, (i0, j0))
            for i in range(r):
                for j in range(c):
                    C[i0 + i][j0 + j] = acc[i][j]
            cycles += T
            macs += mc
            tiles += 1
            snaps.extend(sn)
    return Result(C, cycles, 0, macs, rows, cols, tiles, "output-stationary", snaps)
