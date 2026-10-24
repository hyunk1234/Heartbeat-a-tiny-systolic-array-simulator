"""Weight-stationary systolic array.

The array is K x N PEs. B (k x n) is preloaded so PE(t, j) holds weight
B[t][j]; loading shifts one row in per cycle, so it costs K cycles per tile.
Then the rows of A stream in from the left: element A[i][t] enters PE row t
at cycle i + t and moves right one column per cycle. Partial sums start at
zero above row 0, move down one row per cycle, and each PE adds
weight * activation. C[i][j] falls out of the bottom of column j at cycle
i + (K - 1) + j, so a tile takes m + K + N - 2 compute cycles.

If k exceeds the array height, the k dimension is split and the partial
results of each pass are summed in an accumulator buffer below the array
(as real accelerators do). If n exceeds the width, columns are tiled too.
"""
from __future__ import annotations

from typing import Optional

from .core import Matrix, Result, Snapshot, check_matmul, copy_grid, grid


def _run_tile(A, B, t0, j0, K, N, m, trace, cycle0, tile):
    W = [[B[t0 + t][j0 + j] for j in range(N)] for t in range(K)]
    x, xt = grid(K, N), grid(K, N)    # activation value / which row of A it belongs to
    ps, pt = grid(K, N), grid(K, N)   # partial sum value / which row of A it belongs to
    out = grid(m, N, 0.0)
    macs = 0
    snaps = []
    T = m + K + N - 2
    for cyc in range(T):
        nx, nxt, nps, npt = grid(K, N), grid(K, N), grid(K, N), grid(K, N)
        active = grid(K, N, False)
        for t in range(K):
            for j in range(N):
                if j == 0:
                    i = cyc - t
                    a, tag = (A[i][t0 + t], i) if 0 <= i < m else (None, None)
                else:
                    a, tag = x[t][j - 1], xt[t][j - 1]
                if a is not None:
                    if t == 0:
                        pin = 0.0
                    else:
                        pin = ps[t - 1][j]
                        assert pt[t - 1][j] == tag, "partial sum and activation out of step"
                    s = pin + W[t][j] * a
                    macs += 1
                    active[t][j] = True
                    nps[t][j], npt[t][j] = s, tag
                    if t == K - 1:
                        out[tag][j] = s
                nx[t][j], nxt[t][j] = a, tag
        x, xt, ps, pt = nx, nxt, nps, npt
        if trace:
            snaps.append(Snapshot(cycle0 + cyc, copy_grid(x), copy_grid(W), copy_grid(ps), active, tile))
    return out, T, macs, snaps


def simulate(A: Matrix, B: Matrix, rows: Optional[int] = None, cols: Optional[int] = None,
             trace: bool = False) -> Result:
    """Multiply A (m x k) by B (k x n) on a rows x cols weight-stationary array.

    rows tiles the k dimension, cols tiles n; they default to k and n.
    """
    m, k, n = check_matmul(A, B)
    rows = rows or k
    cols = cols or n
    if rows < 1 or cols < 1:
        raise ValueError("array dimensions must be positive")
    C = grid(m, n, 0.0)
    cycles = load = macs = tiles = 0
    snaps = []
    for j0 in range(0, n, cols):
        for t0 in range(0, k, rows):
            K, N = min(rows, k - t0), min(cols, n - j0)
            load += K
            out, T, mc, sn = _run_tile(A, B, t0, j0, K, N, m, trace, cycles + load, (t0, j0))
            for i in range(m):
                for j in range(N):
                    C[i][j0 + j] += out[i][j]
            cycles += T
            macs += mc
            tiles += 1
            snaps.extend(sn)
    return Result(C, cycles, load, macs, rows, cols, tiles, "weight-stationary", snaps)
